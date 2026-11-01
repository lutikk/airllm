// Package llama loads Llama-family model configurations and estimates their
// layer-wise memory requirements.
package llama

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

const defaultRopeTheta = 10000.0

// TokenIDs accepts either a single JSON number or an array of numbers, as used
// by different Hugging Face generations of Llama configurations.
type TokenIDs []int64

// UnmarshalJSON implements json.Unmarshaler.
func (ids *TokenIDs) UnmarshalJSON(data []byte) error {
	var single int64
	if err := json.Unmarshal(data, &single); err == nil {
		*ids = TokenIDs{single}
		return nil
	}

	var multiple []int64
	if err := json.Unmarshal(data, &multiple); err != nil {
		return fmt.Errorf("must be an integer or an array of integers")
	}
	*ids = TokenIDs(multiple)
	return nil
}

// Config contains the Hugging Face configuration fields needed by the dense
// Llama-family layer planner, including Gemma and Gemma2.
type Config struct {
	Architectures           []string `json:"architectures"`
	HiddenSize              int      `json:"hidden_size"`
	IntermediateSize        int      `json:"intermediate_size"`
	NumAttentionHeads       int      `json:"num_attention_heads"`
	NumKeyValueHeads        int      `json:"num_key_value_heads"`
	NumHiddenLayers         int      `json:"num_hidden_layers"`
	VocabSize               int      `json:"vocab_size"`
	RMSNormEps              float64  `json:"rms_norm_eps"`
	RopeTheta               float64  `json:"rope_theta"`
	HeadDim                 int      `json:"head_dim"`
	MaxPositionEmbeddings   int      `json:"max_position_embeddings"`
	SlidingWindow           int      `json:"sliding_window"`
	SlidingWindowPattern    int      `json:"sliding_window_pattern"`
	LayerTypes              []string `json:"layer_types"`
	QueryPreAttentionScalar float64  `json:"query_pre_attn_scalar"`
	AttnLogitSoftcapping    float64  `json:"attn_logit_softcapping"`
	FinalLogitSoftcapping   float64  `json:"final_logit_softcapping"`
	TieWordEmbeddings       bool     `json:"tie_word_embeddings"`
	EOSTokenIDs             TokenIDs `json:"eos_token_id"`
}

// LoadConfig reads and validates a Hugging Face config.json file.
func LoadConfig(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("llama: read config %q: %w", path, err)
	}

	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return Config{}, fmt.Errorf("llama: decode config %q: %w", path, err)
	}
	if config.RopeTheta == 0 {
		config.RopeTheta = defaultRopeTheta
	}
	if config.HeadDim == 0 {
		if config.NumAttentionHeads > 0 && config.HiddenSize > 0 {
			config.HeadDim = config.HiddenSize / config.NumAttentionHeads
		}
	}
	if err := config.Validate(); err != nil {
		return Config{}, fmt.Errorf("llama: invalid config %q: %w", path, err)
	}
	return config, nil
}

// Validate rejects architectures and dimensions the layer planner does not
// support. It is also useful for configurations constructed directly in Go.
func (c Config) Validate() error {
	if len(c.Architectures) == 0 {
		return fmt.Errorf("architectures is required")
	}
	supported := false
	for _, architecture := range c.Architectures {
		switch architecture {
		case "LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", "GemmaForCausalLM", "Gemma2ForCausalLM":
			supported = true
		}
	}
	if !supported {
		return fmt.Errorf("unsupported architectures %v: want Llama, Mistral, Qwen2, Gemma, or Gemma2 for causal LM", c.Architectures)
	}

	positiveFields := []struct {
		name  string
		value int
	}{
		{"hidden_size", c.HiddenSize},
		{"intermediate_size", c.IntermediateSize},
		{"num_attention_heads", c.NumAttentionHeads},
		{"num_key_value_heads", c.NumKeyValueHeads},
		{"num_hidden_layers", c.NumHiddenLayers},
		{"vocab_size", c.VocabSize},
		{"head_dim", c.HeadDim},
		{"max_position_embeddings", c.MaxPositionEmbeddings},
	}
	for _, field := range positiveFields {
		if field.value <= 0 {
			return fmt.Errorf("%s must be positive, got %d", field.name, field.value)
		}
	}
	if c.HiddenSize%c.NumAttentionHeads != 0 {
		return fmt.Errorf("hidden_size (%d) must be divisible by num_attention_heads (%d)", c.HiddenSize, c.NumAttentionHeads)
	}
	if !c.IsGemma() && c.HeadDim != c.HiddenSize/c.NumAttentionHeads {
		return fmt.Errorf("head_dim (%d) must equal hidden_size / num_attention_heads (%d)", c.HeadDim, c.HiddenSize/c.NumAttentionHeads)
	}
	if c.HeadDim%2 != 0 {
		return fmt.Errorf("head_dim must be even for rotary embeddings, got %d", c.HeadDim)
	}
	if c.NumKeyValueHeads > c.NumAttentionHeads || c.NumAttentionHeads%c.NumKeyValueHeads != 0 {
		return fmt.Errorf("num_attention_heads (%d) must be divisible by num_key_value_heads (%d)", c.NumAttentionHeads, c.NumKeyValueHeads)
	}
	if c.RMSNormEps <= 0 || math.IsNaN(c.RMSNormEps) || math.IsInf(c.RMSNormEps, 0) {
		return fmt.Errorf("rms_norm_eps must be a finite positive number, got %v", c.RMSNormEps)
	}
	if c.RopeTheta <= 0 || math.IsNaN(c.RopeTheta) || math.IsInf(c.RopeTheta, 0) {
		return fmt.Errorf("rope_theta must be a finite positive number, got %v", c.RopeTheta)
	}
	if c.SlidingWindow < 0 || c.SlidingWindowPattern < 0 {
		return fmt.Errorf("sliding_window and sliding_window_pattern must not be negative")
	}
	if len(c.LayerTypes) != 0 && len(c.LayerTypes) != c.NumHiddenLayers {
		return fmt.Errorf("layer_types length %d must equal num_hidden_layers %d", len(c.LayerTypes), c.NumHiddenLayers)
	}
	for _, layerType := range c.LayerTypes {
		if layerType != "full_attention" && layerType != "sliding_attention" {
			return fmt.Errorf("unsupported layer type %q", layerType)
		}
	}
	for name, value := range map[string]float64{
		"query_pre_attn_scalar":   c.QueryPreAttentionScalar,
		"attn_logit_softcapping":  c.AttnLogitSoftcapping,
		"final_logit_softcapping": c.FinalLogitSoftcapping,
	} {
		if value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("%s must be a finite non-negative number, got %v", name, value)
		}
	}
	if len(c.EOSTokenIDs) == 0 {
		return fmt.Errorf("eos_token_id must contain at least one token ID")
	}
	for _, id := range c.EOSTokenIDs {
		if id < 0 || id >= int64(c.VocabSize) {
			return fmt.Errorf("eos_token_id %d is outside vocabulary size %d", id, c.VocabSize)
		}
	}
	return nil
}

// IsQwen2 reports whether the configuration uses Qwen2 attention projections,
// which include Q/K/V biases. Qwen2.5 uses the same Hugging Face architecture.
func (c Config) IsQwen2() bool {
	return c.hasArchitecture("Qwen2ForCausalLM")
}

// IsGemma reports whether the configuration uses Gemma normalization and
// embedding semantics.
func (c Config) IsGemma() bool {
	return c.IsGemma2() || c.hasArchitecture("GemmaForCausalLM")
}

// IsGemma2 reports whether the four-norm, GeGLU Gemma2 decoder is required.
func (c Config) IsGemma2() bool {
	return c.hasArchitecture("Gemma2ForCausalLM")
}

func (c Config) hasArchitecture(want string) bool {
	for _, architecture := range c.Architectures {
		if architecture == want {
			return true
		}
	}
	return false
}

// LayerSlidingWindow returns the attention window for a layer, or zero for
// full attention. Gemma2 alternates full and sliding layers when no explicit
// layer_types array is present.
func (c Config) LayerSlidingWindow(layer int) int {
	if c.SlidingWindow == 0 {
		return 0
	}
	if len(c.LayerTypes) != 0 {
		if c.LayerTypes[layer] == "sliding_attention" {
			return c.SlidingWindow
		}
		return 0
	}
	if c.IsGemma2() {
		pattern := c.SlidingWindowPattern
		if pattern == 0 {
			pattern = 2
		}
		if layer%pattern != 0 {
			return c.SlidingWindow
		}
		return 0
	}
	return c.SlidingWindow
}

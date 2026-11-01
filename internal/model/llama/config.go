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

// Config contains the Hugging Face configuration fields needed by the Llama
// and Mistral layer planner.
type Config struct {
	Architectures         []string `json:"architectures"`
	HiddenSize            int      `json:"hidden_size"`
	IntermediateSize      int      `json:"intermediate_size"`
	NumAttentionHeads     int      `json:"num_attention_heads"`
	NumKeyValueHeads      int      `json:"num_key_value_heads"`
	NumHiddenLayers       int      `json:"num_hidden_layers"`
	VocabSize             int      `json:"vocab_size"`
	RMSNormEps            float64  `json:"rms_norm_eps"`
	RopeTheta             float64  `json:"rope_theta"`
	HeadDim               int      `json:"head_dim"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`
	SlidingWindow         int      `json:"sliding_window"`
	TieWordEmbeddings     bool     `json:"tie_word_embeddings"`
	EOSTokenIDs           TokenIDs `json:"eos_token_id"`
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
		if architecture == "LlamaForCausalLM" || architecture == "MistralForCausalLM" {
			supported = true
		}
	}
	if !supported {
		return fmt.Errorf("unsupported architectures %v: want LlamaForCausalLM or MistralForCausalLM", c.Architectures)
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
	if c.HeadDim != c.HiddenSize/c.NumAttentionHeads {
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
	if c.SlidingWindow < 0 {
		return fmt.Errorf("sliding_window must not be negative, got %d", c.SlidingWindow)
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

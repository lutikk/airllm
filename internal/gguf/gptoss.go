package gguf

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// GPTOSSMetadata contains the GPT-OSS configuration stored in GGUF metadata.
// Both Ollama's gptoss.* keys and upstream gpt-oss.* keys are accepted.
type GPTOSSMetadata struct {
	Architecture           string
	HiddenSize             int
	NumHeads               int
	NumKVHeads             int
	HeadDim                int
	NumLayers              int
	IntermediateSize       int
	ExpertIntermediateSize int
	NumExperts             int
	NumExpertsPerToken     int
	VocabSize              int
	RMSNormEpsilon         float64
	RoPETheta              float64
	ContextLength          int
	OriginalContextLength  int
	SlidingWindow          int
	EOSTokenID             int64
	BOSTokenID             int64
}

// GPTOSSMetadata extracts and validates GPT-OSS architecture fields.
func (r *Reader) GPTOSSMetadata() (GPTOSSMetadata, error) {
	var result GPTOSSMetadata
	architecture, err := metadataString(r.metadata, "general.architecture")
	if err != nil {
		return GPTOSSMetadata{}, err
	}
	var prefix string
	switch architecture {
	case "gptoss":
		prefix = "gptoss."
	case "gpt-oss":
		prefix = "gpt-oss."
	default:
		return GPTOSSMetadata{}, fmt.Errorf("gguf: unsupported architecture %q: only gptoss and gpt-oss are supported", architecture)
	}
	result.Architecture = architecture

	intFields := []struct {
		suffix      string
		destination *int
		optional    bool
	}{
		{"embedding_length", &result.HiddenSize, false},
		{"attention.head_count", &result.NumHeads, false},
		{"attention.head_count_kv", &result.NumKVHeads, false},
		{"attention.key_length", &result.HeadDim, true},
		{"block_count", &result.NumLayers, false},
		{"feed_forward_length", &result.IntermediateSize, false},
		{"expert_feed_forward_length", &result.ExpertIntermediateSize, false},
		{"expert_count", &result.NumExperts, false},
		{"expert_used_count", &result.NumExpertsPerToken, false},
		{"context_length", &result.ContextLength, false},
		{"rope.scaling.original_context_length", &result.OriginalContextLength, true},
		{"attention.sliding_window", &result.SlidingWindow, true},
	}
	for _, field := range intFields {
		value, present, err := gptossMetadataInt(r.metadata, prefix, field.suffix)
		if err != nil {
			return GPTOSSMetadata{}, err
		}
		if !present {
			if field.optional {
				continue
			}
			return GPTOSSMetadata{}, fmt.Errorf("gguf: missing metadata %q", prefix+field.suffix)
		}
		if value <= 0 || value > int64(maxInt()) {
			return GPTOSSMetadata{}, fmt.Errorf("gguf: metadata %q must be a positive platform-sized integer, got %d", prefix+field.suffix, value)
		}
		*field.destination = int(value)
	}
	if result.HeadDim == 0 {
		result.HeadDim = result.HiddenSize / result.NumHeads
	}
	if result.RMSNormEpsilon, err = metadataFloat(r.metadata, prefix+"attention.layer_norm_rms_epsilon"); err != nil {
		return GPTOSSMetadata{}, err
	}
	if result.RMSNormEpsilon <= 0 || math.IsNaN(result.RMSNormEpsilon) || math.IsInf(result.RMSNormEpsilon, 0) {
		return GPTOSSMetadata{}, fmt.Errorf("gguf: invalid RMS norm epsilon %v", result.RMSNormEpsilon)
	}
	result.RoPETheta = 10000
	if value, ok := r.metadata[prefix+"rope.freq_base"]; ok {
		if result.RoPETheta, err = numberFloat(value); err != nil || result.RoPETheta <= 0 || math.IsNaN(result.RoPETheta) || math.IsInf(result.RoPETheta, 0) {
			return GPTOSSMetadata{}, fmt.Errorf("gguf: invalid %srope.freq_base %v", prefix, value)
		}
	}
	if result.EOSTokenID, err = metadataInt(r.metadata, "tokenizer.ggml.eos_token_id"); err != nil {
		return GPTOSSMetadata{}, err
	}
	if result.BOSTokenID, err = metadataInt(r.metadata, "tokenizer.ggml.bos_token_id"); err != nil {
		return GPTOSSMetadata{}, err
	}
	tokens, ok := r.metadata["tokenizer.ggml.tokens"].([]string)
	if !ok || len(tokens) == 0 {
		return GPTOSSMetadata{}, errors.New("gguf: tokenizer.ggml.tokens must be a non-empty string array")
	}
	result.VocabSize = len(tokens)
	if result.HiddenSize%result.NumHeads != 0 || result.NumKVHeads > result.NumHeads || result.NumHeads%result.NumKVHeads != 0 {
		return GPTOSSMetadata{}, errors.New("gguf: inconsistent GPT-OSS attention head dimensions")
	}
	if result.NumExpertsPerToken > result.NumExperts {
		return GPTOSSMetadata{}, errors.New("gguf: GPT-OSS expert_used_count exceeds expert_count")
	}
	if result.EOSTokenID < 0 || result.EOSTokenID >= int64(result.VocabSize) || result.BOSTokenID < 0 || result.BOSTokenID >= int64(result.VocabSize) {
		return GPTOSSMetadata{}, errors.New("gguf: BOS or EOS token ID is outside the vocabulary")
	}
	return result, nil
}

func gptossMetadataInt(metadata map[string]any, prefix, suffix string) (int64, bool, error) {
	value, ok := metadata[prefix+suffix]
	if !ok {
		return 0, false, nil
	}
	integer, err := numberInt(value)
	if err != nil {
		return 0, true, fmt.Errorf("gguf: metadata %q: %w", prefix+suffix, err)
	}
	return integer, true, nil
}

// GPTOSSTensorName maps a GGUF tensor to its canonical GPT-OSS name. Fused
// attention and expert tensors remain fused instead of being split or copied.
func GPTOSSTensorName(name string) (string, bool) {
	switch name {
	case "token_embd.weight":
		return "model.embed_tokens.weight", true
	case "output_norm.weight":
		return "model.norm.weight", true
	case "output.weight":
		return "lm_head.weight", true
	}
	parts := strings.Split(name, ".")
	if len(parts) != 4 || parts[0] != "blk" {
		return "", false
	}
	if _, err := strconv.ParseUint(parts[1], 10, 31); err != nil {
		return "", false
	}
	mapped, ok := map[string]string{
		"attn_norm.weight":        "input_layernorm.weight",
		"attn_qkv.weight":         "self_attn.qkv_proj.weight",
		"attn_qkv.bias":           "self_attn.qkv_proj.bias",
		"attn_sinks.weight":       "self_attn.sinks",
		"attn_output.weight":      "self_attn.o_proj.weight",
		"attn_output.bias":        "self_attn.o_proj.bias",
		"ffn_norm.weight":         "post_attention_layernorm.weight",
		"ffn_gate_inp.weight":     "mlp.router.weight",
		"ffn_gate_inp.bias":       "mlp.router.bias",
		"ffn_gate_up_exps.weight": "mlp.experts.gate_up_proj",
		"ffn_gate_up_exps.bias":   "mlp.experts.gate_up_proj_bias",
		"ffn_gate_exps.weight":    "mlp.experts.gate_proj",
		"ffn_gate_exps.bias":      "mlp.experts.gate_proj_bias",
		"ffn_up_exps.weight":      "mlp.experts.up_proj",
		"ffn_up_exps.bias":        "mlp.experts.up_proj_bias",
		"ffn_down_exps.weight":    "mlp.experts.down_proj",
		"ffn_down_exps.bias":      "mlp.experts.down_proj_bias",
	}[parts[2]+"."+parts[3]]
	if !ok {
		return "", false
	}
	return "model.layers." + parts[1] + "." + mapped, true
}

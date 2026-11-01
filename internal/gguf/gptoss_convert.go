package gguf

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	runtimeTokenizer "github.com/lutikk/airllm/internal/tokenizer"
)

func convertGPTOSS(ctx context.Context, reader *Reader, destDir string) error {
	metadata, err := reader.GPTOSSMetadata()
	if err != nil {
		return err
	}
	tokens, merges, err := gptossTokenizerParts(reader.metadata)
	if err != nil {
		return err
	}
	tensors, tied, err := mapGPTOSSTensors(reader.Tensors())
	if err != nil {
		return err
	}
	if err := validateGPTOSSConvertedShapes(tensors, metadata, tied); err != nil {
		return err
	}
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return fmt.Errorf("gguf: create destination %q: %w", destDir, err)
	}
	if err := writeSafetensors(ctx, reader, tensors, filepath.Join(destDir, "model.safetensors")); err != nil {
		return err
	}
	if err := writeJSONAtomic(filepath.Join(destDir, "config.json"), gptossConfig(metadata, tied)); err != nil {
		return err
	}
	tokenizerPath := filepath.Join(destDir, "tokenizer.json")
	if err := writeJSONAtomic(tokenizerPath, buildGPTOSSTokenizer(tokens, merges, metadata.BOSTokenID, metadata.EOSTokenID)); err != nil {
		return err
	}
	if _, err := runtimeTokenizer.NewFromFile(tokenizerPath); err != nil {
		return fmt.Errorf("gguf: converted GPT-OSS tokenizer is not supported by the runtime tokenizer: %w", err)
	}
	return nil
}

func gptossConfig(metadata GPTOSSMetadata, tied bool) map[string]any {
	originalContext := metadata.OriginalContextLength
	if originalContext == 0 {
		originalContext = metadata.ContextLength
	}
	layerTypes := make([]string, metadata.NumLayers)
	for i := range layerTypes {
		layerTypes[i] = "full_attention"
		if (i+1)%2 == 1 {
			layerTypes[i] = "sliding_attention"
		}
	}
	return map[string]any{
		"architectures":           []string{"GptOssForCausalLM"},
		"model_type":              "gpt_oss",
		"hidden_size":             metadata.HiddenSize,
		"intermediate_size":       metadata.ExpertIntermediateSize,
		"num_attention_heads":     metadata.NumHeads,
		"num_key_value_heads":     metadata.NumKVHeads,
		"head_dim":                metadata.HeadDim,
		"num_hidden_layers":       metadata.NumLayers,
		"num_local_experts":       metadata.NumExperts,
		"num_experts_per_tok":     metadata.NumExpertsPerToken,
		"vocab_size":              metadata.VocabSize,
		"rms_norm_eps":            metadata.RMSNormEpsilon,
		"rope_theta":              metadata.RoPETheta,
		"max_position_embeddings": metadata.ContextLength,
		"sliding_window":          metadata.SlidingWindow,
		"layer_types":             layerTypes,
		"attention_bias":          true,
		"hidden_act":              "silu",
		"swiglu_alpha":            1.702,
		"swiglu_limit":            7.0,
		"tie_word_embeddings":     tied,
		"bos_token_id":            metadata.BOSTokenID,
		"eos_token_id":            metadata.EOSTokenID,
		"rope_parameters": map[string]any{
			"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0,
			"beta_slow": 1.0, "truncate": false,
			"original_max_position_embeddings": originalContext,
		},
	}
}

func mapGPTOSSTensors(source []TensorInfo) ([]convertedTensor, bool, error) {
	result := make([]convertedTensor, 0, len(source))
	seen := make(map[string]string)
	hasOutput := false
	for _, tensor := range source {
		if tensor.Name == "rope_freqs.weight" || strings.HasSuffix(tensor.Name, ".rope_freqs.weight") {
			continue
		}
		name, ok := GPTOSSTensorName(tensor.Name)
		if !ok {
			return nil, false, fmt.Errorf("gguf: no GPT-OSS Hugging Face mapping for tensor %q", tensor.Name)
		}
		if previous, exists := seen[name]; exists {
			return nil, false, fmt.Errorf("gguf: tensors %q and %q both map to %q", previous, tensor.Name, name)
		}
		seen[name] = tensor.Name
		if name == "lm_head.weight" {
			hasOutput = true
		}
		elements, err := elementCount(tensor.Shape)
		if err != nil || elements > math.MaxInt64/2 {
			return nil, false, fmt.Errorf("gguf: tensor %q F16 output size overflows int64", tensor.Name)
		}
		shape := make([]int64, len(tensor.Shape))
		for i, dimension := range tensor.Shape {
			if dimension > math.MaxInt64 {
				return nil, false, fmt.Errorf("gguf: tensor %q shape dimension overflows int64", tensor.Name)
			}
			shape[len(shape)-1-i] = int64(dimension)
		}
		result = append(result, convertedTensor{source: tensor, name: name, shape: shape, size: int64(elements) * 2})
	}
	if len(result) == 0 {
		return nil, false, errors.New("gguf: no convertible tensors")
	}
	sort.Slice(result, func(i, j int) bool { return result[i].name < result[j].name })
	return result, !hasOutput, nil
}

func validateGPTOSSConvertedShapes(tensors []convertedTensor, metadata GPTOSSMetadata, tied bool) error {
	hidden := int64(metadata.HiddenSize)
	headDim := int64(metadata.HeadDim)
	qWidth := int64(metadata.NumHeads) * headDim
	kvWidth := int64(metadata.NumKVHeads) * headDim
	qkvWidth := qWidth + 2*kvWidth
	experts := int64(metadata.NumExperts)
	intermediate := int64(metadata.ExpertIntermediateSize)
	expected := map[string][]int64{
		"model.embed_tokens.weight": {int64(metadata.VocabSize), hidden},
		"model.norm.weight":         {hidden},
	}
	if !tied {
		expected["lm_head.weight"] = []int64{int64(metadata.VocabSize), hidden}
	}
	actual := make(map[string][]int64, len(tensors))
	for _, tensor := range tensors {
		actual[tensor.name] = tensor.shape
	}
	for layer := 0; layer < metadata.NumLayers; layer++ {
		prefix := fmt.Sprintf("model.layers.%d.", layer)
		expected[prefix+"self_attn.qkv_proj.weight"] = []int64{qkvWidth, hidden}
		expected[prefix+"self_attn.qkv_proj.bias"] = []int64{qkvWidth}
		expected[prefix+"self_attn.sinks"] = []int64{int64(metadata.NumHeads)}
		expected[prefix+"self_attn.o_proj.weight"] = []int64{hidden, qWidth}
		expected[prefix+"self_attn.o_proj.bias"] = []int64{hidden}
		expected[prefix+"input_layernorm.weight"] = []int64{hidden}
		expected[prefix+"post_attention_layernorm.weight"] = []int64{hidden}
		expected[prefix+"mlp.router.weight"] = []int64{experts, hidden}
		expected[prefix+"mlp.router.bias"] = []int64{experts}
		if _, fused := actual[prefix+"mlp.experts.gate_up_proj"]; fused {
			expected[prefix+"mlp.experts.gate_up_proj"] = []int64{experts, 2 * intermediate, hidden}
			expected[prefix+"mlp.experts.gate_up_proj_bias"] = []int64{experts, 2 * intermediate}
		} else {
			expected[prefix+"mlp.experts.gate_proj"] = []int64{experts, intermediate, hidden}
			expected[prefix+"mlp.experts.gate_proj_bias"] = []int64{experts, intermediate}
			expected[prefix+"mlp.experts.up_proj"] = []int64{experts, intermediate, hidden}
			expected[prefix+"mlp.experts.up_proj_bias"] = []int64{experts, intermediate}
		}
		expected[prefix+"mlp.experts.down_proj"] = []int64{experts, hidden, intermediate}
		expected[prefix+"mlp.experts.down_proj_bias"] = []int64{experts, hidden}
	}
	for name, shape := range expected {
		got, ok := actual[name]
		if !ok {
			return fmt.Errorf("gguf: required tensor %q is missing", name)
		}
		if !sameShape(got, shape) {
			return fmt.Errorf("gguf: tensor %q has shape %v, want %v", name, got, shape)
		}
		delete(actual, name)
	}
	if len(actual) != 0 {
		names := make([]string, 0, len(actual))
		for name := range actual {
			names = append(names, name)
		}
		sort.Strings(names)
		return fmt.Errorf("gguf: tensors outside configured GPT-OSS architecture: %v", names)
	}
	return nil
}

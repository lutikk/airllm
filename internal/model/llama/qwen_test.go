package llama

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

const tinyQwenConfig = `{
  "architectures": ["Qwen2ForCausalLM"],
  "model_type": "qwen2",
  "hidden_size": 896,
  "intermediate_size": 4864,
  "num_attention_heads": 14,
  "num_key_value_heads": 2,
  "num_hidden_layers": 24,
  "vocab_size": 151936,
  "rms_norm_eps": 0.000001,
  "rope_theta": 1000000,
  "max_position_embeddings": 32768,
  "tie_word_embeddings": true,
  "eos_token_id": [151643, 151645]
}`

func TestLoadConfigQwen25(t *testing.T) {
	config := loadTestConfig(t, tinyQwenConfig)
	if !config.IsQwen2() || config.RopeTheta != 1_000_000 || config.HeadDim != 64 {
		t.Fatalf("unexpected Qwen2 config: %+v", config)
	}
	if !config.TieWordEmbeddings || len(config.EOSTokenIDs) != 2 {
		t.Fatalf("Qwen2 token options were not preserved: %+v", config)
	}
	planner, err := NewLayerPlanner(config, 2)
	if err != nil {
		t.Fatal(err)
	}
	// The Q/K/V bias vectors add 896 + 2*(2*64) elements per layer.
	if got, want := planner.qkvBiasElements(), uint64(1152); got != want {
		t.Fatalf("qkvBiasElements() = %d, want %d", got, want)
	}
}

func TestRuntimeSyntheticQwenEndToEnd(t *testing.T) {
	modelDir := writeSyntheticQwenModel(t)
	runtime, err := Open(modelDir, 8)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close()

	var generated []uint32
	err = runtime.Generate(context.Background(), []uint32{0}, 5, func(token uint32) error {
		generated = append(generated, token)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint32{1, 2}; !reflect.DeepEqual(generated, want) {
		t.Fatalf("generated tokens = %v, want %v", generated, want)
	}
}

func writeSyntheticQwenModel(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	config := `{
	  "architectures": ["Qwen2ForCausalLM"],
	  "hidden_size": 2,
	  "intermediate_size": 2,
	  "num_attention_heads": 1,
	  "num_key_value_heads": 1,
	  "num_hidden_layers": 1,
	  "vocab_size": 3,
	  "rms_norm_eps": 0.00001,
	  "rope_theta": 1000000,
	  "head_dim": 2,
	  "max_position_embeddings": 8,
	  "tie_word_embeddings": false,
	  "eos_token_id": 2
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	tokenizer := `{
	  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
	  "decoder": {"type": "ByteLevel"},
	  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "<|im_end|>": 2}, "merges": []}
	}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokenizer), 0o600); err != nil {
		t.Fatal(err)
	}
	tensors := map[string]syntheticTensor{
		"model.embed_tokens.weight":                      {shape: []int64{3, 2}, values: []float32{1, 0, 0, 1, -1, 0}},
		"model.norm.weight":                              {shape: []int64{2}, values: []float32{1, 1}},
		"lm_head.weight":                                 {shape: []int64{3, 2}, values: []float32{0, 0, 1, 0, 0, 1}},
		"model.layers.0.input_layernorm.weight":          {shape: []int64{2}, values: []float32{1, 1}},
		"model.layers.0.post_attention_layernorm.weight": {shape: []int64{2}, values: []float32{1, 1}},
		"model.layers.0.self_attn.q_proj.bias":           {shape: []int64{2}, values: []float32{0.25, -0.5}},
		"model.layers.0.self_attn.k_proj.bias":           {shape: []int64{2}, values: []float32{-0.25, 0.5}},
		"model.layers.0.self_attn.v_proj.bias":           {shape: []int64{2}, values: []float32{1, 2}},
	}
	zeroTensors := map[string][]int64{
		"model.layers.0.self_attn.q_proj.weight": {2, 2},
		"model.layers.0.self_attn.k_proj.weight": {2, 2},
		"model.layers.0.self_attn.v_proj.weight": {2, 2},
		"model.layers.0.self_attn.o_proj.weight": {2, 2},
		"model.layers.0.mlp.gate_proj.weight":    {2, 2},
		"model.layers.0.mlp.up_proj.weight":      {2, 2},
		"model.layers.0.mlp.down_proj.weight":    {2, 2},
	}
	for name, shape := range zeroTensors {
		tensors[name] = syntheticTensor{shape: shape, values: make([]float32, 4)}
	}
	writeSyntheticSafetensors(t, filepath.Join(dir, "model.safetensors"), tensors)
	return dir
}

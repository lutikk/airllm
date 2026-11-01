package llama

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestLoadConfigGemma2Behaviors(t *testing.T) {
	config := loadTestConfig(t, `{
	  "architectures": ["Gemma2ForCausalLM"],
	  "hidden_size": 4, "intermediate_size": 8,
	  "num_attention_heads": 1, "num_key_value_heads": 1,
	  "num_hidden_layers": 3, "vocab_size": 8,
	  "rms_norm_eps": 0.000001, "rope_theta": 10000, "head_dim": 8,
	  "max_position_embeddings": 8192, "sliding_window": 4096,
	  "query_pre_attn_scalar": 256, "attn_logit_softcapping": 50,
	  "final_logit_softcapping": 30, "tie_word_embeddings": true,
	  "eos_token_id": [1, 7]
	}`)
	if !config.IsGemma() || !config.IsGemma2() || config.HeadDim != 8 {
		t.Fatalf("unexpected Gemma2 config: %+v", config)
	}
	if got := []int{config.LayerSlidingWindow(0), config.LayerSlidingWindow(1), config.LayerSlidingWindow(2)}; !reflect.DeepEqual(got, []int{0, 4096, 0}) {
		t.Fatalf("default Gemma2 windows = %v", got)
	}
	config.LayerTypes = []string{"sliding_attention", "full_attention", "sliding_attention"}
	if got := []int{config.LayerSlidingWindow(0), config.LayerSlidingWindow(1), config.LayerSlidingWindow(2)}; !reflect.DeepEqual(got, []int{4096, 0, 4096}) {
		t.Fatalf("explicit Gemma2 windows = %v", got)
	}
	planner, err := NewLayerPlanner(config, 2)
	if err != nil {
		t.Fatal(err)
	}
	// Q/O are 4x8, K/V are 4x8, MLP is 3x4x8, plus four norm vectors.
	want := uint64(2*4*8+2*4*8+3*4*8+4*4) * 2
	if got := planner.MaxLayerBytes(); got != want {
		t.Fatalf("Gemma2 MaxLayerBytes = %d, want %d", got, want)
	}
}

func TestRuntimeSyntheticGemmaVariants(t *testing.T) {
	for _, architecture := range []string{"GemmaForCausalLM", "Gemma2ForCausalLM"} {
		t.Run(architecture, func(t *testing.T) {
			modelDir := writeSyntheticGemmaModel(t, architecture)
			runtime, err := Open(modelDir, 8)
			if err != nil {
				t.Fatal(err)
			}
			defer runtime.Close()
			embedded := runtime.embedSequence([]uint32{0})
			assertFloat32Close(t, embedded[0], float32(math.Sqrt2))
			assertFloat32Close(t, embedded[1], 0)

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
		})
	}
}

func writeSyntheticGemmaModel(t *testing.T, architecture string) string {
	t.Helper()
	dir := t.TempDir()
	config := `{
	  "architectures": ["` + architecture + `"],
	  "hidden_size": 2, "intermediate_size": 2,
	  "num_attention_heads": 1, "num_key_value_heads": 1,
	  "num_hidden_layers": 1, "vocab_size": 3,
	  "rms_norm_eps": 0.00001, "rope_theta": 10000, "head_dim": 4,
	  "max_position_embeddings": 8, "sliding_window": 4,
	  "query_pre_attn_scalar": 16, "attn_logit_softcapping": 50,
	  "final_logit_softcapping": 30, "tie_word_embeddings": false,
	  "eos_token_id": 2
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	tokenizer := `{"model":{"type":"BPE","vocab":{"zero":0,"one":1,"eos":2},"merges":[]}}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokenizer), 0o600); err != nil {
		t.Fatal(err)
	}
	tensors := map[string]syntheticTensor{
		"model.embed_tokens.weight":              {shape: []int64{3, 2}, values: []float32{1, 0, 0, 1, -1, 0}},
		"model.norm.weight":                      {shape: []int64{2}, values: []float32{0, 0}},
		"lm_head.weight":                         {shape: []int64{3, 2}, values: []float32{0, 0, 1, 0, 0, 1}},
		"model.layers.0.input_layernorm.weight":  {shape: []int64{2}, values: []float32{0, 0}},
		"model.layers.0.self_attn.q_proj.weight": {shape: []int64{4, 2}, values: make([]float32, 8)},
		"model.layers.0.self_attn.k_proj.weight": {shape: []int64{4, 2}, values: make([]float32, 8)},
		"model.layers.0.self_attn.v_proj.weight": {shape: []int64{4, 2}, values: make([]float32, 8)},
		"model.layers.0.self_attn.o_proj.weight": {shape: []int64{2, 4}, values: make([]float32, 8)},
		"model.layers.0.mlp.gate_proj.weight":    {shape: []int64{2, 2}, values: make([]float32, 4)},
		"model.layers.0.mlp.up_proj.weight":      {shape: []int64{2, 2}, values: make([]float32, 4)},
		"model.layers.0.mlp.down_proj.weight":    {shape: []int64{2, 2}, values: make([]float32, 4)},
	}
	if architecture == "Gemma2ForCausalLM" {
		tensors["model.layers.0.post_attention_layernorm.weight"] = syntheticTensor{shape: []int64{2}, values: []float32{0, 0}}
		tensors["model.layers.0.pre_feedforward_layernorm.weight"] = syntheticTensor{shape: []int64{2}, values: []float32{0, 0}}
		tensors["model.layers.0.post_feedforward_layernorm.weight"] = syntheticTensor{shape: []int64{2}, values: []float32{0, 0}}
	} else {
		tensors["model.layers.0.post_attention_layernorm.weight"] = syntheticTensor{shape: []int64{2}, values: []float32{0, 0}}
	}
	writeSyntheticSafetensors(t, filepath.Join(dir, "model.safetensors"), tensors)
	return dir
}

func assertFloat32Close(t *testing.T, got, want float32) {
	t.Helper()
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Fatalf("got %v, want %v", got, want)
	}
}

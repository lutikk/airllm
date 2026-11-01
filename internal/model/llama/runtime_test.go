package llama

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

func TestRuntimeSyntheticEndToEnd(t *testing.T) {
	modelDir := writeSyntheticModel(t)
	runtime, err := Open(modelDir, 8)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := runtime.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	}()

	var generated []uint32
	err = runtime.Generate(context.Background(), []uint32{0}, 5, func(token uint32) error {
		generated = append(generated, token)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	// Prompt embedding [1,0] selects token 1. Feeding token 1 back selects EOS
	// token 2 through its [0,1] embedding, so generation must stop after two.
	if want := []uint32{1, 2}; !reflect.DeepEqual(generated, want) {
		t.Fatalf("generated tokens = %v, want %v", generated, want)
	}
}

type syntheticTensor struct {
	shape  []int64
	values []float32
}

func writeSyntheticModel(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	config := `{
	  "architectures": ["LlamaForCausalLM"],
	  "hidden_size": 2,
	  "intermediate_size": 2,
	  "num_attention_heads": 1,
	  "num_key_value_heads": 1,
	  "num_hidden_layers": 1,
	  "vocab_size": 3,
	  "rms_norm_eps": 0.00001,
	  "rope_theta": 10000,
	  "head_dim": 2,
	  "max_position_embeddings": 8,
	  "tie_word_embeddings": false,
	  "eos_token_id": 2
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	tokenizer := `{
	  "model": {
	    "type": "BPE",
	    "vocab": {"zero": 0, "one": 1, "eos": 2},
	    "merges": []
	  }
	}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokenizer), 0o600); err != nil {
		t.Fatal(err)
	}

	tensors := map[string]syntheticTensor{
		"model.embed_tokens.weight": {shape: []int64{3, 2}, values: []float32{
			1, 0,
			0, 1,
			-1, 0,
		}},
		"model.norm.weight": {shape: []int64{2}, values: []float32{1, 1}},
		"lm_head.weight": {shape: []int64{3, 2}, values: []float32{
			0, 0,
			1, 0,
			0, 1,
		}},
		"model.layers.0.input_layernorm.weight":          {shape: []int64{2}, values: []float32{1, 1}},
		"model.layers.0.post_attention_layernorm.weight": {shape: []int64{2}, values: []float32{1, 1}},
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

func writeSyntheticSafetensors(t *testing.T, path string, tensors map[string]syntheticTensor) {
	t.Helper()
	type headerEntry struct {
		DType       string   `json:"dtype"`
		Shape       []int64  `json:"shape"`
		DataOffsets [2]int64 `json:"data_offsets"`
	}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	header := make(map[string]headerEntry, len(tensors))
	var data []byte
	for _, name := range names {
		tensor := tensors[name]
		start := int64(len(data))
		for _, value := range tensor.values {
			var encoded [4]byte
			binary.LittleEndian.PutUint32(encoded[:], math.Float32bits(value))
			data = append(data, encoded[:]...)
		}
		header[name] = headerEntry{
			DType:       "F32",
			Shape:       tensor.shape,
			DataOffsets: [2]int64{start, int64(len(data))},
		}
	}
	encodedHeader, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	file := make([]byte, 8, 8+len(encodedHeader)+len(data))
	binary.LittleEndian.PutUint64(file, uint64(len(encodedHeader)))
	file = append(file, encodedHeader...)
	file = append(file, data...)
	if err := os.WriteFile(path, file, 0o600); err != nil {
		t.Fatal(err)
	}
}

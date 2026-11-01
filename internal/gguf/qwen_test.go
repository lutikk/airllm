package gguf

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/lutikk/airllm/internal/model/llama"
)

func TestConvertSyntheticQwenAndOpenRuntime(t *testing.T) {
	source := filepath.Join(t.TempDir(), "qwen.gguf")
	writeFixture(t, source, 3, qwenFixtureMetadata(), qwenFixtureTensors())
	destination := filepath.Join(t.TempDir(), "converted")
	if err := Convert(context.Background(), source, destination); err != nil {
		t.Fatal(err)
	}
	config, err := llama.LoadConfig(filepath.Join(destination, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !config.IsQwen2() || !config.TieWordEmbeddings || config.RopeTheta != 1_000_000 {
		t.Fatalf("unexpected converted Qwen config: %+v", config)
	}
	runtime, err := llama.Open(destination, 8)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close()
	ids, err := runtime.Encode("a<|im_end|>", false)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 2 || ids[0] != 1 || ids[1] != 2 {
		t.Fatalf("Qwen BPE IDs = %v, want [1 2]", ids)
	}
	decoded, err := runtime.Decode([]uint32{1}, false)
	if err != nil || decoded != "a" {
		t.Fatalf("Qwen BPE Decode = %q, %v; want a", decoded, err)
	}
}

func TestConvertQwenRequiresQKVBias(t *testing.T) {
	source := filepath.Join(t.TempDir(), "qwen.gguf")
	tensors := qwenFixtureTensors()
	filtered := tensors[:0]
	for _, tensor := range tensors {
		if !strings.HasSuffix(tensor.name, ".bias") {
			filtered = append(filtered, tensor)
		}
	}
	writeFixture(t, source, 3, qwenFixtureMetadata(), filtered)
	err := Convert(context.Background(), source, filepath.Join(t.TempDir(), "converted"))
	if err == nil || !strings.Contains(err.Error(), "q_proj.bias") {
		t.Fatalf("Convert() error = %v, want missing q_proj.bias", err)
	}
}

func qwenFixtureMetadata() map[string]any {
	metadata := map[string]any{
		"general.architecture":                   "qwen2",
		"general.alignment":                      uint32(32),
		"qwen2.embedding_length":                 uint32(8),
		"qwen2.attention.head_count":             uint32(2),
		"qwen2.attention.head_count_kv":          uint32(2),
		"qwen2.block_count":                      uint32(1),
		"qwen2.feed_forward_length":              uint32(16),
		"qwen2.context_length":                   uint32(32),
		"qwen2.attention.layer_norm_rms_epsilon": float32(1e-6),
		"qwen2.rope.freq_base":                   float32(1_000_000),
		"tokenizer.ggml.model":                   "gpt2",
		"tokenizer.ggml.tokens":                  []string{"<|endoftext|>", "a", "<|im_end|>", "Ġ"},
		"tokenizer.ggml.merges":                  []string{},
		"tokenizer.ggml.bos_token_id":            uint32(0),
		"tokenizer.ggml.eos_token_id":            uint32(2),
	}
	return metadata
}

func qwenFixtureTensors() []fixtureTensor {
	tensors := fixtureTensors()
	return append(tensors,
		fixtureTensor{name: "blk.0.attn_q.bias", shape: []uint64{8}, type_: F32, data: fixtureF32(8)},
		fixtureTensor{name: "blk.0.attn_k.bias", shape: []uint64{8}, type_: F32, data: fixtureF32(8)},
		fixtureTensor{name: "blk.0.attn_v.bias", shape: []uint64{8}, type_: F32, data: fixtureF32(8)},
	)
}

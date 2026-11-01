package gguf

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/lutikk/airllm/internal/model/llama"
)

func TestConvertSyntheticGemmaVariants(t *testing.T) {
	for _, architecture := range []string{"gemma", "gemma2"} {
		t.Run(architecture, func(t *testing.T) {
			source := filepath.Join(t.TempDir(), architecture+".gguf")
			writeFixture(t, source, 3, gemmaFixtureMetadata(architecture), gemmaFixtureTensors(architecture))
			destination := filepath.Join(t.TempDir(), "converted")
			if err := Convert(context.Background(), source, destination); err != nil {
				t.Fatal(err)
			}
			config, err := llama.LoadConfig(filepath.Join(destination, "config.json"))
			if err != nil {
				t.Fatal(err)
			}
			if !config.IsGemma() || config.IsGemma2() != (architecture == "gemma2") {
				t.Fatalf("unexpected converted architecture: %+v", config)
			}
			if config.HeadDim != 8 || config.QueryPreAttentionScalar != 256 || !config.TieWordEmbeddings {
				t.Fatalf("Gemma flags were not converted: %+v", config)
			}
			if config.SlidingWindow != 16 || config.SlidingWindowPattern != 2 {
				t.Fatalf("Gemma window flags were not converted: %+v", config)
			}
			if architecture == "gemma2" && (config.AttnLogitSoftcapping != 50 || config.FinalLogitSoftcapping != 30) {
				t.Fatalf("Gemma2 softcaps were not converted: %+v", config)
			}
			runtime, err := llama.Open(destination, 8)
			if err != nil {
				t.Fatal(err)
			}
			if err := runtime.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestConvertGemma2RequiresPostOutputNorms(t *testing.T) {
	source := filepath.Join(t.TempDir(), "gemma2.gguf")
	tensors := gemmaFixtureTensors("gemma2")
	filtered := tensors[:0]
	for _, tensor := range tensors {
		if !strings.Contains(tensor.name, "post_norm") {
			filtered = append(filtered, tensor)
		}
	}
	writeFixture(t, source, 3, gemmaFixtureMetadata("gemma2"), filtered)
	err := Convert(context.Background(), source, filepath.Join(t.TempDir(), "converted"))
	if err == nil || !strings.Contains(err.Error(), "post_attention_layernorm") {
		t.Fatalf("Convert() error = %v, want missing Gemma2 norm", err)
	}
}

func gemmaFixtureMetadata(architecture string) map[string]any {
	metadata := map[string]any{
		"general.architecture":                             architecture,
		"general.alignment":                                uint32(32),
		architecture + ".embedding_length":                 uint32(8),
		architecture + ".attention.head_count":             uint32(2),
		architecture + ".attention.head_count_kv":          uint32(2),
		architecture + ".attention.key_length":             uint32(8),
		architecture + ".attention.value_length":           uint32(8),
		architecture + ".block_count":                      uint32(1),
		architecture + ".feed_forward_length":              uint32(16),
		architecture + ".context_length":                   uint32(32),
		architecture + ".attention.layer_norm_rms_epsilon": float32(1e-6),
		architecture + ".rope.freq_base":                   float32(10000),
		architecture + ".attention.sliding_window":         uint32(16),
		architecture + ".attention.sliding_window_pattern": uint32(2),
		architecture + ".attention.query_pre_attn_scalar":  float32(256),
		"tokenizer.ggml.model":                             "llama",
		"tokenizer.ggml.tokens":                            []string{"<unk>", "<s>", "</s>", "a"},
		"tokenizer.ggml.merges":                            []string{},
		"tokenizer.ggml.unknown_token_id":                  uint32(0),
		"tokenizer.ggml.bos_token_id":                      uint32(1),
		"tokenizer.ggml.eos_token_id":                      uint32(2),
	}
	if architecture == "gemma2" {
		metadata[architecture+".attention.logit_softcapping"] = float32(50)
		metadata[architecture+".final_logit_softcapping"] = float32(30)
	}
	return metadata
}

func gemmaFixtureTensors(architecture string) []fixtureTensor {
	tensors := []fixtureTensor{
		{name: "token_embd.weight", shape: []uint64{8, 4}, type_: F32, data: fixtureF32(32)},
		{name: "output_norm.weight", shape: []uint64{8}, type_: F32, data: fixtureF32(8)},
		{name: "blk.0.attn_norm.weight", shape: []uint64{8}, type_: F32, data: fixtureF32(8)},
		{name: "blk.0.attn_q.weight", shape: []uint64{8, 16}, type_: F32, data: fixtureF32(128)},
		{name: "blk.0.attn_k.weight", shape: []uint64{8, 16}, type_: F32, data: fixtureF32(128)},
		{name: "blk.0.attn_v.weight", shape: []uint64{8, 16}, type_: F32, data: fixtureF32(128)},
		{name: "blk.0.attn_output.weight", shape: []uint64{16, 8}, type_: F32, data: fixtureF32(128)},
		{name: "blk.0.ffn_down.weight", shape: []uint64{16, 8}, type_: F32, data: fixtureF32(128)},
		{name: "blk.0.ffn_gate.weight", shape: []uint64{8, 16}, type_: F32, data: fixtureF32(128)},
		{name: "blk.0.ffn_norm.weight", shape: []uint64{8}, type_: F32, data: fixtureF32(8)},
		{name: "blk.0.ffn_up.weight", shape: []uint64{8, 16}, type_: F32, data: fixtureF32(128)},
	}
	if architecture == "gemma2" {
		tensors = append(tensors,
			fixtureTensor{name: "blk.0.attn_post_norm.weight", shape: []uint64{8}, type_: F32, data: fixtureF32(8)},
			fixtureTensor{name: "blk.0.ffn_post_norm.weight", shape: []uint64{8}, type_: F32, data: fixtureF32(8)},
		)
	}
	return tensors
}

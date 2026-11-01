package gguf

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	runtimeTokenizer "github.com/lutikk/airllm/internal/tokenizer"
	"github.com/lutikk/airllm/internal/weights/safetensors"
)

func TestGPTOSSEncodedSizes(t *testing.T) {
	for _, test := range []struct {
		tensorType TensorType
		shape      []uint64
		want       int64
	}{
		{BF16, []uint64{32}, 64},
		{BF16, []uint64{8, 4}, 64},
		{MXFP4, []uint64{32}, 17},
		{MXFP4, []uint64{32, 2}, 34},
	} {
		got, err := encodedSize(test.tensorType, test.shape)
		if err != nil {
			t.Fatalf("encodedSize(%s, %v): %v", test.tensorType, test.shape, err)
		}
		if got != test.want {
			t.Errorf("encodedSize(%s, %v) = %d, want %d", test.tensorType, test.shape, got, test.want)
		}
	}
	if _, err := encodedSize(MXFP4, []uint64{31}); err == nil {
		t.Fatal("MXFP4 accepted a partial 32-element block")
	}
	if BF16.String() != "BF16" || MXFP4.String() != "MXFP4" {
		t.Fatalf("unexpected type names: %s, %s", BF16, MXFP4)
	}
}

func TestBF16GoldenDecode(t *testing.T) {
	values := []float32{0, float32(math.Copysign(0, -1)), 1, -2.5, 3.1415927, 65504, 131008, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())}
	source := make([]byte, len(values)*2)
	for i, value := range values {
		binary.LittleEndian.PutUint16(source[i*2:], uint16(math.Float32bits(value)>>16))
	}
	decoded := decodeGoldenTensor(t, BF16, source, len(values))
	for i := range values {
		got := float16ToFloat32(binary.LittleEndian.Uint16(decoded[i*2:]))
		bf16 := math.Float32frombits(uint32(binary.LittleEndian.Uint16(source[i*2:])) << 16)
		want := float16ToFloat32(float32ToFloat16(bf16))
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(got)) {
				t.Errorf("element %d = %v, want NaN", i, got)
			}
		} else if got != want {
			t.Errorf("element %d = %v, want %v", i, got, want)
		}
	}
}

func TestMXFP4GoldenAllNibbles(t *testing.T) {
	block := make([]byte, 17)
	block[0] = 127 // E8M0 scale 1
	for i := 0; i < 16; i++ {
		block[1+i] = byte(i) | byte(15-i)<<4
	}
	decoded := decodeGoldenTensor(t, MXFP4, block, 32)
	for i := 0; i < 16; i++ {
		for pair, nibble := range []byte{byte(i), byte(15 - i)} {
			index := i*2 + pair
			got := float16ToFloat32(binary.LittleEndian.Uint16(decoded[index*2:]))
			want := mxfp4Values[nibble]
			if got != want || math.Signbit(float64(got)) != math.Signbit(float64(want)) {
				t.Errorf("nibble %d at element %d = %v (signbit %v), want %v (signbit %v)", nibble, index, got, math.Signbit(float64(got)), want, math.Signbit(float64(want)))
			}
		}
	}
}

func TestMXFP4GoldenScales(t *testing.T) {
	for _, exponent := range []byte{0, 1, 126, 127, 128, 141, 142, 143, 254, 255} {
		block := make([]byte, 17)
		block[0] = exponent
		block[1] = 0x73 // 4 and 6
		decoded := decodeGoldenTensor(t, MXFP4, block, 32)
		for i, nibble := range []byte{3, 7} {
			got := float16ToFloat32(binary.LittleEndian.Uint16(decoded[i*2:]))
			scale := float32(math.Ldexp(1, int(exponent)-127))
			if exponent == 255 {
				scale = float32(math.NaN())
			}
			want := float16ToFloat32(float32ToFloat16(scale * mxfp4Values[nibble]))
			if math.IsNaN(float64(want)) {
				if !math.IsNaN(float64(got)) {
					t.Errorf("scale %d element %d = %v, want NaN", exponent, i, got)
				}
			} else if got != want {
				t.Errorf("scale %d element %d = %v, want %v", exponent, i, got, want)
			}
		}
	}
}

func decodeGoldenTensor(t *testing.T, tensorType TensorType, source []byte, elements int) []byte {
	t.Helper()
	info := TensorInfo{Name: "output.weight", Type: tensorType, Shape: []uint64{uint64(elements)}, ByteSize: int64(len(source))}
	var destination bytes.Buffer
	if err := writeF16Tensor(context.Background(), &destination, info, source); err != nil {
		t.Fatal(err)
	}
	if destination.Len() != elements*2 {
		t.Fatalf("decoded byte length = %d, want %d", destination.Len(), elements*2)
	}
	return destination.Bytes()
}

func TestGPTOSSMetadataDialects(t *testing.T) {
	for _, prefix := range []string{"gptoss.", "gpt-oss."} {
		t.Run(prefix, func(t *testing.T) {
			architecture := "gptoss"
			if prefix == "gpt-oss." {
				architecture = "gpt-oss"
			}
			metadata := map[string]any{
				"general.architecture":        architecture,
				"tokenizer.ggml.tokens":       []string{"a", "b", "c", "d"},
				"tokenizer.ggml.bos_token_id": uint32(1),
				"tokenizer.ggml.eos_token_id": uint32(2),
			}
			values := map[string]any{
				"embedding_length": uint32(64), "attention.head_count": uint32(8),
				"attention.head_count_kv": uint32(2), "block_count": uint32(4),
				"feed_forward_length": uint32(128), "expert_feed_forward_length": uint32(32),
				"expert_count": uint32(16), "expert_used_count": uint32(4),
				"context_length": uint32(4096), "rope.scaling.original_context_length": uint32(1024),
				"attention.sliding_window":         uint32(128),
				"attention.layer_norm_rms_epsilon": float32(1e-5), "rope.freq_base": float32(150000),
			}
			for suffix, value := range values {
				metadata[prefix+suffix] = value
			}
			reader := &Reader{metadata: metadata}
			got, err := reader.GPTOSSMetadata()
			if err != nil {
				t.Fatal(err)
			}
			if got.Architecture != architecture || got.HiddenSize != 64 || got.NumHeads != 8 || got.NumKVHeads != 2 || got.NumLayers != 4 || got.IntermediateSize != 128 || got.ExpertIntermediateSize != 32 || got.NumExperts != 16 || got.NumExpertsPerToken != 4 || got.ContextLength != 4096 || got.OriginalContextLength != 1024 || got.SlidingWindow != 128 || got.VocabSize != 4 || got.RoPETheta != 150000 || got.BOSTokenID != 1 || got.EOSTokenID != 2 {
				t.Fatalf("unexpected metadata: %+v", got)
			}
		})
	}
}

func TestGPTOSSTensorMapping(t *testing.T) {
	mapping := map[string]string{
		"token_embd.weight":             "model.embed_tokens.weight",
		"output_norm.weight":            "model.norm.weight",
		"output.weight":                 "lm_head.weight",
		"blk.3.attn_norm.weight":        "model.layers.3.input_layernorm.weight",
		"blk.3.attn_qkv.weight":         "model.layers.3.self_attn.qkv_proj.weight",
		"blk.3.attn_qkv.bias":           "model.layers.3.self_attn.qkv_proj.bias",
		"blk.3.attn_sinks.weight":       "model.layers.3.self_attn.sinks",
		"blk.3.attn_output.weight":      "model.layers.3.self_attn.o_proj.weight",
		"blk.3.attn_output.bias":        "model.layers.3.self_attn.o_proj.bias",
		"blk.3.ffn_norm.weight":         "model.layers.3.post_attention_layernorm.weight",
		"blk.3.ffn_gate_inp.weight":     "model.layers.3.mlp.router.weight",
		"blk.3.ffn_gate_inp.bias":       "model.layers.3.mlp.router.bias",
		"blk.3.ffn_gate_up_exps.weight": "model.layers.3.mlp.experts.gate_up_proj",
		"blk.3.ffn_gate_up_exps.bias":   "model.layers.3.mlp.experts.gate_up_proj_bias",
		"blk.3.ffn_gate_exps.weight":    "model.layers.3.mlp.experts.gate_proj",
		"blk.3.ffn_gate_exps.bias":      "model.layers.3.mlp.experts.gate_proj_bias",
		"blk.3.ffn_up_exps.weight":      "model.layers.3.mlp.experts.up_proj",
		"blk.3.ffn_up_exps.bias":        "model.layers.3.mlp.experts.up_proj_bias",
		"blk.3.ffn_down_exps.weight":    "model.layers.3.mlp.experts.down_proj",
		"blk.3.ffn_down_exps.bias":      "model.layers.3.mlp.experts.down_proj_bias",
	}
	for source, want := range mapping {
		got, ok := GPTOSSTensorName(source)
		if !ok || got != want {
			t.Errorf("GPTOSSTensorName(%q) = %q, %v; want %q, true", source, got, ok, want)
		}
	}
	for _, source := range []string{"blk.x.attn_qkv.weight", "blk.3.attn_q.weight", "output.bias"} {
		if got, ok := GPTOSSTensorName(source); ok {
			t.Errorf("GPTOSSTensorName(%q) unexpectedly mapped to %q", source, got)
		}
	}
}

func TestPrepareGPTOSSTokenizerWithoutWeights(t *testing.T) {
	source := filepath.Join(t.TempDir(), "tiny-gptoss.gguf")
	writeFixture(t, source, 3, gptossFixtureMetadata(), gptossFixtureTensors())
	destination := filepath.Join(t.TempDir(), "native")
	path, err := PrepareGPTOSSTokenizer(source, destination)
	if err != nil {
		t.Fatal(err)
	}
	if path != filepath.Join(destination, "tokenizer.json") {
		t.Fatalf("tokenizer path = %q", path)
	}
	if _, err := runtimeTokenizer.NewFromFile(path); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(destination)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != "tokenizer.json" {
		t.Fatalf("Prepare created unexpected files: %v", entries)
	}
}

func TestConvertSyntheticGPTOSS(t *testing.T) {
	source := filepath.Join(t.TempDir(), "tiny-gptoss.gguf")
	writeFixture(t, source, 3, gptossFixtureMetadata(), gptossFixtureTensors())
	destination := filepath.Join(t.TempDir(), "converted")
	if err := Convert(context.Background(), source, destination); err != nil {
		t.Fatal(err)
	}

	configData, err := os.ReadFile(filepath.Join(destination, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var config map[string]any
	if err := json.Unmarshal(configData, &config); err != nil {
		t.Fatal(err)
	}
	if config["model_type"] != "gpt_oss" || config["head_dim"] != float64(2) || config["num_local_experts"] != float64(2) || config["num_experts_per_tok"] != float64(1) || config["attention_bias"] != true {
		t.Fatalf("unexpected GPT-OSS config: %#v", config)
	}
	layerTypes, ok := config["layer_types"].([]any)
	if !ok || len(layerTypes) != 1 || layerTypes[0] != "sliding_attention" {
		t.Fatalf("unexpected layer types: %#v", config["layer_types"])
	}

	model, err := safetensors.Open(filepath.Join(destination, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	if got := len(model.Tensors()); got != 18 {
		t.Fatalf("converted tensor count = %d, want 18", got)
	}
	for name, shape := range map[string][]int64{
		"model.layers.0.self_attn.qkv_proj.weight":  {8, 4},
		"model.layers.0.self_attn.qkv_proj.bias":    {8},
		"model.layers.0.self_attn.sinks":            {2},
		"model.layers.0.mlp.router.weight":          {2, 4},
		"model.layers.0.mlp.experts.gate_proj":      {2, 3, 4},
		"model.layers.0.mlp.experts.gate_proj_bias": {2, 3},
		"model.layers.0.mlp.experts.up_proj":        {2, 3, 4},
		"model.layers.0.mlp.experts.down_proj":      {2, 4, 3},
		"model.layers.0.mlp.experts.down_proj_bias": {2, 4},
	} {
		tensor, ok := model.TensorByName(name)
		if !ok || tensor.DType != safetensors.F16 || !sameShape(tensor.Shape, shape) {
			t.Errorf("unexpected tensor %s: %+v, present %v; want F16 %v", name, tensor, ok, shape)
		}
	}

	tokenizerData, err := os.ReadFile(filepath.Join(destination, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	var tokenizer map[string]any
	if err := json.Unmarshal(tokenizerData, &tokenizer); err != nil {
		t.Fatal(err)
	}
	if tokenizer["post_processor"] != nil || tokenizer["normalizer"] != nil {
		t.Fatalf("GPT-OSS tokenizer must not add Llama processing: %#v", tokenizer)
	}
	modelConfig := tokenizer["model"].(map[string]any)
	if modelConfig["type"] != "BPE" || modelConfig["unk_token"] != nil || modelConfig["byte_fallback"] != false {
		t.Fatalf("unexpected GPT-OSS BPE model: %#v", modelConfig)
	}
	preTokenizer := tokenizer["pre_tokenizer"].(map[string]any)
	if preTokenizer["type"] != "Sequence" {
		t.Fatalf("unexpected pre-tokenizer: %#v", preTokenizer)
	}
	parts := preTokenizer["pretokenizers"].([]any)
	split := parts[0].(map[string]any)
	if split["type"] != "Split" || split["pattern"].(map[string]any)["Regex"] != o200kSplitPattern {
		t.Fatalf("unexpected O200K splitter: %#v", split)
	}
	byteLevel := parts[1].(map[string]any)
	if byteLevel["type"] != "ByteLevel" {
		t.Fatalf("missing ByteLevel pre-tokenizer: %#v", byteLevel)
	}
	if strings.Contains(string(tokenizerData), "Metaspace") || strings.Contains(string(tokenizerData), "<unk>") {
		t.Fatal("GPT-OSS tokenizer contains Llama Metaspace or unknown-token behavior")
	}
}

func gptossFixtureMetadata() map[string]any {
	return map[string]any{
		"general.architecture":                        "gptoss",
		"general.alignment":                           uint32(32),
		"gptoss.embedding_length":                     uint32(4),
		"gptoss.attention.head_count":                 uint32(2),
		"gptoss.attention.head_count_kv":              uint32(1),
		"gptoss.attention.key_length":                 uint32(2),
		"gptoss.block_count":                          uint32(1),
		"gptoss.feed_forward_length":                  uint32(3),
		"gptoss.expert_feed_forward_length":           uint32(3),
		"gptoss.expert_count":                         uint32(2),
		"gptoss.expert_used_count":                    uint32(1),
		"gptoss.context_length":                       uint32(128),
		"gptoss.rope.scaling.original_context_length": uint32(32),
		"gptoss.attention.sliding_window":             uint32(16),
		"gptoss.attention.layer_norm_rms_epsilon":     float32(1e-5),
		"gptoss.rope.freq_base":                       float32(150000),
		"tokenizer.ggml.model":                        "gpt2",
		"tokenizer.ggml.tokens":                       []string{"a", "<|start|>", "<|end|>", "Ġb"},
		"tokenizer.ggml.merges":                       []string{},
		"tokenizer.ggml.bos_token_id":                 uint32(1),
		"tokenizer.ggml.eos_token_id":                 uint32(2),
	}
}

func gptossFixtureTensors() []fixtureTensor {
	f32 := func(shape ...uint64) []byte {
		elements, err := elementCount(shape)
		if err != nil {
			panic(err)
		}
		return fixtureF32(int(elements))
	}
	return []fixtureTensor{
		{name: "token_embd.weight", shape: []uint64{4, 4}, type_: F32, data: f32(4, 4)},
		{name: "output_norm.weight", shape: []uint64{4}, type_: F32, data: f32(4)},
		{name: "output.weight", shape: []uint64{4, 4}, type_: F32, data: f32(4, 4)},
		{name: "blk.0.attn_norm.weight", shape: []uint64{4}, type_: F32, data: f32(4)},
		{name: "blk.0.attn_qkv.weight", shape: []uint64{4, 8}, type_: F32, data: f32(4, 8)},
		{name: "blk.0.attn_qkv.bias", shape: []uint64{8}, type_: F32, data: f32(8)},
		{name: "blk.0.attn_sinks.weight", shape: []uint64{2}, type_: F32, data: f32(2)},
		{name: "blk.0.attn_output.weight", shape: []uint64{4, 4}, type_: F32, data: f32(4, 4)},
		{name: "blk.0.attn_output.bias", shape: []uint64{4}, type_: F32, data: f32(4)},
		{name: "blk.0.ffn_norm.weight", shape: []uint64{4}, type_: F32, data: f32(4)},
		{name: "blk.0.ffn_gate_inp.weight", shape: []uint64{4, 2}, type_: F32, data: f32(4, 2)},
		{name: "blk.0.ffn_gate_inp.bias", shape: []uint64{2}, type_: F32, data: f32(2)},
		{name: "blk.0.ffn_gate_exps.weight", shape: []uint64{4, 3, 2}, type_: F32, data: f32(4, 3, 2)},
		{name: "blk.0.ffn_gate_exps.bias", shape: []uint64{3, 2}, type_: F32, data: f32(3, 2)},
		{name: "blk.0.ffn_up_exps.weight", shape: []uint64{4, 3, 2}, type_: F32, data: f32(4, 3, 2)},
		{name: "blk.0.ffn_up_exps.bias", shape: []uint64{3, 2}, type_: F32, data: f32(3, 2)},
		{name: "blk.0.ffn_down_exps.weight", shape: []uint64{3, 4, 2}, type_: F32, data: f32(3, 4, 2)},
		{name: "blk.0.ffn_down_exps.bias", shape: []uint64{4, 2}, type_: F32, data: f32(4, 2)},
	}
}

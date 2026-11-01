package gguf

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/lutikk/airllm/internal/model/llama"
	"github.com/lutikk/airllm/internal/weights/safetensors"
)

type fixtureTensor struct {
	name  string
	shape []uint64
	type_ TensorType
	data  []byte
}

func TestOpenV2V3AndReadOneTensor(t *testing.T) {
	for _, version := range []uint32{2, 3} {
		t.Run(string(rune('0'+version)), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tiny.gguf")
			tensors := fixtureTensors()
			writeFixture(t, path, version, fixtureMetadata(), tensors)
			reader, err := Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer reader.Close()

			metadata, err := reader.LlamaMetadata()
			if err != nil {
				t.Fatal(err)
			}
			if metadata.Architecture != "llama" || metadata.HiddenSize != 8 || metadata.NumHeads != 2 || metadata.NumKVHeads != 2 || metadata.NumLayers != 1 || metadata.IntermediateSize != 16 || metadata.VocabSize != 4 || metadata.ContextLength != 32 || metadata.BOSTokenID != 1 || metadata.EOSTokenID != 2 {
				t.Fatalf("unexpected Llama metadata: %+v", metadata)
			}
			if metadata.RMSNormEpsilon != float64(float32(1e-5)) || metadata.RoPETheta != 10000 {
				t.Fatalf("unexpected float metadata: %+v", metadata)
			}
			info, ok := reader.Tensor("blk.0.attn_k.weight")
			if !ok || info.Type != Q8_0 || len(info.Shape) != 2 || info.Shape[0] != 8 || info.Shape[1] != 8 || info.ByteSize != 68 {
				t.Fatalf("unexpected Q8_0 index entry: %+v, %v", info, ok)
			}
			data, err := reader.ReadTensor(info.Name)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(data, tensors[3].data) {
				t.Fatal("ReadTensor did not return exactly the indexed bytes")
			}
			if _, err := reader.ReadTensor("absent"); err == nil {
				t.Fatal("expected an error for an absent tensor")
			}
		})
	}
}

func TestConvertSyntheticLlamaAndOpenRuntime(t *testing.T) {
	for _, version := range []uint32{2, 3} {
		t.Run(string(rune('0'+version)), func(t *testing.T) {
			source := filepath.Join(t.TempDir(), "tiny.gguf")
			writeFixture(t, source, version, fixtureMetadata(), fixtureTensors())
			destination := filepath.Join(t.TempDir(), "converted")
			if err := Convert(context.Background(), source, destination); err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{"config.json", "model.safetensors", "tokenizer.json"} {
				if _, err := os.Stat(filepath.Join(destination, name)); err != nil {
					t.Fatalf("%s: %v", name, err)
				}
			}
			if _, err := os.Stat(filepath.Join(destination, "model.safetensors.index.json")); !os.IsNotExist(err) {
				t.Fatalf("unexpected sharded index: %v", err)
			}

			config, err := llama.LoadConfig(filepath.Join(destination, "config.json"))
			if err != nil {
				t.Fatal(err)
			}
			if !config.TieWordEmbeddings || config.HeadDim != 4 || config.VocabSize != 4 {
				t.Fatalf("unexpected converted config: %+v", config)
			}
			model, err := safetensors.Open(filepath.Join(destination, "model.safetensors"))
			if err != nil {
				t.Fatal(err)
			}
			defer model.Close()
			if got := len(model.Tensors()); got != 11 {
				t.Fatalf("converted tensor count = %d, want 11", got)
			}
			embedding, ok := model.TensorByName("model.embed_tokens.weight")
			if !ok || embedding.DType != safetensors.F16 || !sameShape(embedding.Shape, []int64{4, 8}) {
				t.Fatalf("unexpected embedding: %+v, %v", embedding, ok)
			}
			assertF16Values(t, model, "model.embed_tokens.weight", []float32{-0.75, -0.5, -0.25})
			assertF16Values(t, model, "model.layers.0.self_attn.q_proj.weight", []float32{-0.75, -0.5, -0.25})
			assertF16Values(t, model, "model.layers.0.self_attn.k_proj.weight", []float32{-1, -0.5, 0})
			assertF16Values(t, model, "model.layers.0.self_attn.v_proj.weight", []float32{-2, 1.75, -1.75, 1.5})

			runtime, err := llama.Open(destination, 8)
			if err != nil {
				t.Fatalf("llama.Open converted model: %v", err)
			}
			if err := runtime.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestConvertWithoutMergesReturnsHonestError(t *testing.T) {
	source := filepath.Join(t.TempDir(), "tiny.gguf")
	metadata := fixtureMetadata()
	delete(metadata, "tokenizer.ggml.merges")
	writeFixture(t, source, 3, metadata, fixtureTensors())
	err := Convert(context.Background(), source, filepath.Join(t.TempDir(), "converted"))
	if err == nil || !strings.Contains(err.Error(), "without GGUF merges is not supported") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestConvertHonorsCancellation(t *testing.T) {
	source := filepath.Join(t.TempDir(), "tiny.gguf")
	writeFixture(t, source, 3, fixtureMetadata(), fixtureTensors())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := Convert(ctx, source, filepath.Join(t.TempDir(), "converted")); err != context.Canceled {
		t.Fatalf("Convert error = %v, want context.Canceled", err)
	}
}

func assertF16Values(t *testing.T, model *safetensors.File, name string, expected []float32) {
	t.Helper()
	data, err := model.ReadTensor(name)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) < len(expected)*2 {
		t.Fatalf("%s is too short", name)
	}
	for i, want := range expected {
		got := float16ToFloat32(binary.LittleEndian.Uint16(data[i*2:]))
		if got != want {
			t.Errorf("%s[%d] = %v, want %v", name, i, got, want)
		}
	}
}

func fixtureMetadata() map[string]any {
	return map[string]any{
		"general.architecture":                   "llama",
		"general.alignment":                      uint32(32),
		"llama.embedding_length":                 uint32(8),
		"llama.attention.head_count":             uint32(2),
		"llama.attention.head_count_kv":          uint32(2),
		"llama.block_count":                      uint32(1),
		"llama.feed_forward_length":              uint32(16),
		"llama.context_length":                   uint32(32),
		"llama.attention.layer_norm_rms_epsilon": float32(1e-5),
		"llama.rope.freq_base":                   float32(10000),
		"tokenizer.ggml.model":                   "llama",
		"tokenizer.ggml.tokens":                  []string{"<unk>", "<s>", "</s>", "a"},
		"tokenizer.ggml.merges":                  []string{},
		"tokenizer.ggml.unknown_token_id":        uint32(0),
		"tokenizer.ggml.bos_token_id":            uint32(1),
		"tokenizer.ggml.eos_token_id":            uint32(2),
	}
}

func fixtureTensors() []fixtureTensor {
	return []fixtureTensor{
		{name: "token_embd.weight", shape: []uint64{8, 4}, type_: F32, data: fixtureF32(32)},
		{name: "output_norm.weight", shape: []uint64{8}, type_: F32, data: fixtureF32(8)},
		{name: "blk.0.attn_norm.weight", shape: []uint64{8}, type_: F32, data: fixtureF32(8)},
		{name: "blk.0.attn_k.weight", shape: []uint64{8, 8}, type_: Q8_0, data: fixtureQ8(64)},
		{name: "blk.0.attn_output.weight", shape: []uint64{8, 8}, type_: F32, data: fixtureF32(64)},
		{name: "blk.0.attn_q.weight", shape: []uint64{8, 8}, type_: F16, data: fixtureF16(64)},
		{name: "blk.0.attn_v.weight", shape: []uint64{8, 8}, type_: Q4_0, data: fixtureQ4(64)},
		{name: "blk.0.ffn_down.weight", shape: []uint64{16, 8}, type_: F32, data: fixtureF32(128)},
		{name: "blk.0.ffn_gate.weight", shape: []uint64{8, 16}, type_: F32, data: fixtureF32(128)},
		{name: "blk.0.ffn_norm.weight", shape: []uint64{8}, type_: F32, data: fixtureF32(8)},
		{name: "blk.0.ffn_up.weight", shape: []uint64{8, 16}, type_: F32, data: fixtureF32(128)},
	}
}

func fixtureF32(elements int) []byte {
	result := make([]byte, elements*4)
	for i := 0; i < elements; i++ {
		value := float32(i%7-3) / 4
		binary.LittleEndian.PutUint32(result[i*4:], math.Float32bits(value))
	}
	return result
}

func fixtureF16(elements int) []byte {
	result := make([]byte, elements*2)
	for i := 0; i < elements; i++ {
		value := float32(i%7-3) / 4
		binary.LittleEndian.PutUint16(result[i*2:], float32ToFloat16(value))
	}
	return result
}

func fixtureQ8(elements int) []byte {
	result := make([]byte, elements/32*34)
	for block := 0; block < elements/32; block++ {
		offset := block * 34
		binary.LittleEndian.PutUint16(result[offset:], float32ToFloat16(0.5))
		for i := 0; i < 32; i++ {
			result[offset+2+i] = byte(int8(i%5 - 2))
		}
	}
	return result
}

func fixtureQ4(elements int) []byte {
	result := make([]byte, elements/32*18)
	for block := 0; block < elements/32; block++ {
		offset := block * 18
		binary.LittleEndian.PutUint16(result[offset:], float32ToFloat16(0.25))
		for i := 0; i < 16; i++ {
			low := byte(i % 16)
			high := byte((15 - i) % 16)
			result[offset+2+i] = low | high<<4
		}
	}
	return result
}

func writeFixture(t *testing.T, path string, version uint32, metadata map[string]any, tensors []fixtureTensor) {
	t.Helper()
	var buffer bytes.Buffer
	buffer.WriteString("GGUF")
	writeLE(t, &buffer, version)
	writeLE(t, &buffer, uint64(len(tensors)))
	writeLE(t, &buffer, uint64(len(metadata)))
	keys := make([]string, 0, len(metadata))
	for key := range metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		writeString(t, &buffer, key)
		writeMetadata(t, &buffer, metadata[key])
	}
	offset := uint64(0)
	for _, tensor := range tensors {
		offset = (offset + 31) / 32 * 32
		writeString(t, &buffer, tensor.name)
		writeLE(t, &buffer, uint32(len(tensor.shape)))
		for _, dimension := range tensor.shape {
			writeLE(t, &buffer, dimension)
		}
		writeLE(t, &buffer, uint32(tensor.type_))
		writeLE(t, &buffer, offset)
		offset += uint64(len(tensor.data))
	}
	for buffer.Len()%32 != 0 {
		buffer.WriteByte(0)
	}
	for _, tensor := range tensors {
		for buffer.Len()%32 != 0 {
			buffer.WriteByte(0)
		}
		buffer.Write(tensor.data)
	}
	if err := os.WriteFile(path, buffer.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeMetadata(t *testing.T, buffer *bytes.Buffer, value any) {
	t.Helper()
	switch value := value.(type) {
	case string:
		writeLE(t, buffer, uint32(valueString))
		writeString(t, buffer, value)
	case uint32:
		writeLE(t, buffer, uint32(valueUint32))
		writeLE(t, buffer, value)
	case float32:
		writeLE(t, buffer, uint32(valueFloat32))
		writeLE(t, buffer, value)
	case []string:
		writeLE(t, buffer, uint32(valueArray))
		writeLE(t, buffer, uint32(valueString))
		writeLE(t, buffer, uint64(len(value)))
		for _, item := range value {
			writeString(t, buffer, item)
		}
	default:
		t.Fatalf("unsupported fixture metadata %T", value)
	}
}

func writeString(t *testing.T, buffer *bytes.Buffer, value string) {
	t.Helper()
	writeLE(t, buffer, uint64(len(value)))
	buffer.WriteString(value)
}

func writeLE(t *testing.T, buffer *bytes.Buffer, value any) {
	t.Helper()
	if err := binary.Write(buffer, binary.LittleEndian, value); err != nil {
		t.Fatal(err)
	}
}

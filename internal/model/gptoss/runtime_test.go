package gptoss

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	"github.com/lutikk/airllm/internal/gguf"
)

func TestRuntimeSyntheticNativeGGUFEndToEnd(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	writeSyntheticGPTOSS(t, path)
	if _, err := gguf.PrepareGPTOSSTokenizer(path, filepath.Dir(path)); err != nil {
		t.Fatal(err)
	}
	runtime, err := Open(path, 8)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := runtime.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	}()

	first, err := runtime.loadDecoderLayer(0)
	if err != nil {
		t.Fatal(err)
	}
	second, err := runtime.loadDecoderLayer(1)
	if err != nil {
		t.Fatal(err)
	}
	if first.slidingWindow != 128 || second.slidingWindow != 0 {
		t.Fatalf("layer windows = %d, %d; want 128, 0", first.slidingWindow, second.slidingWindow)
	}
	if !first.weights.fusedGateUp {
		t.Fatal("synthetic fused gate_up tensor was not detected")
	}
	for _, token := range []uint32{returnTokenID, stopTokenID} {
		if _, ok := runtime.stop[token]; !ok {
			t.Errorf("required stop token %d is not configured", token)
		}
	}
	encoded, err := runtime.Encode("z", false)
	if err != nil || !reflect.DeepEqual(encoded, []uint32{0}) {
		t.Fatalf("Encode() = %v, %v; want [0]", encoded, err)
	}
	decoded, err := runtime.Decode([]uint32{0}, false)
	if err != nil || decoded != "z" {
		t.Fatalf("Decode() = %q, %v; want z", decoded, err)
	}

	var generated []uint32
	err = runtime.Generate(context.Background(), []uint32{0}, 5, func(token uint32) error {
		generated = append(generated, token)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	// Attention is zero and each selected expert adds a 0.25 first-dimension
	// bias. The normalized router weights sum to one, so token 0 still selects
	// token 1, whose transformed residual selects metadata EOS token 2.
	if want := []uint32{1, 2}; !reflect.DeepEqual(generated, want) {
		t.Fatalf("generated tokens = %v, want %v", generated, want)
	}
}

type syntheticTensor struct {
	name   string
	shape  []uint64
	values []float32
}

func writeSyntheticGPTOSS(t *testing.T, path string) {
	t.Helper()
	metadata := map[string]any{
		"general.architecture":                        "gptoss",
		"general.alignment":                           uint32(32),
		"gptoss.embedding_length":                     uint32(2),
		"gptoss.attention.head_count":                 uint32(1),
		"gptoss.attention.head_count_kv":              uint32(1),
		"gptoss.attention.key_length":                 uint32(2),
		"gptoss.block_count":                          uint32(2),
		"gptoss.feed_forward_length":                  uint32(2),
		"gptoss.expert_feed_forward_length":           uint32(2),
		"gptoss.expert_count":                         uint32(4),
		"gptoss.expert_used_count":                    uint32(4),
		"gptoss.context_length":                       uint32(8),
		"gptoss.rope.scaling.original_context_length": uint32(4096),
		"gptoss.attention.sliding_window":             uint32(128),
		"gptoss.attention.layer_norm_rms_epsilon":     float32(1e-5),
		"gptoss.rope.freq_base":                       float32(10000),
		"tokenizer.ggml.model":                        "gpt2",
		"tokenizer.ggml.tokens":                       []string{"z", "one", "eos", "unused"},
		"tokenizer.ggml.merges":                       []string{},
		"tokenizer.ggml.bos_token_id":                 uint32(0),
		"tokenizer.ggml.eos_token_id":                 uint32(2),
	}
	tensors := []syntheticTensor{
		{name: "token_embd.weight", shape: []uint64{2, 4}, values: []float32{1, 0, 0, 1, -1, 0, 0, 0}},
		{name: "output_norm.weight", shape: []uint64{2}, values: []float32{1, 1}},
		{name: "output.weight", shape: []uint64{2, 4}, values: []float32{0, 0, 1, 0, 0, 1, 0, 0}},
	}
	for layer := 0; layer < 2; layer++ {
		prefix := "blk." + string(rune('0'+layer)) + "."
		tensors = append(tensors,
			syntheticTensor{name: prefix + "attn_norm.weight", shape: []uint64{2}, values: []float32{1, 1}},
			syntheticTensor{name: prefix + "attn_qkv.weight", shape: []uint64{2, 6}, values: make([]float32, 12)},
			syntheticTensor{name: prefix + "attn_qkv.bias", shape: []uint64{6}, values: make([]float32, 6)},
			syntheticTensor{name: prefix + "attn_sinks.weight", shape: []uint64{1}, values: make([]float32, 1)},
			syntheticTensor{name: prefix + "attn_output.weight", shape: []uint64{2, 2}, values: make([]float32, 4)},
			syntheticTensor{name: prefix + "attn_output.bias", shape: []uint64{2}, values: make([]float32, 2)},
			syntheticTensor{name: prefix + "ffn_norm.weight", shape: []uint64{2}, values: []float32{1, 1}},
			syntheticTensor{name: prefix + "ffn_gate_inp.weight", shape: []uint64{2, 4}, values: make([]float32, 8)},
			syntheticTensor{name: prefix + "ffn_gate_inp.bias", shape: []uint64{4}, values: make([]float32, 4)},
			syntheticTensor{name: prefix + "ffn_gate_up_exps.weight", shape: []uint64{2, 4, 4}, values: make([]float32, 32)},
			syntheticTensor{name: prefix + "ffn_gate_up_exps.bias", shape: []uint64{4, 4}, values: make([]float32, 16)},
			syntheticTensor{name: prefix + "ffn_down_exps.weight", shape: []uint64{2, 2, 4}, values: make([]float32, 16)},
			syntheticTensor{name: prefix + "ffn_down_exps.bias", shape: []uint64{2, 4}, values: []float32{0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0}},
		)
	}
	writeNativeGGUF(t, path, metadata, tensors)
}

func writeNativeGGUF(t *testing.T, path string, metadata map[string]any, tensors []syntheticTensor) {
	t.Helper()
	var header bytes.Buffer
	header.WriteString("GGUF")
	_ = binary.Write(&header, binary.LittleEndian, uint32(3))
	_ = binary.Write(&header, binary.LittleEndian, uint64(len(tensors)))
	_ = binary.Write(&header, binary.LittleEndian, uint64(len(metadata)))
	keys := make([]string, 0, len(metadata))
	for key := range metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		writeGGUFString(t, &header, key)
		writeGGUFMetadata(t, &header, metadata[key])
	}
	offset := uint64(0)
	for _, tensor := range tensors {
		writeGGUFString(t, &header, tensor.name)
		_ = binary.Write(&header, binary.LittleEndian, uint32(len(tensor.shape)))
		for _, dimension := range tensor.shape {
			_ = binary.Write(&header, binary.LittleEndian, dimension)
		}
		_ = binary.Write(&header, binary.LittleEndian, uint32(0)) // F32
		_ = binary.Write(&header, binary.LittleEndian, offset)
		offset = (offset + uint64(len(tensor.values)*4) + 31) &^ 31
	}
	dataOffset := (header.Len() + 31) &^ 31
	file := make([]byte, dataOffset)
	copy(file, header.Bytes())
	for _, tensor := range tensors {
		for _, value := range tensor.values {
			var encoded [4]byte
			binary.LittleEndian.PutUint32(encoded[:], math.Float32bits(value))
			file = append(file, encoded[:]...)
		}
		for len(file)%32 != 0 {
			file = append(file, 0)
		}
	}
	if err := os.WriteFile(path, file, 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeGGUFString(t *testing.T, destination *bytes.Buffer, value string) {
	t.Helper()
	if err := binary.Write(destination, binary.LittleEndian, uint64(len(value))); err != nil {
		t.Fatal(err)
	}
	destination.WriteString(value)
}

func writeGGUFMetadata(t *testing.T, destination *bytes.Buffer, value any) {
	t.Helper()
	switch value := value.(type) {
	case string:
		_ = binary.Write(destination, binary.LittleEndian, uint32(8))
		writeGGUFString(t, destination, value)
	case uint32:
		_ = binary.Write(destination, binary.LittleEndian, uint32(4))
		_ = binary.Write(destination, binary.LittleEndian, value)
	case float32:
		_ = binary.Write(destination, binary.LittleEndian, uint32(6))
		_ = binary.Write(destination, binary.LittleEndian, value)
	case []string:
		_ = binary.Write(destination, binary.LittleEndian, uint32(9))
		_ = binary.Write(destination, binary.LittleEndian, uint32(8))
		_ = binary.Write(destination, binary.LittleEndian, uint64(len(value)))
		for _, item := range value {
			writeGGUFString(t, destination, item)
		}
	default:
		t.Fatalf("unsupported synthetic metadata type %T", value)
	}
}

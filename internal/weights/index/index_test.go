package index

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/lutikk/airllm/internal/weights/safetensors"
)

type testTensor struct {
	name  string
	dtype string
	shape []int64
	data  []byte
}

func TestOpenModelMultiShard(t *testing.T) {
	dir := t.TempDir()
	writeShard(t, dir, "shard-b.safetensors", []testTensor{
		{name: "model.layers.10.mlp.weight", dtype: "F16", shape: []int64{2}, data: []byte{10, 0, 11, 0}},
		{name: "model.layers.2.attn.weight", dtype: "F32", shape: []int64{1}, data: []byte{2, 0, 0, 0}},
		{name: "lm_head.weight", dtype: "F16", shape: []int64{1}, data: []byte{20, 0}},
	})
	writeShard(t, dir, "shard-a.safetensors", []testTensor{
		{name: "model.embed_tokens.weight", dtype: "F16", shape: []int64{2, 2}, data: []byte{1, 0, 2, 0, 3, 0, 4, 0}},
		{name: "model.norm.weight", dtype: "F32", shape: []int64{1}, data: []byte{0, 0, 128, 63}},
	})
	writeWeightMap(t, dir, map[string]string{
		"model.embed_tokens.weight":  "shard-a.safetensors",
		"model.layers.2.attn.weight": "shard-b.safetensors",
		"model.layers.10.mlp.weight": "shard-b.safetensors",
		"model.norm.weight":          "shard-a.safetensors",
		"lm_head.weight":             "shard-b.safetensors",
	})

	model, err := OpenModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	wantLayers := []string{"embedding", "model.layers.2", "model.layers.10", "model.norm", "lm_head"}
	if got := model.LayerNames(); !reflect.DeepEqual(got, wantLayers) {
		t.Fatalf("LayerNames() = %v, want %v", got, wantLayers)
	}
	layer, err := model.Layer(2)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(layer), 1; got != want {
		t.Fatalf("Layer(2) contains %d tensors, want %d", got, want)
	}
	if got, want := layer[0].Name, "model.layers.10.mlp.weight"; got != want {
		t.Fatalf("Layer(2) tensor = %q, want %q", got, want)
	}

	ref, ok := model.Tensor("model.layers.10.mlp.weight")
	if !ok {
		t.Fatal("indexed tensor not found")
	}
	if ref.Shard != "shard-b.safetensors" || ref.DType != safetensors.F16 || ref.Size != 4 || !reflect.DeepEqual(ref.Shape, []int64{2}) {
		t.Fatalf("unexpected tensor reference: %+v", ref)
	}
	data, err := model.ReadTensor("model.layers.10.mlp.weight")
	if err != nil {
		t.Fatal(err)
	}
	if want := []byte{10, 0, 11, 0}; !reflect.DeepEqual(data, want) {
		t.Fatalf("ReadTensor() = %v, want %v", data, want)
	}

	ref.Shape[0] = 99
	again, _ := model.Tensor(ref.Name)
	if again.Shape[0] != 2 {
		t.Fatal("caller modified internal tensor metadata")
	}
	layer[0].Shape[0] = 88
	layerAgain, _ := model.Layer(2)
	if layerAgain[0].Shape[0] != 2 {
		t.Fatal("caller modified internal layer metadata")
	}

	if _, err := model.Layer(-1); err == nil {
		t.Fatal("Layer accepted a negative index")
	}
	if _, err := model.Layer(len(wantLayers)); err == nil {
		t.Fatal("Layer accepted an out-of-range index")
	}
	if _, err := model.ReadTensor("missing.weight"); err == nil || !strings.Contains(err.Error(), `tensor "missing.weight" not found`) {
		t.Fatalf("missing tensor error = %v", err)
	}
}

func TestOpenModelSingleWithoutLMHead(t *testing.T) {
	dir := t.TempDir()
	writeShard(t, dir, singleFileName, []testTensor{
		{name: "model.embed_tokens.weight", dtype: "U8", shape: []int64{3}, data: []byte{3, 2, 1}},
		{name: "model.layers.0.self_attn.q_proj.weight", dtype: "F32", shape: []int64{1}, data: []byte{0, 0, 0, 64}},
		{name: "model.norm.weight", dtype: "U8", shape: []int64{1}, data: []byte{1}},
	})

	model, err := OpenModel(dir)
	if err != nil {
		t.Fatalf("tied lm_head must not be an error: %v", err)
	}
	defer model.Close()

	wantLayers := []string{"embedding", "model.layers.0", "model.norm"}
	if got := model.LayerNames(); !reflect.DeepEqual(got, wantLayers) {
		t.Fatalf("LayerNames() = %v, want %v", got, wantLayers)
	}
	data, err := model.ReadTensor("model.embed_tokens.weight")
	if err != nil {
		t.Fatal(err)
	}
	if want := []byte{3, 2, 1}; !reflect.DeepEqual(data, want) {
		t.Fatalf("ReadTensor() = %v, want %v", data, want)
	}

	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("second Close() failed: %v", err)
	}
	if _, err := model.ReadTensor("model.embed_tokens.weight"); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("read after Close error = %v", err)
	}
}

func TestOpenModelMissingFiles(t *testing.T) {
	t.Run("model", func(t *testing.T) {
		_, err := OpenModel(t.TempDir())
		if err == nil || !strings.Contains(err.Error(), "no model found") {
			t.Fatalf("OpenModel error = %v", err)
		}
	})

	t.Run("shard", func(t *testing.T) {
		dir := t.TempDir()
		writeWeightMap(t, dir, map[string]string{"model.norm.weight": "missing.safetensors"})
		_, err := OpenModel(dir)
		if err == nil || !strings.Contains(err.Error(), "missing.safetensors") {
			t.Fatalf("OpenModel error = %v", err)
		}
	})
}

func TestOpenModelTensorMissingFromShard(t *testing.T) {
	dir := t.TempDir()
	writeShard(t, dir, "actual.safetensors", []testTensor{
		{name: "model.norm.weight", dtype: "U8", shape: []int64{1}, data: []byte{1}},
	})
	writeShard(t, dir, "other.safetensors", []testTensor{
		{name: "model.embed_tokens.weight", dtype: "U8", shape: []int64{1}, data: []byte{2}},
	})
	writeWeightMap(t, dir, map[string]string{
		"model.norm.weight":         "actual.safetensors",
		"model.embed_tokens.weight": "missing.safetensors",
		"other.weight":              "other.safetensors",
	})

	_, err := OpenModel(dir)
	if err == nil || !strings.Contains(err.Error(), "missing.safetensors") {
		t.Fatalf("OpenModel error = %v", err)
	}
}

func TestOpenModelIndexedTensorAbsentFromHeader(t *testing.T) {
	dir := t.TempDir()
	writeShard(t, dir, "weights.safetensors", []testTensor{
		{name: "model.norm.weight", dtype: "U8", shape: []int64{1}, data: []byte{1}},
	})
	writeWeightMap(t, dir, map[string]string{"lm_head.weight": "weights.safetensors"})

	_, err := OpenModel(dir)
	if err == nil || !strings.Contains(err.Error(), `tensor "lm_head.weight"`) || !strings.Contains(err.Error(), "missing from shard") {
		t.Fatalf("OpenModel error = %v", err)
	}
}

func writeWeightMap(t *testing.T, dir string, weightMap map[string]string) {
	t.Helper()
	contents, err := json.Marshal(map[string]any{"weight_map": weightMap})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, indexFileName), contents, 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeShard(t *testing.T, dir, name string, tensors []testTensor) {
	t.Helper()
	header := make(map[string]any, len(tensors))
	var data []byte
	for _, tensor := range tensors {
		start := len(data)
		data = append(data, tensor.data...)
		header[tensor.name] = map[string]any{
			"dtype":        tensor.dtype,
			"shape":        tensor.shape,
			"data_offsets": []int{start, len(data)},
		}
	}
	encodedHeader, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	contents := make([]byte, 8, 8+len(encodedHeader)+len(data))
	binary.LittleEndian.PutUint64(contents, uint64(len(encodedHeader)))
	contents = append(contents, encodedHeader...)
	contents = append(contents, data...)
	if err := os.WriteFile(filepath.Join(dir, name), contents, 0o600); err != nil {
		t.Fatal(err)
	}
}

package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func writeTestFile(t *testing.T, header string, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "weights.safetensors")
	contents := make([]byte, 8, 8+len(header)+len(data))
	binary.LittleEndian.PutUint64(contents, uint64(len(header)))
	contents = append(contents, header...)
	contents = append(contents, data...)
	if err := os.WriteFile(path, contents, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func makeHeader(t *testing.T, entries map[string]any, padding int) string {
	t.Helper()
	header, err := json.Marshal(entries)
	if err != nil {
		t.Fatal(err)
	}
	return string(header) + strings.Repeat(" ", padding)
}

func TestOpenAndReadTensors(t *testing.T) {
	header := makeHeader(t, map[string]any{
		"__metadata__": map[string]string{"format": "pt"},
		"weight": map[string]any{
			"dtype": "F32", "shape": []int64{2, 2}, "data_offsets": []int64{0, 16},
		},
		"scale": map[string]any{
			"dtype": "F16", "shape": []int64{}, "data_offsets": []int64{16, 18},
		},
		"empty": map[string]any{
			"dtype": "I64", "shape": []int64{0, 100}, "data_offsets": []int64{18, 18},
		},
	}, 7)
	data := []byte{
		0, 0, 128, 63, 0, 0, 0, 64,
		0, 0, 64, 64, 0, 0, 128, 64,
		0, 60,
	}
	path := writeTestFile(t, header, data)

	file, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	tensors := file.Tensors()
	if got, want := len(tensors), 3; got != want {
		t.Fatalf("got %d tensors, want %d", got, want)
	}
	if got, want := []string{tensors[0].Name, tensors[1].Name, tensors[2].Name}, []string{"empty", "scale", "weight"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("tensor order = %v, want %v", got, want)
	}

	weight, ok := file.TensorByName("weight")
	if !ok {
		t.Fatal("weight metadata not found")
	}
	if weight.DType != F32 || !reflect.DeepEqual(weight.Shape, []int64{2, 2}) || weight.ByteSize != 16 {
		t.Fatalf("unexpected weight metadata: %+v", weight)
	}
	if got, want := weight.ByteOffset, int64(8+len(header)); got != want {
		t.Fatalf("weight offset = %d, want %d", got, want)
	}

	got, err := file.ReadTensor("weight")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, data[:16]) {
		t.Fatalf("weight bytes = %v, want %v", got, data[:16])
	}
	scale, err := file.ReadTensor("scale")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scale, data[16:18]) {
		t.Fatalf("scale bytes = %v, want %v", scale, data[16:18])
	}
	empty, err := file.ReadTensor("empty")
	if err != nil {
		t.Fatal(err)
	}
	if len(empty) != 0 {
		t.Fatalf("empty tensor contains %d bytes", len(empty))
	}
	if _, err := file.ReadTensor("missing"); err == nil {
		t.Fatal("ReadTensor accepted a missing name")
	}
	if _, ok := file.TensorByName("__metadata__"); ok {
		t.Fatal("metadata exposed as a tensor")
	}

	weight.Shape[0] = 99
	again, _ := file.TensorByName("weight")
	if again.Shape[0] != 2 {
		t.Fatal("caller modified internal shape metadata")
	}
}

func TestSupportedDTypes(t *testing.T) {
	tests := []struct {
		dtype DType
		size  int64
	}{
		{BOOL, 1}, {U8, 1}, {I8, 1},
		{I16, 2}, {U16, 2}, {F16, 2}, {BF16, 2},
		{I32, 4}, {U32, 4}, {F32, 4},
		{I64, 8}, {U64, 8}, {F64, 8},
	}
	for _, test := range tests {
		t.Run(string(test.dtype), func(t *testing.T) {
			header := makeHeader(t, map[string]any{
				"x": map[string]any{"dtype": test.dtype, "shape": []int64{3}, "data_offsets": []int64{0, 3 * test.size}},
			}, 0)
			file, err := Open(writeTestFile(t, header, make([]byte, 3*test.size)))
			if err != nil {
				t.Fatal(err)
			}
			defer file.Close()
			tensor, _ := file.TensorByName("x")
			if tensor.ByteSize != 3*test.size {
				t.Fatalf("size = %d, want %d", tensor.ByteSize, 3*test.size)
			}
		})
	}
}

func TestMalformedFiles(t *testing.T) {
	validEntry := func() map[string]any {
		return map[string]any{"dtype": "F32", "shape": []int64{1}, "data_offsets": []int64{0, 4}}
	}
	tests := []struct {
		name    string
		header  string
		data    []byte
		wantErr string
	}{
		{name: "invalid JSON", header: "{", wantErr: "decode header"},
		{name: "non-object header", header: "[]", wantErr: "top-level"},
		{name: "trailing JSON", header: "{}{}", wantErr: "unexpected value"},
		{name: "duplicate tensor", header: `{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, data: make([]byte, 4), wantErr: "duplicate tensor"},
		{name: "unsupported dtype", header: makeHeader(t, map[string]any{"x": map[string]any{"dtype": "F8_E4M3", "shape": []int64{1}, "data_offsets": []int64{0, 1}}}, 0), data: make([]byte, 1), wantErr: "unsupported dtype"},
		{name: "missing shape", header: makeHeader(t, map[string]any{"x": map[string]any{"dtype": "F32", "data_offsets": []int64{0, 4}}}, 0), data: make([]byte, 4), wantErr: "missing shape"},
		{name: "missing offsets", header: makeHeader(t, map[string]any{"x": map[string]any{"dtype": "F32", "shape": []int64{1}}}, 0), data: make([]byte, 4), wantErr: "data_offsets"},
		{name: "wrong offset count", header: makeHeader(t, map[string]any{"x": map[string]any{"dtype": "F32", "shape": []int64{1}, "data_offsets": []int64{0}}}, 0), data: make([]byte, 4), wantErr: "exactly two"},
		{name: "negative dimension", header: makeHeader(t, map[string]any{"x": map[string]any{"dtype": "F32", "shape": []int64{-1}, "data_offsets": []int64{0, 0}}}, 0), wantErr: "negative shape"},
		{name: "shape overflow", header: makeHeader(t, map[string]any{"x": map[string]any{"dtype": "F64", "shape": []int64{math.MaxInt64, 2}, "data_offsets": []int64{0, 0}}}, 0), wantErr: "overflows"},
		{name: "negative start", header: makeHeader(t, map[string]any{"x": map[string]any{"dtype": "F32", "shape": []int64{1}, "data_offsets": []int64{-4, 0}}}, 0), wantErr: "invalid data_offsets"},
		{name: "reversed offsets", header: makeHeader(t, map[string]any{"x": map[string]any{"dtype": "F32", "shape": []int64{1}, "data_offsets": []int64{4, 0}}}, 0), wantErr: "invalid data_offsets"},
		{name: "shape size mismatch", header: makeHeader(t, map[string]any{"x": map[string]any{"dtype": "F32", "shape": []int64{2}, "data_offsets": []int64{0, 4}}}, 0), data: make([]byte, 4), wantErr: "shape requires"},
		{name: "range past EOF", header: makeHeader(t, map[string]any{"x": validEntry()}, 0), data: make([]byte, 3), wantErr: "exceeds file size"},
		{name: "overlap", header: makeHeader(t, map[string]any{
			"a": validEntry(),
			"b": map[string]any{"dtype": "F32", "shape": []int64{1}, "data_offsets": []int64{2, 6}},
		}, 0), data: make([]byte, 6), wantErr: "overlap"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := Open(writeTestFile(t, test.header, test.data))
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("Open error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

func TestTruncatedHeader(t *testing.T) {
	path := filepath.Join(t.TempDir(), "short.safetensors")
	if err := os.WriteFile(path, []byte{1, 2, 3}, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(path); err == nil {
		t.Fatal("Open accepted a short prefix")
	}

	contents := make([]byte, 8)
	binary.LittleEndian.PutUint64(contents, 100)
	if err := os.WriteFile(path, contents, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(path); err == nil || !strings.Contains(err.Error(), "header length") {
		t.Fatalf("Open error = %v, want header length error", err)
	}
}

func TestReadAfterClose(t *testing.T) {
	header := makeHeader(t, map[string]any{
		"x": map[string]any{"dtype": "U8", "shape": []int64{1}, "data_offsets": []int64{0, 1}},
	}, 0)
	file, err := Open(writeTestFile(t, header, []byte{42}))
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := file.ReadTensor("x"); err == nil {
		t.Fatal("ReadTensor succeeded after Close")
	}
}

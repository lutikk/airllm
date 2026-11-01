// Package safetensors provides streaming access to tensors in safetensors files.
package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
)

// DType identifies the scalar type stored in a tensor.
type DType string

const (
	BOOL DType = "BOOL"
	U8   DType = "U8"
	I8   DType = "I8"
	I16  DType = "I16"
	U16  DType = "U16"
	F16  DType = "F16"
	BF16 DType = "BF16"
	I32  DType = "I32"
	U32  DType = "U32"
	F32  DType = "F32"
	I64  DType = "I64"
	U64  DType = "U64"
	F64  DType = "F64"
)

var dtypeSizes = map[DType]int64{
	BOOL: 1,
	U8:   1,
	I8:   1,
	I16:  2,
	U16:  2,
	F16:  2,
	BF16: 2,
	I32:  4,
	U32:  4,
	F32:  4,
	I64:  8,
	U64:  8,
	F64:  8,
}

// Tensor describes a tensor. ByteOffset is relative to the beginning of the
// file, and ByteSize is the tensor's exact encoded size.
type Tensor struct {
	Name       string
	DType      DType
	Shape      []int64
	ByteOffset int64
	ByteSize   int64
}

// File is an open safetensors file. Its methods are safe for concurrent use.
type File struct {
	file    *os.File
	tensors map[string]Tensor
}

// Open opens path and validates its header and every tensor range. Tensor data
// is not read until ReadTensor is called.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: open %q: %w", path, err)
	}

	result, err := readHeader(f)
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("safetensors: open %q: %w", path, err)
	}
	return result, nil
}

// Close closes the underlying file. Metadata remains available after Close.
func (f *File) Close() error {
	return f.file.Close()
}

// Tensors returns metadata for all tensors, sorted by name. Returned shapes
// are copies and may be modified by the caller.
func (f *File) Tensors() []Tensor {
	result := make([]Tensor, 0, len(f.tensors))
	for _, tensor := range f.tensors {
		result = append(result, cloneTensor(tensor))
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	return result
}

// TensorByName returns metadata for name. The returned shape is a copy.
func (f *File) TensorByName(name string) (Tensor, bool) {
	tensor, ok := f.tensors[name]
	if !ok {
		return Tensor{}, false
	}
	return cloneTensor(tensor), true
}

// ReadTensor reads only the requested tensor's bytes. The returned data is in
// the file's native little-endian representation.
func (f *File) ReadTensor(name string) ([]byte, error) {
	tensor, ok := f.tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	if tensor.ByteSize > int64(^uint(0)>>1) {
		return nil, fmt.Errorf("safetensors: tensor %q is too large for this platform", name)
	}

	data := make([]byte, int(tensor.ByteSize))
	if len(data) == 0 {
		return data, nil
	}
	n, err := f.file.ReadAt(data, tensor.ByteOffset)
	if err != nil {
		return nil, fmt.Errorf("safetensors: read tensor %q: %w", name, err)
	}
	if n != len(data) {
		return nil, fmt.Errorf("safetensors: read tensor %q: %w", name, io.ErrUnexpectedEOF)
	}
	return data, nil
}

func cloneTensor(tensor Tensor) Tensor {
	tensor.Shape = append([]int64(nil), tensor.Shape...)
	return tensor
}

type rawTensor struct {
	DType       string   `json:"dtype"`
	Shape       *[]int64 `json:"shape"`
	DataOffsets *[]int64 `json:"data_offsets"`
}

func readHeader(f *os.File) (*File, error) {
	stat, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat: %w", err)
	}
	fileSize := stat.Size()
	if fileSize < 8 {
		return nil, errors.New("file is shorter than the 8-byte header prefix")
	}

	var prefix [8]byte
	if _, err := f.ReadAt(prefix[:], 0); err != nil {
		return nil, fmt.Errorf("read header length: %w", err)
	}
	headerLength := binary.LittleEndian.Uint64(prefix[:])
	if headerLength > uint64(fileSize-8) {
		return nil, errors.New("header length exceeds file size")
	}
	if headerLength > math.MaxInt64 || headerLength > uint64(^uint(0)>>1) {
		return nil, errors.New("header is too large")
	}

	header := make([]byte, int(headerLength))
	if _, err := f.ReadAt(header, 8); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	dataOffset := int64(8) + int64(headerLength)
	tensors, err := parseHeader(header, dataOffset, fileSize)
	if err != nil {
		return nil, err
	}
	return &File{file: f, tensors: tensors}, nil
}

func parseHeader(header []byte, dataOffset, fileSize int64) (map[string]Tensor, error) {
	decoder := json.NewDecoder(bytes.NewReader(header))
	token, err := decoder.Token()
	if err != nil {
		return nil, fmt.Errorf("decode header: %w", err)
	}
	if delimiter, ok := token.(json.Delim); !ok || delimiter != '{' {
		return nil, errors.New("decode header: top-level value must be an object")
	}

	tensors := make(map[string]Tensor)
	type tensorRange struct {
		name       string
		start, end int64
	}
	ranges := make([]tensorRange, 0)
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return nil, fmt.Errorf("decode header key: %w", err)
		}
		name, ok := token.(string)
		if !ok {
			return nil, errors.New("decode header: tensor name must be a string")
		}
		var raw json.RawMessage
		if err := decoder.Decode(&raw); err != nil {
			return nil, fmt.Errorf("decode header entry %q: %w", name, err)
		}
		if name == "__metadata__" {
			var metadata any
			if err := json.Unmarshal(raw, &metadata); err != nil {
				return nil, fmt.Errorf("decode metadata: %w", err)
			}
			continue
		}
		if _, exists := tensors[name]; exists {
			return nil, fmt.Errorf("duplicate tensor %q", name)
		}

		var encoded rawTensor
		if err := json.Unmarshal(raw, &encoded); err != nil {
			return nil, fmt.Errorf("tensor %q: %w", name, err)
		}
		tensor, end, err := validateTensor(name, encoded, dataOffset, fileSize)
		if err != nil {
			return nil, err
		}
		tensors[name] = tensor
		ranges = append(ranges, tensorRange{name: name, start: tensor.ByteOffset, end: end})
	}
	if _, err := decoder.Token(); err != nil {
		return nil, fmt.Errorf("decode header: %w", err)
	}
	if err := ensureJSONEnd(decoder); err != nil {
		return nil, err
	}

	sort.Slice(ranges, func(i, j int) bool { return ranges[i].start < ranges[j].start })
	for i := 1; i < len(ranges); i++ {
		if ranges[i].start < ranges[i-1].end {
			return nil, fmt.Errorf("tensor ranges %q and %q overlap", ranges[i-1].name, ranges[i].name)
		}
	}
	return tensors, nil
}

func ensureJSONEnd(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return errors.New("decode header: unexpected value after top-level object")
		}
		return fmt.Errorf("decode header: %w", err)
	}
	return nil
}

func validateTensor(name string, encoded rawTensor, dataOffset, fileSize int64) (Tensor, int64, error) {
	elementSize, ok := dtypeSizes[DType(encoded.DType)]
	if !ok {
		return Tensor{}, 0, fmt.Errorf("tensor %q: unsupported dtype %q", name, encoded.DType)
	}
	if encoded.Shape == nil {
		return Tensor{}, 0, fmt.Errorf("tensor %q: missing shape", name)
	}
	if encoded.DataOffsets == nil || len(*encoded.DataOffsets) != 2 {
		return Tensor{}, 0, fmt.Errorf("tensor %q: data_offsets must contain exactly two integers", name)
	}

	elements := int64(1)
	for _, dimension := range *encoded.Shape {
		if dimension < 0 {
			return Tensor{}, 0, fmt.Errorf("tensor %q: negative shape dimension", name)
		}
		if dimension != 0 && elements > math.MaxInt64/dimension {
			return Tensor{}, 0, fmt.Errorf("tensor %q: shape element count overflows int64", name)
		}
		elements *= dimension
	}
	if elements != 0 && elementSize > math.MaxInt64/elements {
		return Tensor{}, 0, fmt.Errorf("tensor %q: byte size overflows int64", name)
	}
	byteSize := elements * elementSize

	offsets := *encoded.DataOffsets
	start, relativeEnd := offsets[0], offsets[1]
	if start < 0 || relativeEnd < start {
		return Tensor{}, 0, fmt.Errorf("tensor %q: invalid data_offsets [%d, %d]", name, start, relativeEnd)
	}
	if relativeEnd-start != byteSize {
		return Tensor{}, 0, fmt.Errorf("tensor %q: shape requires %d bytes, data_offsets contain %d", name, byteSize, relativeEnd-start)
	}
	if start > math.MaxInt64-dataOffset || relativeEnd > math.MaxInt64-dataOffset {
		return Tensor{}, 0, fmt.Errorf("tensor %q: absolute data offset overflows int64", name)
	}
	absoluteStart := dataOffset + start
	absoluteEnd := dataOffset + relativeEnd
	if absoluteEnd > fileSize {
		return Tensor{}, 0, fmt.Errorf("tensor %q: data range exceeds file size", name)
	}

	return Tensor{
		Name:       name,
		DType:      DType(encoded.DType),
		Shape:      append([]int64(nil), (*encoded.Shape)...),
		ByteOffset: absoluteStart,
		ByteSize:   byteSize,
	}, absoluteEnd, nil
}

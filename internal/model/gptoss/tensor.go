package gptoss

import (
	"fmt"
	"math"
	"os"

	"github.com/lutikk/airllm/internal/backend/cpuref"
	"github.com/lutikk/airllm/internal/gguf"
)

type tensorSource struct {
	reader *gguf.Reader
	file   *os.File
}

func openTensorSource(path string) (*tensorSource, error) {
	reader, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	file, err := os.Open(path)
	if err != nil {
		_ = reader.Close()
		return nil, fmt.Errorf("gptoss: open tensor data: %w", err)
	}
	return &tensorSource{reader: reader, file: file}, nil
}

func (s *tensorSource) Close() error {
	if s == nil {
		return nil
	}
	readerErr := s.reader.Close()
	fileErr := s.file.Close()
	if readerErr != nil {
		return readerErr
	}
	return fileErr
}

func (s *tensorSource) read(name string, shape ...int) ([]float32, error) {
	info, ok := s.reader.Tensor(name)
	if !ok {
		return nil, fmt.Errorf("gptoss: tensor %q not found", name)
	}
	if err := checkShape(info, shape); err != nil {
		return nil, err
	}
	data, err := s.reader.ReadTensor(name)
	if err != nil {
		return nil, err
	}
	values, err := decodeTensor(info.Type, data, elementCount(shape))
	if err != nil {
		return nil, fmt.Errorf("gptoss: decode tensor %q: %w", name, err)
	}
	return values, nil
}

type expertTensor struct {
	name         string
	shape        []int
	experts      int
	expertBytes  int64
	expertValues int
	offset       int64
	type_        gguf.TensorType
}

func (s *tensorSource) expert(name string, experts int, shape ...int) (expertTensor, error) {
	info, ok := s.reader.Tensor(name)
	if !ok {
		return expertTensor{}, fmt.Errorf("gptoss: tensor %q not found", name)
	}
	fullShape := append(append([]int(nil), shape...), experts)
	if err := checkShape(info, fullShape); err != nil {
		return expertTensor{}, err
	}
	if experts <= 0 || info.ByteSize%int64(experts) != 0 {
		return expertTensor{}, fmt.Errorf("gptoss: tensor %q cannot be divided into %d experts", name, experts)
	}
	return expertTensor{
		name:         name,
		shape:        append([]int(nil), shape...),
		experts:      experts,
		expertBytes:  info.ByteSize / int64(experts),
		expertValues: elementCount(shape),
		offset:       info.Offset,
		type_:        info.Type,
	}, nil
}

func (s *tensorSource) readExpert(tensor expertTensor, expert int) ([]float32, error) {
	if expert < 0 || expert >= tensor.experts {
		return nil, fmt.Errorf("gptoss: expert %d outside [0, %d)", expert, tensor.experts)
	}
	if tensor.expertBytes > int64(maxInt()) {
		return nil, fmt.Errorf("gptoss: expert tensor %q is too large", tensor.name)
	}
	data := make([]byte, int(tensor.expertBytes))
	offset := tensor.offset + int64(expert)*tensor.expertBytes
	if _, err := s.file.ReadAt(data, offset); err != nil {
		return nil, fmt.Errorf("gptoss: read expert %d from %q: %w", expert, tensor.name, err)
	}
	values, err := decodeTensor(tensor.type_, data, tensor.expertValues)
	if err != nil {
		return nil, fmt.Errorf("gptoss: decode expert %d from %q: %w", expert, tensor.name, err)
	}
	return values, nil
}

func checkShape(info gguf.TensorInfo, shape []int) error {
	if len(info.Shape) != len(shape) {
		return fmt.Errorf("gptoss: tensor %q shape %v, want GGUF shape %v", info.Name, info.Shape, shape)
	}
	for i, dimension := range shape {
		if dimension <= 0 || info.Shape[i] != uint64(dimension) {
			return fmt.Errorf("gptoss: tensor %q shape %v, want GGUF shape %v", info.Name, info.Shape, shape)
		}
	}
	return nil
}

func elementCount(shape []int) int {
	count := 1
	for _, dimension := range shape {
		count *= dimension
	}
	return count
}

func maxInt() int {
	return int(^uint(0) >> 1)
}

func decodeTensor(type_ gguf.TensorType, data []byte, elements int) ([]float32, error) {
	switch type_ {
	case gguf.F32:
		return cpuref.DecodeF32LE(data, elements)
	case gguf.F16:
		return cpuref.DecodeF16LE(data, elements)
	case gguf.BF16:
		return cpuref.DecodeBF16LE(data, elements)
	case gguf.MXFP4:
		if elements%32 != 0 || len(data) != elements/32*17 {
			return nil, fmt.Errorf("MXFP4 byte length %d does not match %d elements", len(data), elements)
		}
		values := make([]float32, elements)
		for block := 0; block < elements/32; block++ {
			encoded := data[block*17 : (block+1)*17]
			scale := float32(math.Ldexp(1, int(encoded[0])-127))
			if encoded[0] == 0xff {
				scale = float32(math.NaN())
			}
			for i, packed := range encoded[1:] {
				values[block*32+i*2] = scale * mxfp4Values[packed&0x0f]
				values[block*32+i*2+1] = scale * mxfp4Values[packed>>4]
			}
		}
		return values, nil
	default:
		return nil, fmt.Errorf("unsupported tensor type %s", type_)
	}
}

var mxfp4Values = [16]float32{
	0, 0.5, 1, 1.5, 2, 3, 4, 6,
	float32(math.Copysign(0, -1)), -0.5, -1, -1.5, -2, -3, -4, -6,
}

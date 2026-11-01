package cpuref

import (
	"encoding/binary"
	"fmt"
	"math"
)

// DecodeF16LE decodes IEEE 754 binary16 values from little-endian bytes.
// The shape product must exactly equal the number of encoded elements.
func DecodeF16LE(data []byte, shape ...int) ([]float32, error) {
	elements, err := shapeElements(shape)
	if err != nil {
		return nil, err
	}
	if len(data) != elements*2 {
		return nil, fmt.Errorf("f16 byte length %d, want %d for shape %v", len(data), elements*2, shape)
	}
	output := make([]float32, elements)
	for i := range output {
		output[i] = halfToFloat32(binary.LittleEndian.Uint16(data[i*2 : i*2+2]))
	}
	return output, nil
}

// DecodeBF16LE decodes bfloat16 values from little-endian bytes.
// The shape product must exactly equal the number of encoded elements.
func DecodeBF16LE(data []byte, shape ...int) ([]float32, error) {
	elements, err := shapeElements(shape)
	if err != nil {
		return nil, err
	}
	if len(data) != elements*2 {
		return nil, fmt.Errorf("bf16 byte length %d, want %d for shape %v", len(data), elements*2, shape)
	}
	output := make([]float32, elements)
	for i := range output {
		bits := uint32(binary.LittleEndian.Uint16(data[i*2:i*2+2])) << 16
		output[i] = math.Float32frombits(bits)
	}
	return output, nil
}

// DecodeF32LE decodes IEEE 754 binary32 values from little-endian bytes.
// The shape product must exactly equal the number of encoded elements.
func DecodeF32LE(data []byte, shape ...int) ([]float32, error) {
	elements, err := shapeElements(shape)
	if err != nil {
		return nil, err
	}
	if len(data) != elements*4 {
		return nil, fmt.Errorf("f32 byte length %d, want %d for shape %v", len(data), elements*4, shape)
	}
	output := make([]float32, elements)
	for i := range output {
		bits := binary.LittleEndian.Uint32(data[i*4 : i*4+4])
		output[i] = math.Float32frombits(bits)
	}
	return output, nil
}

func shapeElements(shape []int) (int, error) {
	if len(shape) == 0 {
		return 0, fmt.Errorf("shape must contain at least one dimension")
	}
	maxInt := int(^uint(0) >> 1)
	elements := 1
	for _, dimension := range shape {
		if dimension <= 0 {
			return 0, fmt.Errorf("shape dimensions must be positive: %v", shape)
		}
		if elements > maxInt/dimension {
			return 0, fmt.Errorf("shape element count overflows int: %v", shape)
		}
		elements *= dimension
	}
	return elements, nil
}

func halfToFloat32(half uint16) float32 {
	sign := uint32(half&0x8000) << 16
	exponent := (half >> 10) & 0x1f
	mantissa := half & 0x03ff

	if exponent == 0x1f {
		return math.Float32frombits(sign | 0x7f800000 | uint32(mantissa)<<13)
	}

	var value float64
	if exponent == 0 {
		value = math.Ldexp(float64(mantissa), -24)
	} else {
		value = math.Ldexp(float64(0x0400+mantissa), int(exponent)-25)
	}
	if sign != 0 {
		value = -value
	}
	return float32(value)
}

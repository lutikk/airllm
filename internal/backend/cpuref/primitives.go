// Package cpuref implements small, pure-Go reference primitives for inference.
// It favors clear indexing and numerically stable accumulation over performance.
package cpuref

import (
	"fmt"
	"math"
)

// Linear computes input * weight^T. Weight has shape [outDim, inDim].
func Linear(input, weight []float32, inDim, outDim int) ([]float32, error) {
	if inDim <= 0 || outDim <= 0 {
		return nil, fmt.Errorf("linear dimensions must be positive: in=%d out=%d", inDim, outDim)
	}
	if len(input) != inDim {
		return nil, fmt.Errorf("linear input length %d, want %d", len(input), inDim)
	}
	if len(weight) != inDim*outDim {
		return nil, fmt.Errorf("linear weight length %d, want %d", len(weight), inDim*outDim)
	}

	output := make([]float32, outDim)
	for out := 0; out < outDim; out++ {
		var sum float64
		row := weight[out*inDim : (out+1)*inDim]
		for in, value := range input {
			sum += float64(value) * float64(row[in])
		}
		output[out] = float32(sum)
	}
	return output, nil
}

// LinearBias computes input * weight^T + bias. Weight has shape [outDim, inDim].
func LinearBias(input, weight, bias []float32, inDim, outDim int) ([]float32, error) {
	if len(bias) != outDim {
		return nil, fmt.Errorf("linear bias length %d, want %d", len(bias), outDim)
	}
	output, err := Linear(input, weight, inDim, outDim)
	if err != nil {
		return nil, err
	}
	for i := range output {
		output[i] += bias[i]
	}
	return output, nil
}

// RMSNorm normalizes input over its full length and applies a per-element weight.
func RMSNorm(input, weight []float32, epsilon float32) ([]float32, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("rmsnorm input must not be empty")
	}
	if len(input) != len(weight) {
		return nil, fmt.Errorf("rmsnorm weight length %d, want %d", len(weight), len(input))
	}
	if epsilon <= 0 || math.IsNaN(float64(epsilon)) {
		return nil, fmt.Errorf("rmsnorm epsilon must be positive: %v", epsilon)
	}

	var sumSquares float64
	for _, value := range input {
		sumSquares += float64(value) * float64(value)
	}
	inverseRMS := 1 / math.Sqrt(sumSquares/float64(len(input))+float64(epsilon))

	output := make([]float32, len(input))
	for i, value := range input {
		output[i] = float32(float64(value) * inverseRMS * float64(weight[i]))
	}
	return output, nil
}

// SiLU applies x * sigmoid(x) element-wise.
func SiLU(input []float32) []float32 {
	output := make([]float32, len(input))
	for i, value := range input {
		x := float64(value)
		output[i] = float32(x / (1 + math.Exp(-x)))
	}
	return output
}

// Multiply computes the element-wise product of a and b.
func Multiply(a, b []float32) ([]float32, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("multiply lengths differ: %d and %d", len(a), len(b))
	}
	output := make([]float32, len(a))
	for i := range a {
		output[i] = a[i] * b[i]
	}
	return output, nil
}

// SiLUGate computes SiLU(gate) * up for a gated MLP.
func SiLUGate(gate, up []float32) ([]float32, error) {
	if len(gate) != len(up) {
		return nil, fmt.Errorf("silu gate lengths differ: %d and %d", len(gate), len(up))
	}
	output := make([]float32, len(gate))
	for i, value := range gate {
		x := float64(value)
		output[i] = float32(x/(1+math.Exp(-x))) * up[i]
	}
	return output, nil
}

// GPTOSSSwiGLU clamps gate above at 7 and up to [-7, 7], then computes
// gate * sigmoid(1.702*gate) * (up + 1).
func GPTOSSSwiGLU(gate, up []float32) ([]float32, error) {
	if len(gate) != len(up) {
		return nil, fmt.Errorf("gpt-oss swiglu lengths differ: %d and %d", len(gate), len(up))
	}
	output := make([]float32, len(gate))
	for i := range gate {
		g := math.Min(float64(gate[i]), 7)
		u := math.Max(-7, math.Min(7, float64(up[i])))
		output[i] = float32(g / (1 + math.Exp(-1.702*g)) * (u + 1))
	}
	return output, nil
}

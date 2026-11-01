package cpuref

import (
	"fmt"
	"math"
)

// RoPE applies Hugging Face rotate-half rotary embeddings to one attention head.
// Input and output have shape [sequenceLength, headDim]. positionOffset is the
// absolute position of the first token, which supports cached decoding.
func RoPE(input []float32, headDim, positionOffset int, theta float64) ([]float32, error) {
	if headDim <= 0 || headDim%2 != 0 {
		return nil, fmt.Errorf("rope head dimension must be positive and even: %d", headDim)
	}
	if positionOffset < 0 {
		return nil, fmt.Errorf("rope position offset must be non-negative: %d", positionOffset)
	}
	if theta <= 0 || math.IsNaN(theta) || math.IsInf(theta, 0) {
		return nil, fmt.Errorf("rope theta must be finite and positive: %v", theta)
	}
	if len(input)%headDim != 0 {
		return nil, fmt.Errorf("rope input length %d is not divisible by head dimension %d", len(input), headDim)
	}

	output := make([]float32, len(input))
	half := headDim / 2
	for token := 0; token < len(input)/headDim; token++ {
		position := float64(positionOffset + token)
		row := input[token*headDim : (token+1)*headDim]
		out := output[token*headDim : (token+1)*headDim]
		for i := 0; i < half; i++ {
			frequency := math.Pow(theta, -2*float64(i)/float64(headDim))
			angle := position * frequency
			sine, cosine := math.Sincos(angle)
			first, second := float64(row[i]), float64(row[i+half])
			out[i] = float32(first*cosine - second*sine)
			out[i+half] = float32(second*cosine + first*sine)
		}
	}
	return output, nil
}

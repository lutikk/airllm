package cpuref

import (
	"fmt"
	"math"
)

// RoPE applies Hugging Face rotate-half rotary embeddings to one attention head.
// Input and output have shape [sequenceLength, headDim]. positionOffset is the
// absolute position of the first token, which supports cached decoding.
func RoPE(input []float32, headDim, positionOffset int, theta float64) ([]float32, error) {
	return rope(input, headDim, positionOffset, theta, nil, 1)
}

// YaRNRoPE applies gpt-oss YaRN rotary embeddings: factor 32, original context
// 4096, beta fast 32, beta slow 1, and concentration 1 + 0.1*ln(32).
func YaRNRoPE(input []float32, headDim, positionOffset int, theta float64) ([]float32, error) {
	const (
		factor               = 32.0
		originalMaxPositions = 4096.0
		betaFast             = 32.0
		betaSlow             = 1.0
	)
	if headDim <= 0 || headDim%2 != 0 {
		return nil, fmt.Errorf("rope head dimension must be positive and even: %d", headDim)
	}
	if theta <= 0 || math.IsNaN(theta) || math.IsInf(theta, 0) {
		return nil, fmt.Errorf("rope theta must be finite and positive: %v", theta)
	}

	frequencies := make([]float64, headDim/2)
	low := math.Max(0, math.Floor(yarnCorrectionDimension(betaFast, headDim, theta, originalMaxPositions)))
	high := math.Min(float64(headDim/2-1), math.Ceil(yarnCorrectionDimension(betaSlow, headDim, theta, originalMaxPositions)))
	for i := range frequencies {
		baseFrequency := math.Pow(theta, -2*float64(i)/float64(headDim))
		ramp := 0.0
		if high != low {
			ramp = (float64(i) - low) / (high - low)
		}
		ramp = math.Max(0, math.Min(1, ramp))
		frequencies[i] = baseFrequency * (ramp + (1-ramp)/factor)
	}
	concentration := 1 + 0.1*math.Log(factor)
	return rope(input, headDim, positionOffset, theta, frequencies, concentration)
}

func yarnCorrectionDimension(beta float64, headDim int, theta, originalMaxPositions float64) float64 {
	return float64(headDim) * math.Log(originalMaxPositions/(beta*2*math.Pi)) / (2 * math.Log(theta))
}

func rope(input []float32, headDim, positionOffset int, theta float64, frequencies []float64, concentration float64) ([]float32, error) {
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
			if frequencies != nil {
				frequency = frequencies[i]
			}
			angle := position * frequency
			sine, cosine := math.Sincos(angle)
			sine *= concentration
			cosine *= concentration
			first, second := float64(row[i]), float64(row[i+half])
			out[i] = float32(first*cosine - second*sine)
			out[i+half] = float32(second*cosine + first*sine)
		}
	}
	return output, nil
}

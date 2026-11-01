package cpuref

import (
	"fmt"
	"math"
)

// LayerConfig describes the dimensions of a Llama-style decoder layer.
type LayerConfig struct {
	HiddenSize              int
	IntermediateSize        int
	NumQueryHeads           int
	NumKVHeads              int
	HeadDim                 int
	NormEpsilon             float32
	RoPETheta               float64
	NormWeightPlusOne       bool
	Gemma2                  bool
	QueryPreAttentionScalar float64
	AttentionLogitSoftcap   float64
	SlidingWindow           int
}

// LayerWeights contains all projections and RMSNorm gains for a decoder layer.
// Projection matrices are row-major [outDim, inDim]. Norm1 precedes attention;
// Norm2 precedes the gated MLP.
type LayerWeights struct {
	Q, K, V, O          []float32
	QBias, KBias, VBias []float32
	Gate, Up, Down      []float32
	Norm1, Norm2        []float32
	PostAttentionNorm   []float32
	PostFeedForwardNorm []float32
}

// DecoderLayer is a pure-Go reference implementation of one decoder block.
type DecoderLayer struct {
	Config  LayerConfig
	Weights LayerWeights
}

// ForwardPrefill processes a sequence starting at position zero. The cache must
// be empty and is populated with rotated keys and values for the whole sequence.
// Input and output have shape [sequenceLength, HiddenSize].
func (l DecoderLayer) ForwardPrefill(input []float32, sequenceLength int, cache *KVCache) ([]float32, error) {
	if sequenceLength <= 0 {
		return nil, fmt.Errorf("prefill sequence length must be positive: %d", sequenceLength)
	}
	if cache != nil && cache.Len() != 0 {
		return nil, fmt.Errorf("prefill requires an empty cache, got length %d", cache.Len())
	}
	return l.forward(input, sequenceLength, cache)
}

// ForwardDecode processes exactly one token at the cache's current position and
// appends its key and value. Input and output have shape [HiddenSize].
func (l DecoderLayer) ForwardDecode(token []float32, cache *KVCache) ([]float32, error) {
	return l.forward(token, 1, cache)
}

func (l DecoderLayer) forward(input []float32, sequenceLength int, cache *KVCache) ([]float32, error) {
	if err := l.validate(input, sequenceLength, cache); err != nil {
		return nil, err
	}
	config := l.Config
	weights := l.Weights
	queryWidth := config.NumQueryHeads * config.HeadDim
	kvWidth := config.NumKVHeads * config.HeadDim
	positionOffset := cache.Len()

	queries := make([]float32, sequenceLength*queryWidth)
	keys := make([]float32, sequenceLength*kvWidth)
	values := make([]float32, sequenceLength*kvWidth)
	for token := 0; token < sequenceLength; token++ {
		hidden := input[token*config.HiddenSize : (token+1)*config.HiddenSize]
		normalized, err := l.rmsNorm(hidden, weights.Norm1)
		if err != nil {
			return nil, err
		}
		q, err := linearOptionalBias(normalized, weights.Q, weights.QBias, config.HiddenSize, queryWidth)
		if err != nil {
			return nil, err
		}
		k, err := linearOptionalBias(normalized, weights.K, weights.KBias, config.HiddenSize, kvWidth)
		if err != nil {
			return nil, err
		}
		v, err := linearOptionalBias(normalized, weights.V, weights.VBias, config.HiddenSize, kvWidth)
		if err != nil {
			return nil, err
		}
		copy(queries[token*queryWidth:(token+1)*queryWidth], q)
		copy(keys[token*kvWidth:(token+1)*kvWidth], k)
		copy(values[token*kvWidth:(token+1)*kvWidth], v)
	}

	var err error
	queries, err = ropeHeads(queries, sequenceLength, config.NumQueryHeads, config.HeadDim, positionOffset, config.RoPETheta)
	if err != nil {
		return nil, err
	}
	keys, err = ropeHeads(keys, sequenceLength, config.NumKVHeads, config.HeadDim, positionOffset, config.RoPETheta)
	if err != nil {
		return nil, err
	}
	if err := cache.Append(keys, values, sequenceLength); err != nil {
		return nil, err
	}
	attention, err := CausalAttentionWithConfig(queries, config.NumQueryHeads, cache, AttentionConfig{
		SlidingWindow:           config.SlidingWindow,
		QueryPreAttentionScalar: config.QueryPreAttentionScalar,
		LogitSoftcap:            config.AttentionLogitSoftcap,
	})
	if err != nil {
		return nil, err
	}

	// Gemma2 normalizes attention and MLP outputs before adding residuals.
	residual := make([]float32, len(input))
	for token := 0; token < sequenceLength; token++ {
		attentionRow := attention[token*queryWidth : (token+1)*queryWidth]
		projected, err := Linear(attentionRow, weights.O, queryWidth, config.HiddenSize)
		if err != nil {
			return nil, err
		}
		if config.Gemma2 {
			projected, err = l.rmsNorm(projected, weights.PostAttentionNorm)
			if err != nil {
				return nil, err
			}
		}
		for i := 0; i < config.HiddenSize; i++ {
			residual[token*config.HiddenSize+i] = input[token*config.HiddenSize+i] + projected[i]
		}
	}

	output := make([]float32, len(input))
	for token := 0; token < sequenceLength; token++ {
		residualRow := residual[token*config.HiddenSize : (token+1)*config.HiddenSize]
		normalized, err := l.rmsNorm(residualRow, weights.Norm2)
		if err != nil {
			return nil, err
		}
		gate, err := Linear(normalized, weights.Gate, config.HiddenSize, config.IntermediateSize)
		if err != nil {
			return nil, err
		}
		up, err := Linear(normalized, weights.Up, config.HiddenSize, config.IntermediateSize)
		if err != nil {
			return nil, err
		}
		var activated []float32
		if config.NormWeightPlusOne {
			activated, err = ApproximateGELUGate(gate, up)
		} else {
			activated, err = SiLUGate(gate, up)
		}
		if err != nil {
			return nil, err
		}
		down, err := Linear(activated, weights.Down, config.IntermediateSize, config.HiddenSize)
		if err != nil {
			return nil, err
		}
		if config.Gemma2 {
			down, err = l.rmsNorm(down, weights.PostFeedForwardNorm)
			if err != nil {
				return nil, err
			}
		}
		for i := 0; i < config.HiddenSize; i++ {
			output[token*config.HiddenSize+i] = residualRow[i] + down[i]
		}
	}
	return output, nil
}

func (l DecoderLayer) validate(input []float32, sequenceLength int, cache *KVCache) error {
	config := l.Config
	if config.HiddenSize <= 0 || config.IntermediateSize <= 0 || config.NumQueryHeads <= 0 || config.NumKVHeads <= 0 || config.HeadDim <= 0 {
		return fmt.Errorf("layer dimensions must be positive: %+v", config)
	}
	if config.HeadDim%2 != 0 {
		return fmt.Errorf("layer head dimension must be even: %d", config.HeadDim)
	}
	if config.NumQueryHeads%config.NumKVHeads != 0 {
		return fmt.Errorf("layer query heads %d must be divisible by kv heads %d", config.NumQueryHeads, config.NumKVHeads)
	}
	if config.NormEpsilon <= 0 || math.IsNaN(float64(config.NormEpsilon)) {
		return fmt.Errorf("layer norm epsilon must be positive: %v", config.NormEpsilon)
	}
	if config.RoPETheta <= 0 || math.IsNaN(config.RoPETheta) || math.IsInf(config.RoPETheta, 0) {
		return fmt.Errorf("layer rope theta must be finite and positive: %v", config.RoPETheta)
	}
	if len(input) != sequenceLength*config.HiddenSize {
		return fmt.Errorf("layer input length %d, want %d", len(input), sequenceLength*config.HiddenSize)
	}
	if cache == nil {
		return fmt.Errorf("layer cache must not be nil")
	}
	if cache.numKVHeads != config.NumKVHeads || cache.headDim != config.HeadDim {
		return fmt.Errorf("layer cache shape kvHeads=%d headDim=%d, want %d and %d", cache.numKVHeads, cache.headDim, config.NumKVHeads, config.HeadDim)
	}
	if cache.Len()+sequenceLength > cache.capacity {
		return fmt.Errorf("layer input exceeds cache capacity: length=%d append=%d capacity=%d", cache.Len(), sequenceLength, cache.capacity)
	}

	queryWidth := config.NumQueryHeads * config.HeadDim
	kvWidth := config.NumKVHeads * config.HeadDim
	checks := []struct {
		name   string
		values []float32
		want   int
	}{
		{"q", l.Weights.Q, queryWidth * config.HiddenSize},
		{"k", l.Weights.K, kvWidth * config.HiddenSize},
		{"v", l.Weights.V, kvWidth * config.HiddenSize},
		{"o", l.Weights.O, config.HiddenSize * queryWidth},
		{"gate", l.Weights.Gate, config.IntermediateSize * config.HiddenSize},
		{"up", l.Weights.Up, config.IntermediateSize * config.HiddenSize},
		{"down", l.Weights.Down, config.HiddenSize * config.IntermediateSize},
		{"norm1", l.Weights.Norm1, config.HiddenSize},
		{"norm2", l.Weights.Norm2, config.HiddenSize},
	}
	for _, check := range checks {
		if len(check.values) != check.want {
			return fmt.Errorf("layer %s weight length %d, want %d", check.name, len(check.values), check.want)
		}
	}
	biases := []struct {
		name   string
		values []float32
		want   int
	}{
		{"q bias", l.Weights.QBias, queryWidth},
		{"k bias", l.Weights.KBias, kvWidth},
		{"v bias", l.Weights.VBias, kvWidth},
	}
	for _, check := range biases {
		if check.values != nil && len(check.values) != check.want {
			return fmt.Errorf("layer %s length %d, want %d", check.name, len(check.values), check.want)
		}
	}
	if config.Gemma2 {
		for name, values := range map[string][]float32{
			"post attention norm":    l.Weights.PostAttentionNorm,
			"post feed forward norm": l.Weights.PostFeedForwardNorm,
		} {
			if len(values) != config.HiddenSize {
				return fmt.Errorf("layer %s length %d, want %d", name, len(values), config.HiddenSize)
			}
		}
	}
	return nil
}

func (l DecoderLayer) rmsNorm(input, weight []float32) ([]float32, error) {
	if l.Config.NormWeightPlusOne {
		return RMSNormPlusOne(input, weight, l.Config.NormEpsilon)
	}
	return RMSNorm(input, weight, l.Config.NormEpsilon)
}

func linearOptionalBias(input, weight, bias []float32, inDim, outDim int) ([]float32, error) {
	if bias == nil {
		return Linear(input, weight, inDim, outDim)
	}
	return LinearBias(input, weight, bias, inDim, outDim)
}

func ropeHeads(input []float32, sequenceLength, heads, headDim, positionOffset int, theta float64) ([]float32, error) {
	output := make([]float32, len(input))
	width := heads * headDim
	for head := 0; head < heads; head++ {
		headSequence := make([]float32, sequenceLength*headDim)
		for token := 0; token < sequenceLength; token++ {
			copy(headSequence[token*headDim:(token+1)*headDim], input[token*width+head*headDim:token*width+(head+1)*headDim])
		}
		rotated, err := RoPE(headSequence, headDim, positionOffset, theta)
		if err != nil {
			return nil, err
		}
		for token := 0; token < sequenceLength; token++ {
			copy(output[token*width+head*headDim:token*width+(head+1)*headDim], rotated[token*headDim:(token+1)*headDim])
		}
	}
	return output, nil
}

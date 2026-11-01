package gptoss

import (
	"fmt"

	"github.com/lutikk/airllm/internal/backend/cpuref"
)

// DecoderLayer contains one resident attention/router layer. Expert tensors are
// represented by GGUF ranges and decoded only after router selection.
type DecoderLayer struct {
	config        Config
	weights       layerWeights
	slidingWindow int
}

func (l DecoderLayer) forwardPrefill(input []float32, sequenceLength int, cache *cpuref.KVCache, source *tensorSource) ([]float32, error) {
	if sequenceLength <= 0 {
		return nil, fmt.Errorf("gptoss: prefill sequence length must be positive")
	}
	if cache != nil && cache.Len() != 0 {
		return nil, fmt.Errorf("gptoss: prefill requires an empty cache")
	}
	return l.forward(input, sequenceLength, cache, source)
}

func (l DecoderLayer) forwardDecode(token []float32, cache *cpuref.KVCache, source *tensorSource) ([]float32, error) {
	return l.forward(token, 1, cache, source)
}

func (l DecoderLayer) forward(input []float32, sequenceLength int, cache *cpuref.KVCache, source *tensorSource) ([]float32, error) {
	config := l.config
	weights := l.weights
	if cache == nil || source == nil {
		return nil, fmt.Errorf("gptoss: layer cache and tensor source must not be nil")
	}
	if len(input) != sequenceLength*config.HiddenSize {
		return nil, fmt.Errorf("gptoss: layer input length %d, want %d", len(input), sequenceLength*config.HiddenSize)
	}
	queryWidth := config.NumHeads * config.HeadDim
	kvWidth := config.NumKVHeads * config.HeadDim
	positionOffset := cache.Len()
	queries := make([]float32, sequenceLength*queryWidth)
	keys := make([]float32, sequenceLength*kvWidth)
	values := make([]float32, sequenceLength*kvWidth)

	for token := 0; token < sequenceLength; token++ {
		hidden := input[token*config.HiddenSize : (token+1)*config.HiddenSize]
		normalized, err := cpuref.RMSNorm(hidden, weights.norm1, float32(config.RMSNormEpsilon))
		if err != nil {
			return nil, err
		}
		qkv, err := cpuref.LinearBias(normalized, weights.qkv, weights.qkvBias, config.HiddenSize, queryWidth+2*kvWidth)
		if err != nil {
			return nil, err
		}
		copy(queries[token*queryWidth:(token+1)*queryWidth], qkv[:queryWidth])
		copy(keys[token*kvWidth:(token+1)*kvWidth], qkv[queryWidth:queryWidth+kvWidth])
		copy(values[token*kvWidth:(token+1)*kvWidth], qkv[queryWidth+kvWidth:])
	}

	var err error
	queries, err = yarnHeads(queries, sequenceLength, config.NumHeads, config.HeadDim, positionOffset, config.RoPETheta)
	if err != nil {
		return nil, err
	}
	keys, err = yarnHeads(keys, sequenceLength, config.NumKVHeads, config.HeadDim, positionOffset, config.RoPETheta)
	if err != nil {
		return nil, err
	}
	if err := cache.Append(keys, values, sequenceLength); err != nil {
		return nil, err
	}
	attention, err := cpuref.CausalAttentionWithConfig(queries, config.NumHeads, cache, cpuref.AttentionConfig{
		Sinks:         weights.sinks,
		SlidingWindow: l.slidingWindow,
	})
	if err != nil {
		return nil, err
	}

	residual := make([]float32, len(input))
	for token := 0; token < sequenceLength; token++ {
		row := attention[token*queryWidth : (token+1)*queryWidth]
		projected, err := cpuref.LinearBias(row, weights.output, weights.outputBias, queryWidth, config.HiddenSize)
		if err != nil {
			return nil, err
		}
		for i := 0; i < config.HiddenSize; i++ {
			residual[token*config.HiddenSize+i] = input[token*config.HiddenSize+i] + projected[i]
		}
	}

	output := make([]float32, len(input))
	for token := 0; token < sequenceLength; token++ {
		residualRow := residual[token*config.HiddenSize : (token+1)*config.HiddenSize]
		normalized, err := cpuref.RMSNorm(residualRow, weights.norm2, float32(config.RMSNormEpsilon))
		if err != nil {
			return nil, err
		}
		routerLogits, err := cpuref.LinearBias(normalized, weights.router, weights.routerBias, config.HiddenSize, config.NumExperts)
		if err != nil {
			return nil, err
		}
		experts, probabilities, err := cpuref.MoERouterTopK(routerLogits, config.NumExpertsPerToken)
		if err != nil {
			return nil, err
		}
		mixture := make([]float64, config.HiddenSize)
		for selected, expert := range experts {
			down, err := l.forwardExpert(normalized, expert, source)
			if err != nil {
				return nil, err
			}
			for i, value := range down {
				mixture[i] += float64(probabilities[selected]) * float64(value)
			}
		}
		for i := 0; i < config.HiddenSize; i++ {
			output[token*config.HiddenSize+i] = residualRow[i] + float32(mixture[i])
		}
	}
	return output, nil
}

func (l DecoderLayer) forwardExpert(hidden []float32, expert int, source *tensorSource) ([]float32, error) {
	config := l.config
	weights := l.weights
	var gate, up []float32
	if weights.fusedGateUp {
		gateUpWeight, err := source.readExpert(weights.gateUp, expert)
		if err != nil {
			return nil, err
		}
		gateUpBias, err := source.readExpert(weights.gateUpBias, expert)
		if err != nil {
			return nil, err
		}
		gateUp, err := cpuref.LinearBias(hidden, gateUpWeight, gateUpBias, config.HiddenSize, 2*config.IntermediateSize)
		if err != nil {
			return nil, err
		}
		gate = make([]float32, config.IntermediateSize)
		up = make([]float32, config.IntermediateSize)
		for i := 0; i < config.IntermediateSize; i++ {
			gate[i] = gateUp[2*i]
			up[i] = gateUp[2*i+1]
		}
	} else {
		gateWeight, err := source.readExpert(weights.gate, expert)
		if err != nil {
			return nil, err
		}
		gateBias, err := source.readExpert(weights.gateBias, expert)
		if err != nil {
			return nil, err
		}
		upWeight, err := source.readExpert(weights.up, expert)
		if err != nil {
			return nil, err
		}
		upBias, err := source.readExpert(weights.upBias, expert)
		if err != nil {
			return nil, err
		}
		gate, err = cpuref.LinearBias(hidden, gateWeight, gateBias, config.HiddenSize, config.IntermediateSize)
		if err != nil {
			return nil, err
		}
		up, err = cpuref.LinearBias(hidden, upWeight, upBias, config.HiddenSize, config.IntermediateSize)
		if err != nil {
			return nil, err
		}
	}
	activated, err := cpuref.GPTOSSSwiGLU(gate, up)
	if err != nil {
		return nil, err
	}
	downWeight, err := source.readExpert(weights.down, expert)
	if err != nil {
		return nil, err
	}
	downBias, err := source.readExpert(weights.downBias, expert)
	if err != nil {
		return nil, err
	}
	return cpuref.LinearBias(activated, downWeight, downBias, config.IntermediateSize, config.HiddenSize)
}

func yarnHeads(input []float32, sequenceLength, heads, headDim, positionOffset int, theta float64) ([]float32, error) {
	output := make([]float32, len(input))
	width := heads * headDim
	for head := 0; head < heads; head++ {
		headSequence := make([]float32, sequenceLength*headDim)
		for token := 0; token < sequenceLength; token++ {
			copy(headSequence[token*headDim:(token+1)*headDim], input[token*width+head*headDim:token*width+(head+1)*headDim])
		}
		rotated, err := cpuref.YaRNRoPE(headSequence, headDim, positionOffset, theta)
		if err != nil {
			return nil, err
		}
		for token := 0; token < sequenceLength; token++ {
			copy(output[token*width+head*headDim:token*width+(head+1)*headDim], rotated[token*headDim:(token+1)*headDim])
		}
	}
	return output, nil
}

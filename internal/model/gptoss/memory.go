package gptoss

import (
	"fmt"
	"math"

	"github.com/lutikk/airllm/internal/gguf"
	"github.com/lutikk/airllm/pkg/memory"
)

// EstimateMemory returns a conservative Float32 CPU budget for native-GGUF
// inference. Decoder weights remain streamed; selected expert buffers are
// included in MaxLayer rather than counting the complete model.
func EstimateMemory(path string, contextLength int) (_ memory.Estimate, returnErr error) {
	if contextLength <= 0 {
		return memory.Estimate{}, fmt.Errorf("gptoss: context length must be positive: %d", contextLength)
	}
	reader, err := gguf.Open(path)
	if err != nil {
		return memory.Estimate{}, err
	}
	defer func() {
		if closeErr := reader.Close(); returnErr == nil && closeErr != nil {
			returnErr = fmt.Errorf("gptoss: close GGUF: %w", closeErr)
		}
	}()
	config, err := LoadConfig(reader)
	if err != nil {
		return memory.Estimate{}, err
	}
	if contextLength > config.ContextLength {
		return memory.Estimate{}, fmt.Errorf("gptoss: context length %d exceeds model maximum %d", contextLength, config.ContextLength)
	}

	hidden := uint64(config.HiddenSize)
	vocab := uint64(config.VocabSize)
	residentElements := satAdd(satMul(vocab, hidden), hidden)
	if _, output := reader.Tensor("output.weight"); output {
		residentElements = satAdd(residentElements, satMul(vocab, hidden))
	}
	queryWidth := uint64(config.NumHeads * config.HeadDim)
	kvWidth := uint64(config.NumKVHeads * config.HeadDim)
	qkvWidth := satAdd(queryWidth, satMul(2, kvWidth))
	intermediate := uint64(config.IntermediateSize)
	experts := uint64(config.NumExperts)
	attentionElements := satAdd(
		satMul(hidden, qkvWidth), qkvWidth,
		uint64(config.NumHeads),
		satMul(queryWidth, hidden), hidden,
		satMul(2, hidden),
		satMul(hidden, experts), experts,
	)
	// gate, up, down and their biases for each selected expert.
	expertElements := satAdd(satMul(3, hidden, intermediate), satMul(2, intermediate), hidden)
	layerElements := satAdd(attentionElements, satMul(uint64(config.NumExpertsPerToken), expertElements))
	return memory.Estimate{
		ResidentWeights: satMul(residentElements, 4),
		MaxLayer:        satMul(layerElements, 4),
		KVCache: memory.LlamaKVCacheBytes(
			uint64(config.NumLayers), uint64(contextLength),
			uint64(config.NumKVHeads), uint64(config.HeadDim), 4,
		),
	}, nil
}

func satAdd(values ...uint64) uint64 {
	result := uint64(0)
	for _, value := range values {
		if math.MaxUint64-result < value {
			return math.MaxUint64
		}
		result += value
	}
	return result
}

func satMul(first uint64, rest ...uint64) uint64 {
	result := first
	for _, value := range rest {
		if result != 0 && value > math.MaxUint64/result {
			return math.MaxUint64
		}
		result *= value
	}
	return result
}

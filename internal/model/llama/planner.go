package llama

import (
	"fmt"
	"math"

	"github.com/lutikk/airllm/pkg/memory"
)

// LayerPlanner estimates the host and device memory needed to run a model one
// transformer layer at a time. All sizes are expressed in bytes.
type LayerPlanner struct {
	config          Config
	bytesPerElement uint64
}

// NewLayerPlanner validates config and creates a planner for the model's
// element width (for example, 2 for float16 and 4 for float32).
func NewLayerPlanner(config Config, bytesPerElement uint64) (LayerPlanner, error) {
	if err := config.Validate(); err != nil {
		return LayerPlanner{}, fmt.Errorf("llama: invalid config: %w", err)
	}
	if bytesPerElement == 0 {
		return LayerPlanner{}, fmt.Errorf("llama: bytes per element must be positive")
	}
	return LayerPlanner{config: config, bytesPerElement: bytesPerElement}, nil
}

// MaxLayerBytes returns the size of the largest individual transformer layer,
// including its attention, MLP, and normalization weights.
func (p LayerPlanner) MaxLayerBytes() uint64 {
	c := p.config
	hidden := uint64(c.HiddenSize)
	intermediate := uint64(c.IntermediateSize)
	queryWidth := uint64(c.NumAttentionHeads) * uint64(c.HeadDim)
	kvWidth := uint64(c.NumKeyValueHeads) * uint64(c.HeadDim)

	// Q and O use query width (which differs from hidden size for some Gemma2
	// models); K and V use the narrower GQA width.
	elements := saturatingAdd(
		saturatingMultiply(2, hidden, queryWidth),
		saturatingMultiply(2, hidden, kvWidth),
		p.qkvBiasElements(),
		saturatingMultiply(3, hidden, intermediate),
		saturatingMultiply(uint64(p.normCount()), hidden),
	)
	return saturatingMultiply(elements, p.bytesPerElement)
}

// KVCacheBytes returns the key/value cache size for contextLength tokens.
func (p LayerPlanner) KVCacheBytes(contextLength uint64) uint64 {
	return memory.LlamaKVCacheBytes(
		uint64(p.config.NumHiddenLayers),
		contextLength,
		uint64(p.config.NumKeyValueHeads),
		uint64(p.config.HeadDim),
		p.bytesPerElement,
	)
}

// Estimate returns the resident host-staging weights, largest streamed layer
// buffer, and KV cache required by memory.Plan. Decoder layers are streamed and
// are not included in ResidentWeights. Workspace and prefetch are left to the caller.
func (p LayerPlanner) Estimate(contextLength uint64) memory.Estimate {
	return memory.Estimate{
		ResidentWeights: p.residentWeightBytes(),
		MaxLayer:        p.MaxLayerBytes(),
		KVCache:         p.KVCacheBytes(contextLength),
	}
}

// Plan compares the estimated memory requirements with a host/device budget.
func (p LayerPlanner) Plan(contextLength uint64, budget memory.Budget) memory.Result {
	return memory.Plan(budget, p.Estimate(contextLength))
}

// residentWeightBytes reports only weights kept resident by the host staging
// policy. Streamed decoder layers contribute to MaxLayerBytes, not this total.
func (p LayerPlanner) residentWeightBytes() uint64 {
	c := p.config
	hidden := uint64(c.HiddenSize)
	embeddings := saturatingMultiply(uint64(c.VocabSize), hidden)
	lmHead := uint64(0)
	if !c.TieWordEmbeddings {
		lmHead = embeddings
	}
	elements := saturatingAdd(
		embeddings,
		lmHead,
		hidden, // final RMSNorm
	)
	return saturatingMultiply(elements, p.bytesPerElement)
}

func (p LayerPlanner) layerElements() uint64 {
	c := p.config
	hidden := uint64(c.HiddenSize)
	queryWidth := uint64(c.NumAttentionHeads) * uint64(c.HeadDim)
	kvWidth := uint64(c.NumKeyValueHeads) * uint64(c.HeadDim)
	return saturatingAdd(
		saturatingMultiply(2, hidden, queryWidth),
		saturatingMultiply(2, hidden, kvWidth),
		p.qkvBiasElements(),
		saturatingMultiply(3, hidden, uint64(c.IntermediateSize)),
		saturatingMultiply(uint64(p.normCount()), hidden),
	)
}

func (p LayerPlanner) normCount() int {
	if p.config.IsGemma2() {
		return 4
	}
	return 2
}

func (p LayerPlanner) qkvBiasElements() uint64 {
	if !p.config.IsQwen2() {
		return 0
	}
	hidden := uint64(p.config.HiddenSize)
	kvWidth := uint64(p.config.NumKeyValueHeads) * uint64(p.config.HeadDim)
	return saturatingAdd(hidden, saturatingMultiply(2, kvWidth))
}

func saturatingAdd(values ...uint64) uint64 {
	result := uint64(0)
	for _, value := range values {
		if math.MaxUint64-result < value {
			return math.MaxUint64
		}
		result += value
	}
	return result
}

func saturatingMultiply(first uint64, rest ...uint64) uint64 {
	result := first
	for _, value := range rest {
		if result != 0 && value > math.MaxUint64/result {
			return math.MaxUint64
		}
		result *= value
	}
	return result
}

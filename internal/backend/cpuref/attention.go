package cpuref

import (
	"fmt"
	"math"
)

// KVCache stores keys and values as [sequenceLength, numKVHeads, headDim].
// Query heads share these entries during GQA; KV heads are never duplicated.
type KVCache struct {
	numKVHeads int
	headDim    int
	capacity   int
	length     int
	keys       []float32
	values     []float32
}

// NewKVCache allocates an empty cache with room for maxSequenceLength tokens.
func NewKVCache(numKVHeads, headDim, maxSequenceLength int) (*KVCache, error) {
	if numKVHeads <= 0 || headDim <= 0 || maxSequenceLength <= 0 {
		return nil, fmt.Errorf("kv cache dimensions must be positive: kvHeads=%d headDim=%d capacity=%d", numKVHeads, headDim, maxSequenceLength)
	}
	stride := numKVHeads * headDim
	return &KVCache{
		numKVHeads: numKVHeads,
		headDim:    headDim,
		capacity:   maxSequenceLength,
		keys:       make([]float32, maxSequenceLength*stride),
		values:     make([]float32, maxSequenceLength*stride),
	}, nil
}

// Len returns the number of cached tokens.
func (c *KVCache) Len() int {
	return c.length
}

// Append adds one contiguous sequence of keys and values to the cache.
func (c *KVCache) Append(keys, values []float32, sequenceLength int) error {
	if sequenceLength <= 0 {
		return fmt.Errorf("kv append sequence length must be positive: %d", sequenceLength)
	}
	if c.length+sequenceLength > c.capacity {
		return fmt.Errorf("kv append exceeds capacity: length=%d append=%d capacity=%d", c.length, sequenceLength, c.capacity)
	}
	want := sequenceLength * c.numKVHeads * c.headDim
	if len(keys) != want || len(values) != want {
		return fmt.Errorf("kv append lengths keys=%d values=%d, want %d", len(keys), len(values), want)
	}

	stride := c.numKVHeads * c.headDim
	start := c.length * stride
	end := start + want
	copy(c.keys[start:end], keys)
	copy(c.values[start:end], values)
	c.length += sequenceLength
	return nil
}

// AttentionConfig contains optional per-layer attention behavior. A zero
// SlidingWindow disables the window. Sinks contains one learned logit per query
// head; each sink participates in softmax normalization but has no value vector.
type AttentionConfig struct {
	Sinks                   []float32
	SlidingWindow           int
	QueryPreAttentionScalar float64
	LogitSoftcap            float64
}

// CausalAttention computes scaled dot-product attention for the most recently
// appended tokens. Query has shape [queryLength, numQueryHeads, headDim], and
// output has the same shape. Each query attends to cached tokens up to its own
// absolute position. The scale is 1/sqrt(headDim).
func CausalAttention(query []float32, numQueryHeads int, cache *KVCache) ([]float32, error) {
	return CausalAttentionWithConfig(query, numQueryHeads, cache, AttentionConfig{})
}

// CausalAttentionWithConfig computes causal attention with optional learned
// sinks and a per-layer sliding window. The window includes the current token.
func CausalAttentionWithConfig(query []float32, numQueryHeads int, cache *KVCache, config AttentionConfig) ([]float32, error) {
	if cache == nil {
		return nil, fmt.Errorf("attention cache must not be nil")
	}
	if numQueryHeads <= 0 || numQueryHeads%cache.numKVHeads != 0 {
		return nil, fmt.Errorf("attention query heads %d must be a positive multiple of kv heads %d", numQueryHeads, cache.numKVHeads)
	}
	if len(config.Sinks) != 0 && len(config.Sinks) != numQueryHeads {
		return nil, fmt.Errorf("attention sinks length %d, want %d", len(config.Sinks), numQueryHeads)
	}
	if config.SlidingWindow < 0 {
		return nil, fmt.Errorf("attention sliding window must be non-negative: %d", config.SlidingWindow)
	}
	if config.QueryPreAttentionScalar < 0 || math.IsNaN(config.QueryPreAttentionScalar) || math.IsInf(config.QueryPreAttentionScalar, 0) {
		return nil, fmt.Errorf("attention query pre-attention scalar must be finite and non-negative: %v", config.QueryPreAttentionScalar)
	}
	if config.LogitSoftcap < 0 || math.IsNaN(config.LogitSoftcap) || math.IsInf(config.LogitSoftcap, 0) {
		return nil, fmt.Errorf("attention logit softcap must be finite and non-negative: %v", config.LogitSoftcap)
	}
	queryStride := numQueryHeads * cache.headDim
	if len(query) == 0 || len(query)%queryStride != 0 {
		return nil, fmt.Errorf("attention query length %d is not a non-empty multiple of %d", len(query), queryStride)
	}
	queryLength := len(query) / queryStride
	if queryLength > cache.length {
		return nil, fmt.Errorf("attention query length %d exceeds cached length %d", queryLength, cache.length)
	}

	output := make([]float32, len(query))
	kvStride := cache.numKVHeads * cache.headDim
	headsPerKV := numQueryHeads / cache.numKVHeads
	scale := 1 / math.Sqrt(float64(cache.headDim))
	if config.QueryPreAttentionScalar > 0 {
		scale = 1 / math.Sqrt(config.QueryPreAttentionScalar)
	}
	scores := make([]float64, cache.length)
	firstQueryPosition := cache.length - queryLength

	for queryIndex := 0; queryIndex < queryLength; queryIndex++ {
		position := firstQueryPosition + queryIndex
		firstKeyPosition := 0
		if config.SlidingWindow > 0 && position >= config.SlidingWindow {
			firstKeyPosition = position - config.SlidingWindow + 1
		}
		for queryHead := 0; queryHead < numQueryHeads; queryHead++ {
			kvHead := queryHead / headsPerKV
			queryOffset := queryIndex*queryStride + queryHead*cache.headDim
			q := query[queryOffset : queryOffset+cache.headDim]

			maxScore := math.Inf(-1)
			if len(config.Sinks) != 0 {
				maxScore = float64(config.Sinks[queryHead])
			}
			for keyPosition := firstKeyPosition; keyPosition <= position; keyPosition++ {
				keyOffset := keyPosition*kvStride + kvHead*cache.headDim
				k := cache.keys[keyOffset : keyOffset+cache.headDim]
				var dot float64
				for d := 0; d < cache.headDim; d++ {
					dot += float64(q[d]) * float64(k[d])
				}
				scores[keyPosition] = dot * scale
				if config.LogitSoftcap > 0 {
					scores[keyPosition] = config.LogitSoftcap * math.Tanh(scores[keyPosition]/config.LogitSoftcap)
				}
				if scores[keyPosition] > maxScore {
					maxScore = scores[keyPosition]
				}
			}

			var denominator float64
			for keyPosition := firstKeyPosition; keyPosition <= position; keyPosition++ {
				scores[keyPosition] = math.Exp(scores[keyPosition] - maxScore)
				denominator += scores[keyPosition]
			}
			if len(config.Sinks) != 0 {
				denominator += math.Exp(float64(config.Sinks[queryHead]) - maxScore)
			}

			outOffset := queryOffset
			for d := 0; d < cache.headDim; d++ {
				var sum float64
				for keyPosition := firstKeyPosition; keyPosition <= position; keyPosition++ {
					valueOffset := keyPosition*kvStride + kvHead*cache.headDim + d
					sum += scores[keyPosition] * float64(cache.values[valueOffset])
				}
				output[outOffset+d] = float32(sum / denominator)
			}
		}
	}
	return output, nil
}

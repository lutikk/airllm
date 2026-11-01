package cpuref

import (
	"fmt"
	"math"
	"sort"
)

// MoERouterTopK selects the largest logits and applies softmax only to those
// selected logits. Indices are returned in descending logit order, with ties
// resolved toward the lower expert index.
func MoERouterTopK(logits []float32, topK int) ([]int, []float32, error) {
	if len(logits) == 0 {
		return nil, nil, fmt.Errorf("moe router logits must not be empty")
	}
	if topK <= 0 || topK > len(logits) {
		return nil, nil, fmt.Errorf("moe router top-k %d outside [1, %d]", topK, len(logits))
	}

	indices := make([]int, len(logits))
	for i := range indices {
		indices[i] = i
	}
	sort.SliceStable(indices, func(i, j int) bool {
		return logits[indices[i]] > logits[indices[j]]
	})
	indices = indices[:topK]

	maxLogit := float64(logits[indices[0]])
	weights := make([]float32, topK)
	var denominator float64
	for i, expert := range indices {
		weight := math.Exp(float64(logits[expert]) - maxLogit)
		weights[i] = float32(weight)
		denominator += weight
	}
	for i := range weights {
		weights[i] = float32(float64(weights[i]) / denominator)
	}
	return indices, weights, nil
}

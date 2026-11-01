package cpuref

import (
	"math"
	"testing"
)

func TestGemmaRMSNormAndGELUGolden(t *testing.T) {
	got, err := RMSNormPlusOne([]float32{3, 4}, []float32{0, -0.5}, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	rms := math.Sqrt(12.5 + 1e-6)
	assertClose(t, got, []float32{float32(3 / rms), float32(2 / rms)})

	gated, err := ApproximateGELUGate([]float32{1}, []float32{2})
	if err != nil {
		t.Fatal(err)
	}
	want := 2 * 0.5 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*1.044715))
	assertClose(t, gated, []float32{float32(want)})
}

func TestGemmaAttentionScaleAndSoftcap(t *testing.T) {
	cache, err := NewKVCache(1, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := cache.Append(
		[]float32{0, 0, 1, 0},
		[]float32{2, 0, 0, 4},
		2,
	); err != nil {
		t.Fatal(err)
	}
	query := []float32{1, 0, 1, 0}
	got, err := CausalAttentionWithConfig(query, 1, cache, AttentionConfig{
		QueryPreAttentionScalar: 16,
		LogitSoftcap:            2,
	})
	if err != nil {
		t.Fatal(err)
	}
	// The first query sees only itself; the second has softcapped scores 0 and
	// 2*tanh(1/8), instead of the default head-dimension scale.
	score := 2 * math.Tanh(1.0/8)
	weight := math.Exp(score) / (1 + math.Exp(score))
	assertClose(t, got, []float32{2, 0, float32(2 * (1 - weight)), float32(4 * weight)})
}

func TestGemma2PostOutputNormsAffectResidual(t *testing.T) {
	layer := DecoderLayer{
		Config: LayerConfig{
			HiddenSize: 2, IntermediateSize: 2, NumQueryHeads: 1, NumKVHeads: 1,
			HeadDim: 2, NormEpsilon: 1e-6, RoPETheta: 10000,
			NormWeightPlusOne: true, Gemma2: true,
		},
		Weights: LayerWeights{
			Q: make([]float32, 4), K: make([]float32, 4), V: make([]float32, 4),
			O: []float32{1, 0, 0, 1}, Gate: make([]float32, 4), Up: make([]float32, 4), Down: make([]float32, 4),
			Norm1: make([]float32, 2), Norm2: make([]float32, 2),
			PostAttentionNorm: []float32{-1, -1}, PostFeedForwardNorm: make([]float32, 2),
		},
	}
	cache, err := NewKVCache(1, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	got, err := layer.ForwardPrefill([]float32{3, 4}, 1, cache)
	if err != nil {
		t.Fatal(err)
	}
	// V is zero, so post-attention RMSNorm produces zero and the block returns
	// its input. Without Gemma2's post-output norm this would still pass, while
	// a missing norm weight would fail validation.
	assertClose(t, got, []float32{3, 4})
}

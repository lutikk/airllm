package cpuref

import (
	"strings"
	"testing"
)

func TestQwen2DecoderLayerAppliesQKVBias(t *testing.T) {
	layer := DecoderLayer{
		Config: LayerConfig{
			HiddenSize: 2, IntermediateSize: 2,
			NumQueryHeads: 1, NumKVHeads: 1, HeadDim: 2,
			NormEpsilon: 1e-6, RoPETheta: 1_000_000,
		},
		Weights: LayerWeights{
			Q: make([]float32, 4), K: make([]float32, 4), V: make([]float32, 4),
			QBias: []float32{0.25, -0.5}, KBias: []float32{-0.25, 0.5}, VBias: []float32{1, 2},
			O:    []float32{1, 0, 0, 1},
			Gate: make([]float32, 4), Up: make([]float32, 4), Down: make([]float32, 4),
			Norm1: []float32{1, 1}, Norm2: []float32{1, 1},
		},
	}
	cache, err := NewKVCache(1, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	output, err := layer.ForwardPrefill([]float32{0.25, -0.5}, 1, cache)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, output, []float32{1.25, 1.5})
}

func TestQwen2DecoderLayerRejectsWrongBiasShape(t *testing.T) {
	layer := testDecoderLayer()
	layer.Weights.QBias = []float32{1}
	cache, err := NewKVCache(1, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	_, err = layer.ForwardPrefill(make([]float32, 4), 1, cache)
	if err == nil || !strings.Contains(err.Error(), "q bias") {
		t.Fatalf("ForwardPrefill() error = %v, want q bias shape error", err)
	}
}

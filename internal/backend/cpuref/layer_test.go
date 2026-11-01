package cpuref

import "testing"

func TestDecoderLayerPrefillAndDecodeGolden(t *testing.T) {
	layer := testDecoderLayer()
	cache, err := NewKVCache(1, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	prefill, err := layer.ForwardPrefill([]float32{
		1, 0.5, -0.5, 2,
		-0.5, 1, 0.25, 0.5,
	}, 2, cache)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, prefill, []float32{
		1.0091444, 0.53885204, -0.4898679, 2.037783,
		-0.50090253, 1.0364378, 0.24955083, 0.5372144,
	})
	if cache.Len() != 2 {
		t.Fatalf("cache length after prefill = %d, want 2", cache.Len())
	}

	decoded, err := layer.ForwardDecode([]float32{0.25, -1, 1.5, 0.5}, cache)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, decoded, []float32{0.26711914, -0.9780289, 1.5090628, 0.5255541})
	if cache.Len() != 3 {
		t.Fatalf("cache length after decode = %d, want 3", cache.Len())
	}
}

func TestDecoderLayerRejectsNonEmptyPrefillCache(t *testing.T) {
	layer := testDecoderLayer()
	cache, err := NewKVCache(1, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := layer.ForwardDecode([]float32{1, 0.5, -0.5, 2}, cache); err != nil {
		t.Fatal(err)
	}
	if _, err := layer.ForwardPrefill(make([]float32, 4), 1, cache); err == nil {
		t.Fatal("ForwardPrefill accepted a non-empty cache")
	}
	if cache.Len() != 1 {
		t.Fatalf("rejected prefill changed cache length to %d", cache.Len())
	}
}

func testDecoderLayer() DecoderLayer {
	return DecoderLayer{
		Config: LayerConfig{
			HiddenSize:       4,
			IntermediateSize: 3,
			NumQueryHeads:    2,
			NumKVHeads:       1,
			HeadDim:          2,
			NormEpsilon:      1e-5,
			RoPETheta:        10000,
		},
		Weights: LayerWeights{
			Q: []float32{
				0.1, 0, 0.2, 0,
				0, 0.1, 0, 0.2,
				0.2, 0, 0, 0.1,
				0, 0.2, 0.1, 0,
			},
			K: []float32{
				0.1, 0.2, 0, 0,
				0, 0, 0.1, 0.2,
			},
			V: []float32{
				0.2, 0, 0.1, 0,
				0, 0.1, 0, 0.2,
			},
			O: []float32{
				0.1, 0, 0, 0,
				0, 0.1, 0, 0,
				0, 0, 0.1, 0,
				0, 0, 0, 0.1,
			},
			Gate: []float32{
				0.1, 0.2, 0, -0.1,
				0, 0.1, 0.2, 0.1,
				0.2, 0, 0.1, 0,
			},
			Up: []float32{
				0.2, 0, 0.1, 0.1,
				0.1, 0.2, 0, 0,
				0, 0.1, 0.2, -0.1,
			},
			Down: []float32{
				0.1, 0, 0.2,
				0, 0.1, 0,
				0.2, 0, 0.1,
				0, 0.2, 0.1,
			},
			Norm1: []float32{1, 1, 1, 1},
			Norm2: []float32{1, 0.5, 1.5, 1},
		},
	}
}

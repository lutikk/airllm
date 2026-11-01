package cpuref

import "testing"

func TestLinearBiasGolden(t *testing.T) {
	got, err := LinearBias(
		[]float32{1, -2, 0.5},
		[]float32{
			0.25, 0.5, -1,
			-2, 1, 0.75,
		},
		[]float32{0.75, -0.5},
		3,
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, got, []float32{-0.5, -4.125})
}

func TestCausalAttentionLearnedSinkGolden(t *testing.T) {
	cache, err := NewKVCache(1, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := cache.Append([]float32{0, 0}, []float32{2, -4}, 1); err != nil {
		t.Fatal(err)
	}

	got, err := CausalAttentionWithConfig(
		[]float32{0, 0},
		1,
		cache,
		AttentionConfig{Sinks: []float32{0}},
	)
	if err != nil {
		t.Fatal(err)
	}
	// The zero-valued key and zero sink each receive probability 1/2. The sink
	// contributes only to the denominator and is then discarded.
	assertClose(t, got, []float32{1, -2})
}

func TestCausalAttentionSlidingWindowGolden(t *testing.T) {
	cache, err := NewKVCache(1, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	if err := cache.Append(
		[]float32{1, 1, 1},
		[]float32{1, 2, 4},
		3,
	); err != nil {
		t.Fatal(err)
	}

	got, err := CausalAttentionWithConfig(
		[]float32{1, 1},
		1,
		cache,
		AttentionConfig{SlidingWindow: 2},
	)
	if err != nil {
		t.Fatal(err)
	}
	// At absolute position 1 the window contains values 1 and 2; at position 2
	// it advances to values 2 and 4, excluding the oldest cached token.
	assertClose(t, got, []float32{1.5, 3})
}

func TestYaRNRoPEGPTOSSGolden(t *testing.T) {
	got, err := YaRNRoPE(
		[]float32{1, 2, 3, 4, 5, 6, 7, 8},
		8,
		4096,
		10000,
	)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, got, []float32{
		-5.787687, 0.7495112, -9.812923, 5.676347,
		-3.6942456, 8.483434, -2.9792004, -10.622614,
	})
}

func TestMoERouterTopKSelectedSoftmaxGolden(t *testing.T) {
	indices, weights, err := MoERouterTopK([]float32{1, 3, 2, -1}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(indices) != 2 || indices[0] != 1 || indices[1] != 2 {
		t.Fatalf("indices = %v, want [1 2]", indices)
	}
	assertClose(t, weights, []float32{0.7310586, 0.26894143})
}

func TestGPTOSSSwiGLUGolden(t *testing.T) {
	got, err := GPTOSSSwiGLU(
		[]float32{-8, 0.5, 8},
		[]float32{-9, 0.5, 9},
	)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, got, []float32{0.00005859857, 0.5255826, 55.999626})
}

func TestGPTOSSPrimitiveValidation(t *testing.T) {
	if _, err := LinearBias([]float32{1}, []float32{1}, nil, 1, 1); err == nil {
		t.Error("LinearBias accepted an incorrectly sized bias")
	}
	cache, err := NewKVCache(1, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := cache.Append([]float32{0}, []float32{0}, 1); err != nil {
		t.Fatal(err)
	}
	if _, err := CausalAttentionWithConfig([]float32{0}, 1, cache, AttentionConfig{Sinks: []float32{0, 0}}); err == nil {
		t.Error("attention accepted one sink per two heads for a one-head query")
	}
	if _, err := CausalAttentionWithConfig([]float32{0}, 1, cache, AttentionConfig{SlidingWindow: -1}); err == nil {
		t.Error("attention accepted a negative sliding window")
	}
	if _, _, err := MoERouterTopK([]float32{1}, 2); err == nil {
		t.Error("MoERouterTopK accepted top-k larger than the expert count")
	}
	if _, err := GPTOSSSwiGLU([]float32{1}, nil); err == nil {
		t.Error("GPTOSSSwiGLU accepted mismatched lengths")
	}
}

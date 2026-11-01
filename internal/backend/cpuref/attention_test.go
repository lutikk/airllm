package cpuref

import "testing"

func TestCausalAttentionGQAGolden(t *testing.T) {
	cache, err := NewKVCache(2, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if err := cache.Append(
		[]float32{1, 0, 0, 1},
		[]float32{2, 0, 0, 4},
		1,
	); err != nil {
		t.Fatal(err)
	}
	if err := cache.Append(
		[]float32{
			0, 1, 1, 0,
			1, 1, -1, 1,
		},
		[]float32{
			1, 1, 2, 2,
			3, -1, -2, 1,
		},
		2,
	); err != nil {
		t.Fatal(err)
	}

	// Four query heads share the two stored KV heads. The first query can only
	// see positions 0 and 1; the second query can see positions 0 through 2.
	query := []float32{
		1, 0, 0, 1, 1, 1, -1, 0,
		1, 1, 0, -1, 1, 0, 0, 1,
	}
	got, err := CausalAttention(query, 4, cache)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{
		1.6697615, 0.33023846, 1.3302385, 0.66976154,
		1, 3, 0.6604769, 3.339523,
		2.2552347, -0.25523478, 2, 0,
		0.8718922, 2.4279616, -0.40667257, 2.401112,
	}
	assertClose(t, got, want)

	// Storage is three tokens by two KV heads by two dimensions, not four
	// expanded query heads. This guards against silently materializing GQA KV.
	if len(cache.keys) != 12 || len(cache.values) != 12 {
		t.Fatalf("cache storage lengths = %d and %d, want 12", len(cache.keys), len(cache.values))
	}
	if cache.Len() != 3 {
		t.Fatalf("cache length = %d, want 3", cache.Len())
	}
}

func TestKVCacheAppendIsTransactional(t *testing.T) {
	cache, err := NewKVCache(1, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := cache.Append([]float32{1, 2}, []float32{3}, 1); err == nil {
		t.Fatal("Append accepted a short value tensor")
	}
	if cache.Len() != 0 {
		t.Fatalf("failed append changed cache length to %d", cache.Len())
	}
	if err := cache.Append([]float32{1, 2}, []float32{3, 4}, 1); err != nil {
		t.Fatal(err)
	}
	if err := cache.Append([]float32{1, 2}, []float32{3, 4}, 1); err == nil {
		t.Fatal("Append accepted data beyond cache capacity")
	}
	if cache.Len() != 1 {
		t.Fatalf("capacity error changed cache length to %d", cache.Len())
	}
}

func TestCausalAttentionValidation(t *testing.T) {
	cache, err := NewKVCache(2, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := cache.Append(make([]float32, 4), make([]float32, 4), 1); err != nil {
		t.Fatal(err)
	}
	if _, err := CausalAttention(make([]float32, 6), 3, cache); err == nil {
		t.Error("CausalAttention accepted query heads not divisible by KV heads")
	}
	if _, err := CausalAttention(make([]float32, 16), 4, cache); err == nil {
		t.Error("CausalAttention accepted more queries than cached tokens")
	}
	if _, err := CausalAttention(make([]float32, 8), 4, nil); err == nil {
		t.Error("CausalAttention accepted a nil cache")
	}
}

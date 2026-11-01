package gptoss

import (
	"path/filepath"
	"testing"
)

func TestEstimateMemoryNativeGGUF(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	writeSyntheticGPTOSS(t, path)
	estimate, err := EstimateMemory(path, 8)
	if err != nil {
		t.Fatal(err)
	}
	// vocab=4, hidden=2, separate lm_head, final norm.
	if want := uint64((4*2*2 + 2) * 4); estimate.ResidentWeights != want {
		t.Fatalf("ResidentWeights = %d, want %d", estimate.ResidentWeights, want)
	}
	if want := uint64(2 * 2 * 8 * 1 * 2 * 4); estimate.KVCache != want {
		t.Fatalf("KVCache = %d, want %d", estimate.KVCache, want)
	}
	if estimate.MaxLayer == 0 {
		t.Fatal("MaxLayer is zero")
	}
}

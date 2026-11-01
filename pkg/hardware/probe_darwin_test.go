//go:build darwin

package hardware

import "testing"

func TestParseSystemProfiler(t *testing.T) {
	input := []byte(`[
		{
			"_name": "Test GPU",
			"spdisplays_vram": "8 GB",
			"spdisplays_ndrvs": []
		}
	]`)
	gpu, err := parseSystemProfiler(input)
	if err != nil {
		t.Fatalf("parseSystemProfiler() error = %v", err)
	}
	if gpu == nil {
		t.Fatal("parseSystemProfiler() returned no GPU")
	}
	if gpu.Name != "Test GPU" {
		t.Fatalf("GPU name = %q, want Test GPU", gpu.Name)
	}
	if gpu.VRAMBytes != 8_000_000_000 {
		t.Fatalf("VRAM = %d, want 8000000000", gpu.VRAMBytes)
	}
}

func TestParseSystemProfilerGracefullyHandlesUnavailableGPU(t *testing.T) {
	gpu, err := parseSystemProfiler([]byte(`[]`))
	if err != nil || gpu != nil {
		t.Fatalf("parseSystemProfiler([]) = (%v, %v), want (nil, nil)", gpu, err)
	}
	if _, err := parseSystemProfiler([]byte(`not json`)); err == nil {
		t.Fatal("parseSystemProfiler() accepted malformed JSON")
	}
}

func TestParseVRAMUnifiedOrUnknown(t *testing.T) {
	for _, value := range []string{"", "Dynamic", "-1 GB"} {
		if got := parseVRAM(value); got != 0 {
			t.Errorf("parseVRAM(%q) = %d, want 0", value, got)
		}
	}
}

//go:build linux

package hardware

import "testing"

func TestParseProcFiles(t *testing.T) {
	cpuinfo := []byte("processor: 0\nmodel name: Example CPU @ 3.00GHz\n")
	if got := parseCPUModel(cpuinfo); got != "Example CPU @ 3.00GHz" {
		t.Fatalf("CPU model = %q", got)
	}
	meminfo := []byte("MemTotal:       16384000 kB\nMemFree: 100 kB\n")
	if got := parseMemTotal(meminfo); got != 16_777_216_000 {
		t.Fatalf("RAM bytes = %d, want 16777216000", got)
	}
}

func TestParseNVIDIAOutput(t *testing.T) {
	gpu, ok := parseNVIDIAOutput([]byte("NVIDIA RTX 4090, 24564\n"))
	if !ok {
		t.Fatal("parseNVIDIAOutput() did not recognize valid output")
	}
	if gpu.Name != "NVIDIA RTX 4090" || gpu.VRAMBytes != 24_564*1024*1024 {
		t.Fatalf("unexpected GPU: %+v", gpu)
	}
	if _, ok := parseNVIDIAOutput([]byte("invalid")); ok {
		t.Fatal("parseNVIDIAOutput() accepted malformed output")
	}
}

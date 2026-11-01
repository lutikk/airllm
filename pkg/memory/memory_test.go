package memory

import (
	"math"
	"strings"
	"testing"
)

func TestParseBytes(t *testing.T) {
	tests := []struct {
		input string
		want  uint64
	}{
		{"42", 42},
		{"42 B", 42},
		{"1KB", 1_000},
		{"1.5 MB", 1_500_000},
		{"2gb", 2_000_000_000},
		{"3 TB", 3_000_000_000_000},
		{"1KiB", 1_024},
		{"1.5 MiB", 1_572_864},
		{"2 GiB", 2_147_483_648},
		{"1TiB", 1_099_511_627_776},
		{" 8 mb ", 8_000_000},
	}
	for _, test := range tests {
		t.Run(test.input, func(t *testing.T) {
			got, err := ParseBytes(test.input)
			if err != nil {
				t.Fatalf("ParseBytes(%q) error = %v", test.input, err)
			}
			if got != test.want {
				t.Fatalf("ParseBytes(%q) = %d, want %d", test.input, got, test.want)
			}
		})
	}
}

func TestParseBytesRejectsInvalidInput(t *testing.T) {
	for _, input := range []string{"", "MB", "one GB", "-1GB", "1 XB", "0.5 B", "18446744073709551616 B"} {
		t.Run(input, func(t *testing.T) {
			if _, err := ParseBytes(input); err == nil {
				t.Fatalf("ParseBytes(%q) unexpectedly succeeded", input)
			}
		})
	}
}

func TestPlanFits(t *testing.T) {
	budget := Budget{HostBytes: 10_000, DeviceBytes: 11_000}
	estimate := Estimate{
		ResidentWeights: 10_000,
		MaxLayer:        4_000,
		KVCache:         2_000,
		Workspace:       1_000,
		Prefetch:        true,
	}
	plan := Plan(budget, estimate)
	if !plan.Fits || !plan.HostFits || !plan.DeviceFits {
		t.Fatalf("Plan() did not fit: %+v", plan)
	}
	if plan.HostRequiredBytes != 10_000 || plan.DeviceRequiredBytes != 11_000 {
		t.Fatalf("Plan() requirements = (%d, %d)", plan.HostRequiredBytes, plan.DeviceRequiredBytes)
	}
	if err := Validate(budget, estimate); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
}

func TestPlanExplainsFailureAndSuggestsFixes(t *testing.T) {
	budget := Budget{HostBytes: 8_000, DeviceBytes: 10_000}
	estimate := Estimate{
		ResidentWeights: 10_000,
		MaxLayer:        4_000,
		KVCache:         2_000,
		Workspace:       1_000,
		Prefetch:        true,
	}
	plan := Plan(budget, estimate)
	if plan.Fits || plan.HostFits || plan.DeviceFits {
		t.Fatalf("Plan() unexpectedly fits: %+v", plan)
	}
	if len(plan.Suggestions) < 3 {
		t.Fatalf("Plan() suggestions = %v", plan.Suggestions)
	}
	err := plan.Validate()
	if err == nil {
		t.Fatal("Result.Validate() unexpectedly succeeded")
	}
	for _, advice := range []string{"prefetch", "context", "resident weights"} {
		if !strings.Contains(err.Error(), advice) {
			t.Errorf("Validate() error %q does not mention %q", err, advice)
		}
	}

	estimate.Prefetch = false
	if !Plan(budget, estimate).DeviceFits {
		t.Fatal("disabling prefetch should release enough device memory")
	}
}

func TestLlamaKVCacheBytes(t *testing.T) {
	const want = 256 * 1024 * 1024
	if got := LlamaKVCacheBytes(32, 2048, 8, 128, 2); got != want {
		t.Fatalf("LlamaKVCacheBytes() = %d, want %d", got, want)
	}
	if got := LlamaKVCacheBytes(0, 2048, 8, 128, 2); got != 0 {
		t.Fatalf("LlamaKVCacheBytes() with zero layers = %d", got)
	}
	if got := LlamaKVCacheBytes(math.MaxUint64, 2, 1, 1, 1); got != math.MaxUint64 {
		t.Fatalf("LlamaKVCacheBytes() overflow = %d, want MaxUint64", got)
	}
}

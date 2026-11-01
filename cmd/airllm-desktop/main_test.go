package main

import (
	"strings"
	"testing"

	"github.com/lutikk/airllm/pkg/hardware"
)

func TestParsePositiveInt(t *testing.T) {
	value, err := parsePositiveInt(" 2048 ", "контекст")
	if err != nil || value != 2048 {
		t.Fatalf("parsePositiveInt() = %d, %v; want 2048, nil", value, err)
	}
	for _, input := range []string{"", "0", "-1", "1.5", "text"} {
		if _, err := parsePositiveInt(input, "контекст"); err == nil {
			t.Errorf("parsePositiveInt(%q) unexpectedly succeeded", input)
		}
	}
}

func TestHardwareSummaryIncludesCPULimitation(t *testing.T) {
	summary := hardwareSummary(hardware.Snapshot{
		CPU: hardware.CPU{Model: "Test CPU", Cores: 8, Arch: "arm64", OS: "darwin", RAMBytes: 16 << 30},
		GPU: &hardware.GPU{Name: "Test GPU"},
	})
	for _, expected := range []string{"Test CPU", "16.0 ГиБ", "Test GPU", "CPU reference"} {
		if !strings.Contains(summary, expected) {
			t.Errorf("hardware summary %q does not contain %q", summary, expected)
		}
	}
}

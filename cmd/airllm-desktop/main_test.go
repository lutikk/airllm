package main

import (
	"strings"
	"testing"

	"fyne.io/fyne/v2/widget"
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

func TestReadDownloadParams(t *testing.T) {
	ui := &desktopUI{
		modelEntry:    widget.NewEntry(),
		modelDirEntry: widget.NewEntry(),
		sourceSelect:  widget.NewSelect([]string{sourceHuggingFace, sourceOllama}, nil),
	}
	ui.modelEntry.SetText(" tinyllama:latest ")
	ui.modelDirEntry.SetText(" /models ")
	ui.sourceSelect.SetSelected(sourceOllama)
	params, err := ui.readDownloadParams()
	if err != nil {
		t.Fatalf("readDownloadParams: %v", err)
	}
	if params.model != "tinyllama:latest" || params.modelDir != "/models" || params.source != sourceOllama {
		t.Fatalf("unexpected params: %+v", params)
	}

	ui.sourceSelect.Selected = ""
	if _, err := ui.readDownloadParams(); err == nil {
		t.Fatal("readDownloadParams accepted an empty source")
	}
}

func TestSafeModelReference(t *testing.T) {
	if got := safeModelReference("registry.ollama.ai/library/tinyllama:latest"); got != "registry.ollama.ai_library_tinyllama_latest" {
		t.Fatalf("safeModelReference = %q", got)
	}
	if got := safeModelReference("../модель"); strings.ContainsAny(got, "/\\") {
		t.Fatalf("unsafe model reference %q", got)
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

package main

import (
	"strings"
	"testing"

	"github.com/lutikk/airllm/internal/ollama"
)

func TestPlanOllamaDownload(t *testing.T) {
	const gib = uint64(1 << 30)
	tests := []struct {
		name           string
		modelSize      int64
		config         string
		freeBytes      uint64
		wantConversion uint64
		wantEstimated  bool
	}{
		{
			name:           "llama includes estimated F16 conversion",
			modelSize:      int64(2 * gib),
			config:         `{"architecture":"llama","parameter_size":"1.1B"}`,
			freeBytes:      5 * gib,
			wantConversion: 2_200_000_000,
			wantEstimated:  true,
		},
		{
			name:          "gptoss native has no conversion estimate",
			modelSize:     int64(12 * gib),
			config:        `{"architecture":"gptoss","parameter_size":"20B"}`,
			freeBytes:     13 * gib,
			wantEstimated: false,
		},
		{
			name:          "unknown architecture is allowed",
			modelSize:     int64(2 * gib),
			config:        `{"architecture":"new-architecture"}`,
			freeBytes:     3 * gib,
			wantEstimated: false,
		},
		{
			name:          "llama without parameter metadata is allowed",
			modelSize:     int64(2 * gib),
			config:        `{"architecture":"llama"}`,
			freeBytes:     3 * gib,
			wantEstimated: false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plan, err := planOllamaDownload(ollama.Inspection{ModelSize: test.modelSize, Config: []byte(test.config)}, test.freeBytes)
			if err != nil {
				t.Fatalf("planOllamaDownload: %v", err)
			}
			if plan.ConversionBytes != test.wantConversion || plan.ConversionEstimated != test.wantEstimated {
				t.Fatalf("conversion = %d, estimated = %t", plan.ConversionBytes, plan.ConversionEstimated)
			}
			wantRequired := uint64(test.modelSize) + test.wantConversion + ollamaDownloadMarginBytes
			if plan.RequiredBytes != wantRequired {
				t.Fatalf("required = %d, want %d", plan.RequiredBytes, wantRequired)
			}
		})
	}
}

func TestPlanOllamaDownloadRejectsInsufficientDisk(t *testing.T) {
	inspection := ollama.Inspection{
		ModelSize: 4 << 30,
		Config:    []byte(`{"architecture":"llama","parameter_size":"1B"}`),
	}
	_, err := planOllamaDownload(inspection, 5<<30)
	if err == nil || !strings.Contains(err.Error(), "недостаточно места") {
		t.Fatalf("planOllamaDownload error = %v", err)
	}
}

func TestOllamaParameterCount(t *testing.T) {
	for input, want := range map[string]uint64{"0.5B": 500_000_000, "20B": 20_000_000_000, "700M": 700_000_000} {
		got, ok := ollamaParameterCount(ollamaConfig{ParameterSize: input})
		if !ok || got != want {
			t.Errorf("ollamaParameterCount(%q) = %d, %t; want %d, true", input, got, ok, want)
		}
	}
	got, ok := ollamaParameterCount(ollamaConfig{Parameters: "123456"})
	if !ok || got != 123456 {
		t.Errorf("numeric parameters = %d, %t", got, ok)
	}
}

func TestFormatDownloadSize(t *testing.T) {
	if got := "Будет скачано " + formatByteSize(6<<30) + "."; got != "Будет скачано 6.0 ГиБ." {
		t.Fatalf("download status = %q", got)
	}
}

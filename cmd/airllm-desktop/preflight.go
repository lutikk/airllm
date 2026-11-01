package main

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/lutikk/airllm/internal/ollama"
)

const ollamaDownloadMarginBytes uint64 = 64 << 20

type ollamaDownloadPlan struct {
	Architecture        string
	DownloadBytes       uint64
	ConversionBytes     uint64
	RequiredBytes       uint64
	ConversionEstimated bool
}

type ollamaConfig struct {
	Architecture  string      `json:"architecture"`
	ParameterSize string      `json:"parameter_size"`
	Parameters    json.Number `json:"parameters"`
}

func planOllamaDownload(inspection ollama.Inspection, freeBytes uint64) (ollamaDownloadPlan, error) {
	if inspection.ModelSize < 0 {
		return ollamaDownloadPlan{}, fmt.Errorf("registry вернул отрицательный размер GGUF")
	}
	plan := ollamaDownloadPlan{DownloadBytes: uint64(inspection.ModelSize)}
	var config ollamaConfig
	if len(inspection.Config) > 0 {
		if err := json.Unmarshal(inspection.Config, &config); err != nil {
			return plan, fmt.Errorf("разбор config.json Ollama: %w", err)
		}
	}
	plan.Architecture = strings.ToLower(strings.TrimSpace(config.Architecture))
	plan.RequiredBytes = plan.DownloadBytes
	if plan.Architecture == "llama" {
		if parameters, ok := ollamaParameterCount(config); ok && parameters <= math.MaxUint64/2 {
			plan.ConversionBytes = parameters * 2
			plan.ConversionEstimated = true
			plan.RequiredBytes = addUint64(plan.RequiredBytes, plan.ConversionBytes)
		}
	}
	plan.RequiredBytes = addUint64(plan.RequiredBytes, ollamaDownloadMarginBytes)
	if freeBytes < plan.RequiredBytes {
		return plan, fmt.Errorf(
			"недостаточно места: требуется %s (GGUF и временные файлы), свободно %s",
			formatByteSize(plan.RequiredBytes), formatByteSize(freeBytes),
		)
	}
	return plan, nil
}

func ollamaParameterCount(config ollamaConfig) (uint64, bool) {
	if config.Parameters != "" {
		value, err := strconv.ParseUint(config.Parameters.String(), 10, 64)
		return value, err == nil && value > 0
	}
	value := strings.TrimSpace(config.ParameterSize)
	if value == "" {
		return 0, false
	}
	unit := uint64(1)
	last := value[len(value)-1]
	switch last {
	case 'K', 'k':
		unit = 1_000
		value = value[:len(value)-1]
	case 'M', 'm':
		unit = 1_000_000
		value = value[:len(value)-1]
	case 'B', 'b':
		unit = 1_000_000_000
		value = value[:len(value)-1]
	case 'T', 't':
		unit = 1_000_000_000_000
		value = value[:len(value)-1]
	}
	number, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
	if err != nil || number <= 0 || math.IsNaN(number) || math.IsInf(number, 0) || number > float64(math.MaxUint64/unit) {
		return 0, false
	}
	return uint64(number * float64(unit)), true
}

func addUint64(left, right uint64) uint64 {
	if math.MaxUint64-left < right {
		return math.MaxUint64
	}
	return left + right
}

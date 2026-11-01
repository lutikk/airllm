// Package gptoss implements native-GGUF GPT-OSS greedy inference.
package gptoss

import (
	"fmt"
	"math"

	"github.com/lutikk/airllm/internal/gguf"
)

const (
	defaultSlidingWindow         = 128
	defaultOriginalContextLength = 4096
)

// Config contains the GPT-OSS dimensions needed by the reference runtime.
type Config struct {
	HiddenSize            int
	HeadDim               int
	NumHeads              int
	NumKVHeads            int
	NumLayers             int
	IntermediateSize      int
	NumExperts            int
	NumExpertsPerToken    int
	VocabSize             int
	RMSNormEpsilon        float64
	RoPETheta             float64
	ContextLength         int
	OriginalContextLength int
	SlidingWindow         int
}

// LoadConfig extracts GPT-OSS metadata without converting the GGUF file.
func LoadConfig(reader *gguf.Reader) (Config, error) {
	if reader == nil {
		return Config{}, fmt.Errorf("gptoss: nil GGUF reader")
	}
	metadata, err := reader.GPTOSSMetadata()
	if err != nil {
		return Config{}, fmt.Errorf("gptoss: %w", err)
	}
	config := Config{
		HiddenSize:            metadata.HiddenSize,
		NumHeads:              metadata.NumHeads,
		NumKVHeads:            metadata.NumKVHeads,
		NumLayers:             metadata.NumLayers,
		IntermediateSize:      metadata.ExpertIntermediateSize,
		NumExperts:            metadata.NumExperts,
		NumExpertsPerToken:    metadata.NumExpertsPerToken,
		VocabSize:             metadata.VocabSize,
		RMSNormEpsilon:        metadata.RMSNormEpsilon,
		RoPETheta:             metadata.RoPETheta,
		ContextLength:         metadata.ContextLength,
		OriginalContextLength: metadata.OriginalContextLength,
		SlidingWindow:         metadata.SlidingWindow,
	}
	if config.OriginalContextLength == 0 {
		config.OriginalContextLength = defaultOriginalContextLength
	}
	if config.SlidingWindow == 0 {
		config.SlidingWindow = defaultSlidingWindow
	}
	config.HeadDim = config.HiddenSize / config.NumHeads
	prefix := metadata.Architecture + "."
	raw := reader.Metadata()
	for _, key := range []string{prefix + "attention.key_length", prefix + "attention.value_length"} {
		if value, ok := raw[key]; ok {
			dimension, err := metadataDimension(value)
			if err != nil || dimension <= 0 {
				return Config{}, fmt.Errorf("gptoss: invalid metadata %q value %v", key, value)
			}
			config.HeadDim = dimension
			break
		}
	}
	if config.OriginalContextLength != defaultOriginalContextLength {
		return Config{}, fmt.Errorf("gptoss: YaRN original context %d, want %d", config.OriginalContextLength, defaultOriginalContextLength)
	}
	if err := config.Validate(); err != nil {
		return Config{}, fmt.Errorf("gptoss: invalid config: %w", err)
	}
	return config, nil
}

// Validate checks dimensions used by the decoder and native tensor loader.
func (c Config) Validate() error {
	positive := []struct {
		name  string
		value int
	}{
		{"hidden size", c.HiddenSize},
		{"head dim", c.HeadDim},
		{"heads", c.NumHeads},
		{"KV heads", c.NumKVHeads},
		{"layers", c.NumLayers},
		{"expert intermediate size", c.IntermediateSize},
		{"experts", c.NumExperts},
		{"experts per token", c.NumExpertsPerToken},
		{"vocabulary", c.VocabSize},
		{"context length", c.ContextLength},
	}
	for _, field := range positive {
		if field.value <= 0 {
			return fmt.Errorf("%s must be positive, got %d", field.name, field.value)
		}
	}
	if c.HeadDim%2 != 0 {
		return fmt.Errorf("head dim must be even, got %d", c.HeadDim)
	}
	if c.NumKVHeads > c.NumHeads || c.NumHeads%c.NumKVHeads != 0 {
		return fmt.Errorf("heads %d must be divisible by KV heads %d", c.NumHeads, c.NumKVHeads)
	}
	if c.NumExpertsPerToken > c.NumExperts {
		return fmt.Errorf("experts per token %d exceeds experts %d", c.NumExpertsPerToken, c.NumExperts)
	}
	if c.RMSNormEpsilon <= 0 || math.IsNaN(c.RMSNormEpsilon) || math.IsInf(c.RMSNormEpsilon, 0) {
		return fmt.Errorf("RMS norm epsilon must be finite and positive")
	}
	if c.RoPETheta <= 0 || math.IsNaN(c.RoPETheta) || math.IsInf(c.RoPETheta, 0) {
		return fmt.Errorf("RoPE theta must be finite and positive")
	}
	if c.SlidingWindow <= 0 {
		return fmt.Errorf("sliding window must be positive, got %d", c.SlidingWindow)
	}
	return nil
}

func metadataDimension(value any) (int, error) {
	var integer int64
	switch value := value.(type) {
	case uint8:
		integer = int64(value)
	case uint16:
		integer = int64(value)
	case uint32:
		integer = int64(value)
	case uint64:
		if value > math.MaxInt64 {
			return 0, fmt.Errorf("integer overflow")
		}
		integer = int64(value)
	case int8:
		integer = int64(value)
	case int16:
		integer = int64(value)
	case int32:
		integer = int64(value)
	case int64:
		integer = value
	default:
		return 0, fmt.Errorf("%v (%T) is not an integer", value, value)
	}
	converted := int(integer)
	if int64(converted) != integer {
		return 0, fmt.Errorf("integer exceeds platform capacity")
	}
	return converted, nil
}

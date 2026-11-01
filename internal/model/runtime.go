// Package model defines the common text-generation runtime contract.
package model

import "context"

// Runtime is implemented by both the safetensors Llama and native-GGUF GPT-OSS
// runtimes.
type Runtime interface {
	Encode(text string, addSpecialTokens bool) ([]uint32, error)
	Decode(ids []uint32, skipSpecialTokens bool) (string, error)
	Generate(ctx context.Context, promptTokens []uint32, maxNew int, onToken func(uint32) error) error
	Close() error
}

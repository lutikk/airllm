package gguf

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	runtimeTokenizer "github.com/lutikk/airllm/internal/tokenizer"
)

const o200kSplitPattern = `[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

// PrepareGPTOSSTokenizer extracts the o200k tokenizer next to a native GPT-OSS
// GGUF. It reads metadata only and never reads or converts model tensors.
func PrepareGPTOSSTokenizer(ggufPath, destDir string) (_ string, returnErr error) {
	reader, err := Open(ggufPath)
	if err != nil {
		return "", err
	}
	defer func() {
		if closeErr := reader.Close(); returnErr == nil && closeErr != nil {
			returnErr = fmt.Errorf("gguf: close input: %w", closeErr)
		}
	}()
	metadata, err := reader.GPTOSSMetadata()
	if err != nil {
		return "", err
	}
	tokens, merges, err := gptossTokenizerParts(reader.metadata)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return "", fmt.Errorf("gguf: create destination %q: %w", destDir, err)
	}
	path := filepath.Join(destDir, "tokenizer.json")
	if err := writeJSONAtomic(path, buildGPTOSSTokenizer(tokens, merges, metadata.BOSTokenID, metadata.EOSTokenID)); err != nil {
		return "", err
	}
	if _, err := runtimeTokenizer.NewFromFile(path); err != nil {
		return "", fmt.Errorf("gguf: prepared GPT-OSS tokenizer is not supported: %w", err)
	}
	return path, nil
}

func gptossTokenizerParts(metadata map[string]any) ([]string, []string, error) {
	model, ok := metadata["tokenizer.ggml.model"].(string)
	if !ok || model != "gpt2" {
		return nil, nil, fmt.Errorf("gguf: GPT-OSS conversion requires tokenizer.ggml.model=gpt2, got %v", metadata["tokenizer.ggml.model"])
	}
	tokens, ok := metadata["tokenizer.ggml.tokens"].([]string)
	if !ok || len(tokens) == 0 {
		return nil, nil, errors.New("gguf: tokenizer.ggml.tokens is missing or empty")
	}
	merges, ok := metadata["tokenizer.ggml.merges"].([]string)
	if !ok {
		return nil, nil, errors.New("gguf: tokenizer.ggml.merges is missing")
	}
	seen := make(map[string]struct{}, len(tokens))
	for id, token := range tokens {
		if token == "" {
			return nil, nil, fmt.Errorf("gguf: tokenizer token %d is empty", id)
		}
		if _, exists := seen[token]; exists {
			return nil, nil, fmt.Errorf("gguf: duplicate tokenizer token %q", token)
		}
		seen[token] = struct{}{}
	}
	return tokens, merges, nil
}

func buildGPTOSSTokenizer(tokens, merges []string, bosID, eosID int64) map[string]any {
	vocab := make(map[string]int, len(tokens))
	added := make([]map[string]any, 0)
	for id, token := range tokens {
		vocab[token] = id
		if int64(id) == bosID || int64(id) == eosID || strings.HasPrefix(token, "<|") && strings.HasSuffix(token, "|>") {
			added = append(added, map[string]any{
				"id": id, "content": token, "single_word": false,
				"lstrip": false, "rstrip": false, "normalized": false, "special": true,
			})
		}
	}
	byteLevel := func(addPrefixSpace, trimOffsets, useRegex bool) map[string]any {
		return map[string]any{
			"type": "ByteLevel", "add_prefix_space": addPrefixSpace,
			"trim_offsets": trimOffsets, "use_regex": useRegex,
		}
	}
	return map[string]any{
		"version":      "1.0",
		"truncation":   nil,
		"padding":      nil,
		"added_tokens": added,
		"normalizer":   nil,
		"pre_tokenizer": map[string]any{
			"type": "Sequence",
			"pretokenizers": []any{
				map[string]any{
					"type": "Split", "pattern": map[string]any{"Regex": o200kSplitPattern},
					"behavior": "isolated", "invert": false,
				},
				byteLevel(false, false, false),
			},
		},
		"post_processor": nil,
		"decoder":        byteLevel(true, true, true),
		"model": map[string]any{
			"type": "BPE", "dropout": nil, "unk_token": nil,
			"continuing_subword_prefix": nil, "end_of_word_suffix": nil,
			"fuse_unk": false, "byte_fallback": false,
			"vocab": vocab, "merges": merges,
		},
	}
}

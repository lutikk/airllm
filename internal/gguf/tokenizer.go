package gguf

import (
	"errors"
	"fmt"
)

func tokenizerParts(metadata map[string]any) ([]string, []string, error) {
	model, ok := metadata["tokenizer.ggml.model"].(string)
	if !ok || model != "llama" && model != "gpt2" {
		return nil, nil, fmt.Errorf("gguf: tokenizer extraction supports tokenizer.ggml.model=llama or gpt2, got %v", metadata["tokenizer.ggml.model"])
	}
	tokens, ok := metadata["tokenizer.ggml.tokens"].([]string)
	if !ok || len(tokens) == 0 {
		return nil, nil, errors.New("gguf: tokenizer.ggml.tokens is missing or empty")
	}
	rawMerges, ok := metadata["tokenizer.ggml.merges"].([]string)
	if !ok {
		return nil, nil, errors.New("gguf: tokenizer.ggml.merges is missing; converting a SentencePiece score vocabulary without GGUF merges is not supported")
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
	return tokens, rawMerges, nil
}

func unknownTokenID(metadata map[string]any, tokens []string) (int, error) {
	if raw, ok := metadata["tokenizer.ggml.unknown_token_id"]; ok {
		value, err := numberInt(raw)
		if err != nil || value < 0 || value >= int64(len(tokens)) {
			return 0, fmt.Errorf("gguf: invalid tokenizer.ggml.unknown_token_id %v", raw)
		}
		return int(value), nil
	}
	for id, token := range tokens {
		if token == "<unk>" {
			return id, nil
		}
	}
	return 0, errors.New("gguf: tokenizer has no unknown_token_id or <unk> token")
}

func buildTokenizer(tokens, merges []string, bosID, eosID, unknownID int) map[string]any {
	vocab := make(map[string]int, len(tokens))
	for id, token := range tokens {
		vocab[token] = id
	}
	bos, eos, unknown := tokens[bosID], tokens[eosID], tokens[unknownID]
	added := make([]map[string]any, 0, 3)
	seen := make(map[int]struct{})
	for _, entry := range []struct {
		id    int
		token string
	}{{bosID, bos}, {eosID, eos}, {unknownID, unknown}} {
		if _, ok := seen[entry.id]; ok {
			continue
		}
		seen[entry.id] = struct{}{}
		added = append(added, map[string]any{
			"id": entry.id, "content": entry.token, "single_word": false,
			"lstrip": false, "rstrip": false, "normalized": false, "special": true,
		})
	}
	return map[string]any{
		"version":      "1.0",
		"truncation":   nil,
		"padding":      nil,
		"added_tokens": added,
		"normalizer":   nil,
		"pre_tokenizer": map[string]any{
			"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true,
		},
		"post_processor": map[string]any{
			"type": "TemplateProcessing",
			"single": []any{
				map[string]any{"SpecialToken": map[string]any{"id": bos, "type_id": 0}},
				map[string]any{"Sequence": map[string]any{"id": "A", "type_id": 0}},
			},
			"pair": []any{
				map[string]any{"SpecialToken": map[string]any{"id": bos, "type_id": 0}},
				map[string]any{"Sequence": map[string]any{"id": "A", "type_id": 0}},
				map[string]any{"Sequence": map[string]any{"id": "B", "type_id": 1}},
			},
			"special_tokens": map[string]any{
				bos: map[string]any{"id": bos, "ids": []int{bosID}, "tokens": []string{bos}},
			},
		},
		"decoder": map[string]any{
			"type": "Sequence",
			"decoders": []any{
				map[string]any{"type": "Replace", "pattern": map[string]any{"String": "▁"}, "content": " "},
				map[string]any{"type": "ByteFallback"},
				map[string]any{"type": "Fuse"},
				map[string]any{"type": "Strip", "content": " ", "start": 1, "stop": 0},
			},
		},
		"model": map[string]any{
			"type": "BPE", "dropout": nil, "unk_token": unknown,
			"continuing_subword_prefix": nil, "end_of_word_suffix": nil,
			"fuse_unk": false, "byte_fallback": true,
			"vocab": vocab, "merges": merges,
		},
	}
}

func buildGPT2Tokenizer(tokens, merges []string, eosID int, metadata map[string]any) map[string]any {
	vocab := make(map[string]int, len(tokens))
	for id, token := range tokens {
		vocab[token] = id
	}
	special := map[int]struct{}{eosID: {}}
	if rawTypes, ok := metadata["tokenizer.ggml.token_type"].([]any); ok && len(rawTypes) == len(tokens) {
		for id, raw := range rawTypes {
			tokenType, err := numberInt(raw)
			if err == nil && (tokenType == 3 || tokenType == 4) {
				special[id] = struct{}{}
			}
		}
	}
	added := make([]any, 0, len(special))
	for id := 0; id < len(tokens); id++ {
		if _, ok := special[id]; !ok {
			continue
		}
		added = append(added, map[string]any{
			"id": id, "content": tokens[id], "single_word": false,
			"lstrip": false, "rstrip": false, "normalized": false, "special": true,
		})
	}
	return map[string]any{
		"version":      "1.0",
		"truncation":   nil,
		"padding":      nil,
		"added_tokens": added,
		"normalizer":   nil,
		"pre_tokenizer": map[string]any{
			"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true,
		},
		"post_processor": nil,
		"decoder":        map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
		"model": map[string]any{
			"type": "BPE", "dropout": nil, "unk_token": nil,
			"continuing_subword_prefix": nil, "end_of_word_suffix": nil,
			"fuse_unk": false, "byte_fallback": false,
			"vocab": vocab, "merges": merges,
		},
	}
}

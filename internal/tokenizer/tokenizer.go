// Package tokenizer loads Hugging Face tokenizer.json files without cgo or
// external runtime libraries.
package tokenizer

import (
	"fmt"
	"strings"

	sugarme "github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/decoder"
	"github.com/sugarme/tokenizer/model/bpe"
	"github.com/sugarme/tokenizer/normalizer"
	"github.com/sugarme/tokenizer/pretokenizer"
	"github.com/sugarme/tokenizer/processor"
)

// Tokenizer is a Hugging Face tokenizer. It is safe to use after construction
// without any Python or native tokenizer runtime.
type Tokenizer struct {
	inner *sugarme.Tokenizer
}

// NewFromFile loads a tokenizer.json file. The supported configuration covers
// TinyLlama/Llama-style BPE, Metaspace, TemplateProcessing, and decoder chains.
func NewFromFile(path string) (*Tokenizer, error) {
	config, err := sugarme.ConfigFromFile(path)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer config: %w", err)
	}
	model, err := newModel(config.Model)
	if err != nil {
		return nil, err
	}
	inner := sugarme.NewTokenizer(model)

	if config.Normalizer != nil {
		normalizerChain, err := newNormalizer(config.Normalizer)
		if err != nil {
			return nil, err
		}
		inner.WithNormalizer(normalizerChain)
	}
	if config.PreTokenizer != nil {
		preTokenizer, err := newPreTokenizer(config.PreTokenizer)
		if err != nil {
			return nil, err
		}
		inner.WithPreTokenizer(preTokenizer)
	}
	if config.PostProcessor != nil {
		postProcessor, err := newPostProcessor(config.PostProcessor)
		if err != nil {
			return nil, err
		}
		inner.WithPostProcessor(postProcessor)
	}
	if config.Decoder != nil {
		decoderChain, err := newDecoder(config.Decoder)
		if err != nil {
			return nil, err
		}
		inner.WithDecoder(decoderChain)
	}

	for _, token := range config.AddedTokens {
		added := sugarme.NewAddedToken(token.Content, token.Special,
			sugarme.WithSingleWord(token.SingleWord),
			sugarme.WithLStrip(token.Lstrip),
			sugarme.WithRStrip(token.Rstrip),
			sugarme.WithNormalized(token.Normalized),
		)
		if token.Special {
			inner.AddSpecialTokens([]sugarme.AddedToken{added})
		} else {
			inner.AddTokens([]sugarme.AddedToken{added})
		}
	}

	return &Tokenizer{inner: inner}, nil
}

// Encode converts text to token IDs, optionally applying the tokenizer's
// post-processor (for example, adding TinyLlama's BOS token).
func (t *Tokenizer) Encode(text string, addSpecialTokens bool) ([]uint32, error) {
	encoding, err := t.inner.EncodeSingle(text, addSpecialTokens)
	if err != nil {
		return nil, fmt.Errorf("encode text: %w", err)
	}
	ids := make([]uint32, len(encoding.Ids))
	for i, id := range encoding.Ids {
		if id < 0 {
			return nil, fmt.Errorf("encode text: negative token ID %d", id)
		}
		ids[i] = uint32(id)
	}
	return ids, nil
}

// Decode converts token IDs back to text.
func (t *Tokenizer) Decode(ids []uint32, skipSpecialTokens bool) (string, error) {
	intIDs := make([]int, len(ids))
	for i, id := range ids {
		if uint64(id) > uint64(^uint(0)>>1) {
			return "", fmt.Errorf("decode tokens: token ID %d overflows int", id)
		}
		intIDs[i] = int(id)
		if _, ok := t.inner.IdToToken(intIDs[i]); !ok {
			return "", fmt.Errorf("decode tokens: unknown token ID %d", id)
		}
	}
	return t.inner.Decode(intIDs, skipSpecialTokens), nil
}

func newModel(config map[string]interface{}) (sugarme.Model, error) {
	if config["type"] != "BPE" {
		return nil, fmt.Errorf("unsupported model type %q", config["type"])
	}
	// Marshal the model through ConfigFromFile's typed representation by using
	// the same JSON field types, without writing an intermediate file.
	vocab, ok := config["vocab"].(map[string]int)
	if !ok {
		// Config.Model is map[string]interface{}, so vocab arrives as a generic
		// map. Handle it explicitly rather than relying on reflection.
		generic, genericOK := config["vocab"].(map[string]interface{})
		if !genericOK {
			return nil, fmt.Errorf("invalid BPE vocab")
		}
		vocab = make(map[string]int, len(generic))
		for token, value := range generic {
			id, ok := value.(float64)
			if !ok || id < 0 || id != float64(int(id)) {
				return nil, fmt.Errorf("invalid BPE ID for token %q", token)
			}
			vocab[token] = int(id)
		}
	}

	merges, err := stringSlice(config["merges"])
	if err != nil {
		return nil, fmt.Errorf("invalid BPE merges: %w", err)
	}
	var unkToken *string
	if value, ok := config["unk_token"].(string); ok && value != "" {
		unkToken = &value
	}
	if config["continuing_subword_prefix"] != nil || config["end_of_word_suffix"] != nil {
		return nil, fmt.Errorf("BPE subword prefixes and suffixes are not supported")
	}
	base, err := bpe.New(vocab, merges, nil, unkToken, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("create BPE model: %w", err)
	}

	byteFallback, _ := config["byte_fallback"].(bool)
	if !byteFallback {
		return base, nil
	}
	byteTokens := make(map[byte]int, 256)
	for value := 0; value < 256; value++ {
		token := fmt.Sprintf("<0x%02X>", value)
		if id, ok := vocab[token]; ok {
			byteTokens[byte(value)] = id
		}
	}
	if unkToken == nil {
		return nil, fmt.Errorf("BPE byte fallback requires an unknown token")
	}
	return &byteFallbackModel{
		Model:      base,
		unkID:      vocab[*unkToken],
		byteTokens: byteTokens,
	}, nil
}

// byteFallbackModel adds the HF BPE byte_fallback behavior missing from
// sugarme/tokenizer's BPE implementation.
type byteFallbackModel struct {
	sugarme.Model
	unkID      int
	byteTokens map[byte]int
}

func (m *byteFallbackModel) Tokenize(sequence string) ([]sugarme.Token, error) {
	tokens, err := m.Model.Tokenize(sequence)
	if err != nil {
		return nil, err
	}
	var result []sugarme.Token
	for _, token := range tokens {
		if token.Id != m.unkID || len(token.Offsets) != 2 {
			result = append(result, token)
			continue
		}
		start, end := token.Offsets[0], token.Offsets[1]
		if start < 0 || end < start || end > len(sequence) {
			return nil, fmt.Errorf("invalid unknown-token offsets %d:%d", start, end)
		}
		for i, value := range []byte(sequence[start:end]) {
			id, ok := m.byteTokens[value]
			if !ok {
				return nil, fmt.Errorf("missing byte-fallback token <0x%02X>", value)
			}
			offset := start + i
			result = append(result, sugarme.NewToken(id, fmt.Sprintf("<0x%02X>", value), []int{offset, offset + 1}))
		}
	}
	return result, nil
}

func newNormalizer(config map[string]interface{}) (normalizer.Normalizer, error) {
	switch config["type"] {
	case "Sequence":
		rawNormalizers, ok := config["normalizers"].([]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid normalizer sequence")
		}
		normalizers := make([]normalizer.Normalizer, 0, len(rawNormalizers))
		for _, raw := range rawNormalizers {
			value, ok := raw.(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("invalid normalizer in sequence")
			}
			item, err := newNormalizer(value)
			if err != nil {
				return nil, err
			}
			normalizers = append(normalizers, item)
		}
		return normalizer.NewSequence(normalizers), nil
	case "Prepend":
		prepend, _ := config["prepend"].(string)
		return normalizer.NewPrepend(prepend), nil
	case "Replace":
		pattern, _ := config["pattern"].(map[string]interface{})
		content, _ := config["content"].(string)
		if value, ok := pattern["String"].(string); ok {
			return normalizer.NewReplace(normalizer.String, value, content), nil
		}
		if value, ok := pattern["Regex"].(string); ok {
			return normalizer.NewReplace(normalizer.Regex, value, content), nil
		}
		return nil, fmt.Errorf("unsupported Replace normalizer pattern")
	default:
		return nil, fmt.Errorf("unsupported normalizer type %q", config["type"])
	}
}

func newPreTokenizer(config map[string]interface{}) (sugarme.PreTokenizer, error) {
	if config["type"] != "Metaspace" {
		return nil, fmt.Errorf("unsupported pre-tokenizer type %q", config["type"])
	}
	replacement, _ := config["replacement"].(string)
	if replacement == "" {
		replacement = "▁"
	}
	scheme := pretokenizer.Never
	switch config["prepend_scheme"] {
	case "always":
		scheme = pretokenizer.Always
	case "first":
		scheme = pretokenizer.First
	case "never", nil:
		if addPrefix, _ := config["add_prefix_space"].(bool); addPrefix {
			scheme = pretokenizer.Always
		}
	default:
		return nil, fmt.Errorf("unsupported Metaspace prepend scheme %q", config["prepend_scheme"])
	}
	return pretokenizer.NewMetaspaceWithScheme(replacement, scheme), nil
}

func newPostProcessor(config map[string]interface{}) (sugarme.PostProcessor, error) {
	if config["type"] != "TemplateProcessing" {
		return nil, fmt.Errorf("unsupported post-processor type %q", config["type"])
	}
	single, err := newTemplate(config["single"])
	if err != nil {
		return nil, fmt.Errorf("invalid single template: %w", err)
	}
	pair, err := newTemplate(config["pair"])
	if err != nil {
		return nil, fmt.Errorf("invalid pair template: %w", err)
	}

	rawSpecialTokens, ok := config["special_tokens"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid TemplateProcessing special tokens")
	}
	specialTokens := make([]processor.SpecialToken, 0, len(rawSpecialTokens))
	for name, raw := range rawSpecialTokens {
		value, ok := raw.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid special token %q", name)
		}
		ids, err := intSlice(value["ids"])
		if err != nil {
			return nil, fmt.Errorf("invalid special token %q IDs: %w", name, err)
		}
		tokens, err := stringSlice(value["tokens"])
		if err != nil {
			return nil, fmt.Errorf("invalid special token %q tokens: %w", name, err)
		}
		specialTokens = append(specialTokens, *processor.NewSpecialToken(name, ids, tokens))
	}
	return processor.NewTemplateProcessing(single, pair, processor.NewTokensFrom(specialTokens)), nil
}

func newTemplate(raw interface{}) (processor.Template, error) {
	parts, ok := raw.([]interface{})
	if !ok {
		return nil, fmt.Errorf("expected an array")
	}
	pieces := make([]string, len(parts))
	for i, rawPart := range parts {
		part, ok := rawPart.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid template piece %d", i)
		}
		if _, ok := part["Sequence"]; ok {
			pieces[i] = "$A"
		} else if special, ok := part["SpecialToken"].(map[string]interface{}); ok {
			id, _ := special["id"].(string)
			pieces[i] = id
		} else {
			return nil, fmt.Errorf("unsupported template piece %d", i)
		}
		if typeID, ok := templateTypeID(part); ok {
			pieces[i] += fmt.Sprintf(":%d", typeID)
		}
	}
	return processor.NewTemplateFromMulti(pieces)
}

func templateTypeID(part map[string]interface{}) (int, bool) {
	for _, key := range []string{"Sequence", "SpecialToken"} {
		if value, ok := part[key].(map[string]interface{}); ok {
			if id, ok := value["type_id"].(float64); ok {
				return int(id), true
			}
		}
	}
	return 0, false
}

func newDecoder(config map[string]interface{}) (sugarme.Decoder, error) {
	switch config["type"] {
	case "Sequence":
		rawDecoders, ok := config["decoders"].([]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid decoder sequence")
		}
		decoders := make([]sugarme.Decoder, 0, len(rawDecoders))
		for _, raw := range rawDecoders {
			value, ok := raw.(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("invalid decoder in sequence")
			}
			item, err := newDecoder(value)
			if err != nil {
				return nil, err
			}
			decoders = append(decoders, item)
		}
		return decoder.NewSequence(decoders), nil
	case "Replace":
		pattern, _ := config["pattern"].(map[string]interface{})
		value, _ := pattern["String"].(string)
		content, _ := config["content"].(string)
		return &replaceDecoder{old: value, new: content}, nil
	case "ByteFallback":
		return decoder.NewByteFallback(), nil
	case "Fuse":
		return decoder.NewFuse(), nil
	case "Strip":
		content, _ := config["content"].(string)
		return decoder.NewStrip(content, intNumber(config["start"]), intNumber(config["stop"])), nil
	case "Metaspace":
		replacement, _ := config["replacement"].(string)
		addPrefix, _ := config["add_prefix_space"].(bool)
		return pretokenizer.NewMetaspace(replacement, addPrefix), nil
	default:
		return nil, fmt.Errorf("unsupported decoder type %q", config["type"])
	}
}

type replaceDecoder struct {
	old string
	new string
}

func (d *replaceDecoder) Decode(tokens []string) string {
	return strings.Join(d.DecodeChain(tokens), "")
}

func (d *replaceDecoder) DecodeChain(tokens []string) []string {
	result := make([]string, len(tokens))
	for i, token := range tokens {
		result[i] = strings.ReplaceAll(token, d.old, d.new)
	}
	return result
}

func stringSlice(raw interface{}) ([]string, error) {
	if raw == nil {
		return nil, nil
	}
	values, ok := raw.([]interface{})
	if !ok {
		if strings, ok := raw.([]string); ok {
			return strings, nil
		}
		return nil, fmt.Errorf("expected an array of strings")
	}
	result := make([]string, len(values))
	for i, value := range values {
		text, ok := value.(string)
		if !ok {
			return nil, fmt.Errorf("element %d is not a string", i)
		}
		result[i] = text
	}
	return result, nil
}

func intSlice(raw interface{}) ([]int, error) {
	values, ok := raw.([]interface{})
	if !ok {
		return nil, fmt.Errorf("expected an array of integers")
	}
	result := make([]int, len(values))
	for i, value := range values {
		number, ok := value.(float64)
		if !ok || number != float64(int(number)) {
			return nil, fmt.Errorf("element %d is not an integer", i)
		}
		result[i] = int(number)
	}
	return result, nil
}

func intNumber(raw interface{}) int {
	value, _ := raw.(float64)
	return int(value)
}

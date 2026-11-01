package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/sugarme/tokenizer/pretokenizer"
)

func TestSplitO200KComponents(t *testing.T) {
	for _, test := range []struct {
		text string
		want []string
	}{
		{"hello world", []string{"hello", " world"}},
		{"!hello", []string{"!hello"}},
		{"1234", []string{"123", "4"}},
		{" \tfoo", []string{" ", "\tfoo"}},
		{"foo  \nbar", []string{"foo", "  \n", "bar"}},
		{"punctuation!!!\n\n", []string{"punctuation", "!!!\n\n"}},
		{"  ", []string{"  "}},
	} {
		ranges := splitO200K(test.text)
		got := make([]string, len(ranges))
		for i, bounds := range ranges {
			got[i] = test.text[bounds[0]:bounds[1]]
		}
		if !reflect.DeepEqual(got, test.want) {
			t.Errorf("splitO200K(%q) = %#v, want %#v", test.text, got, test.want)
		}
	}
}

func TestByteLevelO200KBPEAndDecoder(t *testing.T) {
	vocab := byteLevelVocab()
	merges := []string{"h e", "he l", "hel l", "hell o", "! hello", "1 2", "12 3"}
	for _, token := range []string{"he", "hel", "hell", "hello", "!hello", "12", "123"} {
		vocab[token] = len(vocab)
	}
	startID := len(vocab)
	vocab["<|start|>"] = startID
	config := map[string]any{
		"version": "1.0", "truncation": nil, "padding": nil,
		"added_tokens": []any{map[string]any{
			"id": startID, "content": "<|start|>", "single_word": false,
			"lstrip": false, "rstrip": false, "normalized": false, "special": true,
		}},
		"normalizer": nil,
		"pre_tokenizer": map[string]any{
			"type": "Sequence",
			"pretokenizers": []any{
				map[string]any{"type": "Split", "pattern": map[string]any{"Regex": o200kPattern}, "behavior": "isolated", "invert": false},
				map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
			},
		},
		"post_processor": nil,
		"decoder":        map[string]any{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
		"model": map[string]any{
			"type": "BPE", "dropout": nil, "unk_token": nil,
			"continuing_subword_prefix": nil, "end_of_word_suffix": nil,
			"fuse_unk": false, "byte_fallback": false, "vocab": vocab, "merges": merges,
		},
	}
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	data, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	tok, err := NewFromFile(path)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode("!hello 1234", false)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint32{uint32(vocab["!hello"]), uint32(vocab["Ġ"]), uint32(vocab["123"]), uint32(vocab["4"])}
	if !reflect.DeepEqual(ids, want) {
		t.Fatalf("Encode() = %v, want %v", ids, want)
	}
	decoded, err := tok.Decode(ids, true)
	if err != nil {
		t.Fatal(err)
	}
	if decoded != "!hello 1234" {
		t.Fatalf("Decode() = %q", decoded)
	}

	const text = "<|start|>Привет, 世界!\n12345"
	ids, err = tok.Encode(text, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) == 0 || ids[0] != uint32(startID) {
		t.Fatalf("special token was not preserved: %v", ids)
	}
	decoded, err = tok.Decode(ids, false)
	if err != nil {
		t.Fatal(err)
	}
	if decoded != text {
		t.Fatalf("round trip = %q, want %q", decoded, text)
	}
}

func byteLevelVocab() map[string]int {
	vocab := make(map[string]int, 256)
	for value := 0; value < 256; value++ {
		vocab[pretokenizer.BytesChar[uint8(value)]] = value
	}
	return vocab
}

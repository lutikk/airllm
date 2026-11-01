package tokenizer_test

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/lutikk/airllm/internal/tokenizer"
)

func newTestTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	tok, err := tokenizer.NewFromFile(filepath.Join("testdata", "tiny_tokenizer.json"))
	if err != nil {
		t.Fatalf("NewFromFile() error = %v", err)
	}
	return tok
}

func TestEncodeWithoutSpecialTokens(t *testing.T) {
	tok := newTestTokenizer(t)
	got, err := tok.Encode("hello world", false)
	if err != nil {
		t.Fatalf("Encode() error = %v", err)
	}
	want := []uint32{15, 20}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Encode() = %v, want %v", got, want)
	}
}

func TestEncodeWithSpecialTokens(t *testing.T) {
	tok := newTestTokenizer(t)
	got, err := tok.Encode("hello world", true)
	if err != nil {
		t.Fatalf("Encode() error = %v", err)
	}
	want := []uint32{1, 15, 20}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Encode() = %v, want %v", got, want)
	}
}

func TestDecode(t *testing.T) {
	tok := newTestTokenizer(t)
	got, err := tok.Decode([]uint32{1, 15, 20, 2}, true)
	if err != nil {
		t.Fatalf("Decode() error = %v", err)
	}
	if want := "hello world"; got != want {
		t.Fatalf("Decode() = %q, want %q", got, want)
	}

	got, err = tok.Decode([]uint32{1, 15, 2}, false)
	if err != nil {
		t.Fatalf("Decode() with special tokens error = %v", err)
	}
	if want := "<s> hello</s>"; got != want {
		t.Fatalf("Decode() with special tokens = %q, want %q", got, want)
	}
}

func TestByteFallbackRoundTrip(t *testing.T) {
	tok := newTestTokenizer(t)
	ids, err := tok.Encode("é", false)
	if err != nil {
		t.Fatalf("Encode() error = %v", err)
	}
	wantIDs := []uint32{3, 21, 22}
	if !reflect.DeepEqual(ids, wantIDs) {
		t.Fatalf("Encode() = %v, want %v", ids, wantIDs)
	}
	got, err := tok.Decode(ids, true)
	if err != nil {
		t.Fatalf("Decode() error = %v", err)
	}
	if want := "é"; got != want {
		t.Fatalf("Decode() = %q, want %q", got, want)
	}
}

func TestDecodeRejectsUnknownID(t *testing.T) {
	tok := newTestTokenizer(t)
	if _, err := tok.Decode([]uint32{999}, true); err == nil {
		t.Fatal("Decode() error = nil, want unknown-token error")
	}
}

func TestNewFromFileErrors(t *testing.T) {
	if _, err := tokenizer.NewFromFile(filepath.Join("testdata", "missing.json")); err == nil {
		t.Fatal("NewFromFile() error = nil, want file error")
	}
}

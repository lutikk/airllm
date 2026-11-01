package tokenizer

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	sugarme "github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/normalizer"
	"github.com/sugarme/tokenizer/pretokenizer"
)

const o200kPattern = `[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

type preTokenizerSequence []sugarme.PreTokenizer

func (s preTokenizerSequence) PreTokenize(input *sugarme.PreTokenizedString) (*sugarme.PreTokenizedString, error) {
	var err error
	for _, item := range s {
		input, err = item.PreTokenize(input)
		if err != nil {
			return nil, err
		}
	}
	return input, nil
}

// byteLevelWithoutRegex performs only ByteLevel's alphabet transform. The
// library ByteLevel always applies its own GPT-2 splitter, which is incorrect
// after an o200k Split (for example, it separates the '!' from '!hello').
type byteLevelWithoutRegex struct {
	addPrefixSpace bool
}

func (b byteLevelWithoutRegex) PreTokenize(input *sugarme.PreTokenizedString) (*sugarme.PreTokenizedString, error) {
	return input.Normalize(func(normalized *normalizer.NormalizedString) *normalizer.NormalizedString {
		if b.addPrefixSpace && !strings.HasPrefix(normalized.GetNormalized(), " ") {
			normalized = normalized.Prepend(" ")
		}
		changes := make([]normalizer.ChangeMap, 0, len(normalized.GetNormalized()))
		for _, r := range normalized.GetNormalized() {
			for i, value := range []byte(string(r)) {
				change := 0
				if i > 0 {
					change = 1
				}
				changes = append(changes, normalizer.ChangeMap{RuneVal: pretokenizer.BytesChar[value], Changes: change})
			}
		}
		return normalized.Transform(changes, 0)
	}), nil
}

type o200kSplitPreTokenizer struct{}

func (o200kSplitPreTokenizer) PreTokenize(input *sugarme.PreTokenizedString) (*sugarme.PreTokenizedString, error) {
	return input.Split(func(_ int, normalized *normalizer.NormalizedString) []sugarme.SplitIdx {
		text := normalized.GetNormalized()
		ranges := splitO200K(text)
		result := make([]sugarme.SplitIdx, 0, len(ranges))
		for _, bounds := range ranges {
			part := normalized.Slice(normalizer.NewRange(bounds[0], bounds[1], normalizer.NormalizedTarget))
			if part != nil {
				result = append(result, sugarme.SplitIdx{Normalized: part})
			}
		}
		return result
	}), nil
}

func newO200KOrRegexpSplit(pattern string, behavior normalizer.SplitDelimiterBehavior, invert bool) (sugarme.PreTokenizer, error) {
	if pattern == o200kPattern {
		if behavior != normalizer.IsolatedBehavior || invert {
			return nil, fmt.Errorf("o200k Split supports only non-inverted isolated behavior")
		}
		return o200kSplitPreTokenizer{}, nil
	}
	if _, err := regexp.Compile(pattern); err != nil {
		return nil, fmt.Errorf("invalid Split regex: %w", err)
	}
	return pretokenizer.NewSplit(normalizer.NewRegexpPattern(pattern), behavior, invert), nil
}

// splitO200K implements the six ordered alternatives of the o200k_base regular
// expression. Ranges are byte offsets into text and cover it without gaps.
func splitO200K(text string) [][2]int {
	result := make([][2]int, 0, len(text)/4+1)
	for offset := 0; offset < len(text); {
		end := o200kTokenEnd(text, offset)
		if end <= offset {
			_, size := utf8.DecodeRuneInString(text[offset:])
			end = offset + size
		}
		result = append(result, [2]int{offset, end})
		offset = end
	}
	return result
}

func o200kTokenEnd(text string, start int) int {
	r, size := utf8.DecodeRuneInString(text[start:])
	next := start + size

	// [^\r\n\p{L}\p{N}]?\p{L}+
	letters := start
	if !unicode.IsLetter(r) && !unicode.IsNumber(r) && r != '\r' && r != '\n' {
		letters = next
	}
	if letters < len(text) {
		first, _ := utf8.DecodeRuneInString(text[letters:])
		if unicode.IsLetter(first) {
			end := letters
			for end < len(text) {
				r, size := utf8.DecodeRuneInString(text[end:])
				if !unicode.IsLetter(r) {
					break
				}
				end += size
			}
			return end
		}
	}

	// \p{N}{1,3}
	if unicode.IsNumber(r) {
		end, count := start, 0
		for end < len(text) && count < 3 {
			r, size := utf8.DecodeRuneInString(text[end:])
			if !unicode.IsNumber(r) {
				break
			}
			end += size
			count++
		}
		return end
	}

	//  ?[^\s\p{L}\p{N}]+[\r\n]*
	punctuation := start
	if r == ' ' && next < len(text) {
		punctuation = next
	}
	if punctuation < len(text) {
		first, _ := utf8.DecodeRuneInString(text[punctuation:])
		if isO200KPunctuation(first) {
			end := punctuation
			for end < len(text) {
				r, size := utf8.DecodeRuneInString(text[end:])
				if !isO200KPunctuation(r) {
					break
				}
				end += size
			}
			for end < len(text) {
				r, size := utf8.DecodeRuneInString(text[end:])
				if r != '\r' && r != '\n' {
					break
				}
				end += size
			}
			return end
		}
	}

	if unicode.IsSpace(r) {
		end, lastNewline := start, -1
		for end < len(text) {
			r, size := utf8.DecodeRuneInString(text[end:])
			if !unicode.IsSpace(r) {
				break
			}
			end += size
			if r == '\r' || r == '\n' {
				lastNewline = end
			}
		}
		// \s*[\r\n]+ takes precedence over either trailing-space branch.
		if lastNewline >= 0 {
			return lastNewline
		}
		if end < len(text) {
			// \s+(?!\S) leaves the final whitespace rune for \s+.
			_, lastSize := utf8.DecodeLastRuneInString(text[start:end])
			return end - lastSize
		}
		return end
	}

	return next
}

func isO200KPunctuation(r rune) bool {
	return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
}

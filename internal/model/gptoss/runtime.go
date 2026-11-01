package gptoss

import (
	"context"
	"fmt"
	"path/filepath"
	"sync"

	"github.com/lutikk/airllm/internal/backend/cpuref"
	"github.com/lutikk/airllm/internal/tokenizer"
)

const (
	returnTokenID = uint32(200002)
	stopTokenID   = uint32(200012)
)

// Runtime runs batch-one greedy inference directly from a native GGUF file.
// Decoder layers are resident one at a time; expert weights are read only after
// top-k routing. Generate calls are serialized.
type Runtime struct {
	source        *tensorSource
	tokenizer     *tokenizer.Tokenizer
	config        Config
	contextLength int
	embeddings    []float32
	finalNorm     []float32
	lmHead        []float32
	stop          map[uint32]struct{}

	mu     sync.Mutex
	closed bool
}

// Open loads GPT-OSS metadata, resident tensors, and tokenizer.json beside the
// native GGUF file.
func Open(path string, contextLength int) (*Runtime, error) {
	return OpenWithTokenizer(path, filepath.Join(filepath.Dir(path), "tokenizer.json"), contextLength)
}

// OpenWithTokenizer is Open with an explicit Hugging Face tokenizer.json path.
func OpenWithTokenizer(path, tokenizerPath string, contextLength int) (_ *Runtime, err error) {
	if contextLength <= 0 {
		return nil, fmt.Errorf("gptoss: context length must be positive: %d", contextLength)
	}
	source, err := openTensorSource(path)
	if err != nil {
		return nil, fmt.Errorf("gptoss: open GGUF: %w", err)
	}
	defer func() {
		if err != nil {
			_ = source.Close()
		}
	}()
	config, err := LoadConfig(source.reader)
	if err != nil {
		return nil, err
	}
	if contextLength > config.ContextLength {
		return nil, fmt.Errorf("gptoss: context length %d exceeds model maximum %d", contextLength, config.ContextLength)
	}
	metadata, err := source.reader.GPTOSSMetadata()
	if err != nil {
		return nil, fmt.Errorf("gptoss: %w", err)
	}
	tok, err := tokenizer.NewFromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("gptoss: open tokenizer: %w", err)
	}
	runtime := &Runtime{
		source:        source,
		tokenizer:     tok,
		config:        config,
		contextLength: contextLength,
		stop: map[uint32]struct{}{
			returnTokenID: {},
			stopTokenID:   {},
		},
	}
	if metadata.EOSTokenID >= 0 {
		runtime.stop[uint32(metadata.EOSTokenID)] = struct{}{}
	}
	if err := runtime.loadResidentWeights(); err != nil {
		return nil, err
	}
	return runtime, nil
}

// Encode converts text to GPT-OSS token IDs without adding Llama BOS tokens.
func (r *Runtime) Encode(text string, addSpecialTokens bool) ([]uint32, error) {
	if r.tokenizer == nil {
		return nil, fmt.Errorf("gptoss: runtime is closed")
	}
	return r.tokenizer.Encode(text, addSpecialTokens)
}

// Decode converts GPT-OSS token IDs to text with the ByteLevel decoder.
func (r *Runtime) Decode(ids []uint32, skipSpecialTokens bool) (string, error) {
	if r.tokenizer == nil {
		return "", fmt.Errorf("gptoss: runtime is closed")
	}
	return r.tokenizer.Decode(ids, skipSpecialTokens)
}

// Generate prefills promptTokens and streams up to maxNew greedy tokens. A stop
// token is delivered to onToken before generation stops.
func (r *Runtime) Generate(ctx context.Context, promptTokens []uint32, maxNew int, onToken func(uint32) error) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return fmt.Errorf("gptoss: runtime is closed")
	}
	if len(promptTokens) == 0 {
		return fmt.Errorf("gptoss: prompt must contain at least one token")
	}
	if maxNew < 0 {
		return fmt.Errorf("gptoss: maxNew must be non-negative: %d", maxNew)
	}
	if len(promptTokens) > r.contextLength {
		return fmt.Errorf("gptoss: prompt length %d exceeds context length %d", len(promptTokens), r.contextLength)
	}
	if maxNew > 0 && len(promptTokens)+maxNew-1 > r.contextLength {
		return fmt.Errorf("gptoss: prompt and generation exceed context length %d", r.contextLength)
	}
	for _, token := range promptTokens {
		if token >= uint32(r.config.VocabSize) {
			return fmt.Errorf("gptoss: prompt token %d is outside vocabulary size %d", token, r.config.VocabSize)
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if maxNew == 0 {
		return nil
	}

	caches := make([]*cpuref.KVCache, r.config.NumLayers)
	for i := range caches {
		cache, err := cpuref.NewKVCache(r.config.NumKVHeads, r.config.HeadDim, r.contextLength)
		if err != nil {
			return fmt.Errorf("gptoss: create layer %d KV cache: %w", i, err)
		}
		caches[i] = cache
	}

	hidden := r.embedSequence(promptTokens)
	for layerIndex := 0; layerIndex < r.config.NumLayers; layerIndex++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		layer, err := r.loadDecoderLayer(layerIndex)
		if err != nil {
			return err
		}
		hidden, err = layer.forwardPrefill(hidden, len(promptTokens), caches[layerIndex], r.source)
		if err != nil {
			return fmt.Errorf("gptoss: prefill layer %d: %w", layerIndex, err)
		}
	}

	lastHidden := hidden[(len(promptTokens)-1)*r.config.HiddenSize:]
	next, err := r.nextToken(lastHidden)
	if err != nil {
		return err
	}
	for generated := 0; generated < maxNew; generated++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if onToken != nil {
			if err := onToken(next); err != nil {
				return fmt.Errorf("gptoss: token callback: %w", err)
			}
		}
		if _, stop := r.stop[next]; stop {
			return nil
		}
		if generated+1 == maxNew {
			return nil
		}

		hidden = r.embedSequence([]uint32{next})
		for layerIndex := 0; layerIndex < r.config.NumLayers; layerIndex++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			layer, err := r.loadDecoderLayer(layerIndex)
			if err != nil {
				return err
			}
			hidden, err = layer.forwardDecode(hidden, caches[layerIndex], r.source)
			if err != nil {
				return fmt.Errorf("gptoss: decode layer %d: %w", layerIndex, err)
			}
		}
		next, err = r.nextToken(hidden)
		if err != nil {
			return err
		}
	}
	return nil
}

// Close releases GGUF handles and resident tensors. It is idempotent.
func (r *Runtime) Close() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	r.embeddings = nil
	r.finalNorm = nil
	r.lmHead = nil
	r.tokenizer = nil
	if r.source != nil {
		return r.source.Close()
	}
	return nil
}

func (r *Runtime) embedSequence(tokens []uint32) []float32 {
	hidden := make([]float32, len(tokens)*r.config.HiddenSize)
	for i, token := range tokens {
		start := int(token) * r.config.HiddenSize
		copy(hidden[i*r.config.HiddenSize:(i+1)*r.config.HiddenSize], r.embeddings[start:start+r.config.HiddenSize])
	}
	return hidden
}

func (r *Runtime) nextToken(hidden []float32) (uint32, error) {
	normalized, err := cpuref.RMSNorm(hidden, r.finalNorm, float32(r.config.RMSNormEpsilon))
	if err != nil {
		return 0, fmt.Errorf("gptoss: final norm: %w", err)
	}
	logits, err := cpuref.Linear(normalized, r.lmHead, r.config.HiddenSize, r.config.VocabSize)
	if err != nil {
		return 0, fmt.Errorf("gptoss: lm head: %w", err)
	}
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return uint32(best), nil
}

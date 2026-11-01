package llama

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"sync"

	"github.com/lutikk/airllm/internal/backend/cpuref"
	"github.com/lutikk/airllm/internal/tokenizer"
	weightindex "github.com/lutikk/airllm/internal/weights/index"
)

// Runtime runs batch-one greedy inference while streaming decoder weights from
// disk. Resident tensors are limited to embeddings, final norm, and lm_head;
// decoder layers are loaded one at a time. A Runtime serializes Generate calls.
type Runtime struct {
	model         *weightindex.Model
	tokenizer     *tokenizer.Tokenizer
	config        Config
	contextLength int
	embeddings    []float32
	finalNorm     []float32
	lmHead        []float32
	eos           map[uint32]struct{}

	mu     sync.Mutex
	closed bool
}

// Open loads model metadata and resident weights from modelDir. contextLength
// bounds the total prompt plus generated-token history.
func Open(modelDir string, contextLength int) (_ *Runtime, err error) {
	if contextLength <= 0 {
		return nil, fmt.Errorf("llama: context length must be positive: %d", contextLength)
	}
	config, err := LoadConfig(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return nil, err
	}
	if contextLength > config.MaxPositionEmbeddings {
		return nil, fmt.Errorf("llama: context length %d exceeds model maximum %d", contextLength, config.MaxPositionEmbeddings)
	}
	model, err := weightindex.OpenModel(modelDir)
	if err != nil {
		return nil, fmt.Errorf("llama: open weights: %w", err)
	}
	defer func() {
		if err != nil {
			_ = model.Close()
		}
	}()

	tokenizer, err := tokenizer.NewFromFile(filepath.Join(modelDir, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("llama: open tokenizer: %w", err)
	}
	runtime := &Runtime{
		model:         model,
		tokenizer:     tokenizer,
		config:        config,
		contextLength: contextLength,
		eos:           make(map[uint32]struct{}, len(config.EOSTokenIDs)),
	}
	for _, id := range config.EOSTokenIDs {
		runtime.eos[uint32(id)] = struct{}{}
	}
	if err := runtime.loadResidentWeights(); err != nil {
		return nil, err
	}
	return runtime, nil
}

// Encode converts text to model token IDs using the runtime tokenizer.
func (r *Runtime) Encode(text string, addSpecialTokens bool) ([]uint32, error) {
	if r.tokenizer == nil {
		return nil, fmt.Errorf("llama: runtime is closed")
	}
	return r.tokenizer.Encode(text, addSpecialTokens)
}

// Decode converts model token IDs to text using the runtime tokenizer.
func (r *Runtime) Decode(ids []uint32, skipSpecialTokens bool) (string, error) {
	if r.tokenizer == nil {
		return "", fmt.Errorf("llama: runtime is closed")
	}
	return r.tokenizer.Decode(ids, skipSpecialTokens)
}

// Generate prefills promptTokens and streams up to maxNew greedy tokens. The
// EOS token, when selected, is delivered to onToken before generation stops.
func (r *Runtime) Generate(ctx context.Context, promptTokens []uint32, maxNew int, onToken func(uint32) error) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return fmt.Errorf("llama: runtime is closed")
	}
	if len(promptTokens) == 0 {
		return fmt.Errorf("llama: prompt must contain at least one token")
	}
	if maxNew < 0 {
		return fmt.Errorf("llama: maxNew must be non-negative: %d", maxNew)
	}
	if len(promptTokens) > r.contextLength {
		return fmt.Errorf("llama: prompt length %d exceeds context length %d", len(promptTokens), r.contextLength)
	}
	if maxNew > 0 && len(promptTokens)+maxNew-1 > r.contextLength {
		return fmt.Errorf("llama: prompt and generation require %d positions, context length is %d", len(promptTokens)+maxNew-1, r.contextLength)
	}
	for _, token := range promptTokens {
		if token >= uint32(r.config.VocabSize) {
			return fmt.Errorf("llama: prompt token %d is outside vocabulary size %d", token, r.config.VocabSize)
		}
	}
	if maxNew == 0 {
		return ctx.Err()
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	caches := make([]*cpuref.KVCache, r.config.NumHiddenLayers)
	for i := range caches {
		cache, err := cpuref.NewKVCache(r.config.NumKeyValueHeads, r.config.HeadDim, r.contextLength)
		if err != nil {
			return fmt.Errorf("llama: create layer %d KV cache: %w", i, err)
		}
		caches[i] = cache
	}

	hidden := r.embedSequence(promptTokens)
	for layerIndex := 0; layerIndex < r.config.NumHiddenLayers; layerIndex++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		layer, err := r.loadDecoderLayer(layerIndex)
		if err != nil {
			return err
		}
		hidden, err = layer.ForwardPrefill(hidden, len(promptTokens), caches[layerIndex])
		if err != nil {
			return fmt.Errorf("llama: prefill layer %d: %w", layerIndex, err)
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
				return fmt.Errorf("llama: token callback: %w", err)
			}
		}
		if _, stop := r.eos[next]; stop {
			return nil
		}
		if generated+1 == maxNew {
			return nil
		}

		hidden = r.embedSequence([]uint32{next})
		for layerIndex := 0; layerIndex < r.config.NumHiddenLayers; layerIndex++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			layer, err := r.loadDecoderLayer(layerIndex)
			if err != nil {
				return err
			}
			hidden, err = layer.ForwardDecode(hidden, caches[layerIndex])
			if err != nil {
				return fmt.Errorf("llama: decode layer %d: %w", layerIndex, err)
			}
		}
		next, err = r.nextToken(hidden)
		if err != nil {
			return err
		}
	}
	return nil
}

// Close releases model shard handles and resident tensors. It is idempotent.
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
	if r.model != nil {
		return r.model.Close()
	}
	return nil
}

func (r *Runtime) embedSequence(tokens []uint32) []float32 {
	hidden := make([]float32, len(tokens)*r.config.HiddenSize)
	for i, token := range tokens {
		start := int(token) * r.config.HiddenSize
		copy(hidden[i*r.config.HiddenSize:(i+1)*r.config.HiddenSize], r.embeddings[start:start+r.config.HiddenSize])
	}
	if r.config.IsGemma() {
		scale := float32(math.Sqrt(float64(r.config.HiddenSize)))
		for i := range hidden {
			hidden[i] *= scale
		}
	}
	return hidden
}

func (r *Runtime) nextToken(hidden []float32) (uint32, error) {
	var normalized []float32
	var err error
	if r.config.IsGemma() {
		normalized, err = cpuref.RMSNormPlusOne(hidden, r.finalNorm, float32(r.config.RMSNormEps))
	} else {
		normalized, err = cpuref.RMSNorm(hidden, r.finalNorm, float32(r.config.RMSNormEps))
	}
	if err != nil {
		return 0, fmt.Errorf("llama: final norm: %w", err)
	}
	logits, err := cpuref.Linear(normalized, r.lmHead, r.config.HiddenSize, r.config.VocabSize)
	if err != nil {
		return 0, fmt.Errorf("llama: lm head: %w", err)
	}
	if r.config.FinalLogitSoftcapping > 0 {
		cap := r.config.FinalLogitSoftcapping
		for i, logit := range logits {
			logits[i] = float32(cap * math.Tanh(float64(logit)/cap))
		}
	}
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return uint32(best), nil
}

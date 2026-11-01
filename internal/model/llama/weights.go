package llama

import (
	"fmt"

	"github.com/lutikk/airllm/internal/backend/cpuref"
	weightindex "github.com/lutikk/airllm/internal/weights/index"
	"github.com/lutikk/airllm/internal/weights/safetensors"
)

func (r *Runtime) loadResidentWeights() error {
	var err error
	r.embeddings, err = readTensor(r.model, "model.embed_tokens.weight", r.config.VocabSize, r.config.HiddenSize)
	if err != nil {
		return err
	}
	r.finalNorm, err = readTensor(r.model, "model.norm.weight", r.config.HiddenSize)
	if err != nil {
		return err
	}
	if r.config.TieWordEmbeddings {
		// lm_head uses the same [vocab, hidden] matrix and backing array.
		r.lmHead = r.embeddings
		return nil
	}
	r.lmHead, err = readTensor(r.model, "lm_head.weight", r.config.VocabSize, r.config.HiddenSize)
	return err
}

func (r *Runtime) loadDecoderLayer(layer int) (cpuref.DecoderLayer, error) {
	prefix := fmt.Sprintf("model.layers.%d.", layer)
	hidden := r.config.HiddenSize
	kvWidth := r.config.NumKeyValueHeads * r.config.HeadDim
	intermediate := r.config.IntermediateSize
	weights := cpuref.LayerWeights{}
	var err error
	if weights.Q, err = readTensor(r.model, prefix+"self_attn.q_proj.weight", hidden, hidden); err != nil {
		return cpuref.DecoderLayer{}, err
	}
	if weights.K, err = readTensor(r.model, prefix+"self_attn.k_proj.weight", kvWidth, hidden); err != nil {
		return cpuref.DecoderLayer{}, err
	}
	if weights.V, err = readTensor(r.model, prefix+"self_attn.v_proj.weight", kvWidth, hidden); err != nil {
		return cpuref.DecoderLayer{}, err
	}
	if weights.O, err = readTensor(r.model, prefix+"self_attn.o_proj.weight", hidden, hidden); err != nil {
		return cpuref.DecoderLayer{}, err
	}
	if weights.Gate, err = readTensor(r.model, prefix+"mlp.gate_proj.weight", intermediate, hidden); err != nil {
		return cpuref.DecoderLayer{}, err
	}
	if weights.Up, err = readTensor(r.model, prefix+"mlp.up_proj.weight", intermediate, hidden); err != nil {
		return cpuref.DecoderLayer{}, err
	}
	if weights.Down, err = readTensor(r.model, prefix+"mlp.down_proj.weight", hidden, intermediate); err != nil {
		return cpuref.DecoderLayer{}, err
	}
	if weights.Norm1, err = readTensor(r.model, prefix+"input_layernorm.weight", hidden); err != nil {
		return cpuref.DecoderLayer{}, err
	}
	if weights.Norm2, err = readTensor(r.model, prefix+"post_attention_layernorm.weight", hidden); err != nil {
		return cpuref.DecoderLayer{}, err
	}
	return cpuref.DecoderLayer{
		Config: cpuref.LayerConfig{
			HiddenSize:       hidden,
			IntermediateSize: intermediate,
			NumQueryHeads:    r.config.NumAttentionHeads,
			NumKVHeads:       r.config.NumKeyValueHeads,
			HeadDim:          r.config.HeadDim,
			NormEpsilon:      float32(r.config.RMSNormEps),
			RoPETheta:        r.config.RopeTheta,
		},
		Weights: weights,
	}, nil
}

func readTensor(model *weightindex.Model, name string, shape ...int) ([]float32, error) {
	ref, ok := model.Tensor(name)
	if !ok {
		return nil, fmt.Errorf("llama: tensor %q not found", name)
	}
	if len(ref.Shape) != len(shape) {
		return nil, fmt.Errorf("llama: tensor %q shape %v, want %v", name, ref.Shape, shape)
	}
	for i, dimension := range shape {
		if ref.Shape[i] != int64(dimension) {
			return nil, fmt.Errorf("llama: tensor %q shape %v, want %v", name, ref.Shape, shape)
		}
	}
	data, err := model.ReadTensor(name)
	if err != nil {
		return nil, fmt.Errorf("llama: %w", err)
	}
	var values []float32
	switch ref.DType {
	case safetensors.F16:
		values, err = cpuref.DecodeF16LE(data, shape...)
	case safetensors.BF16:
		values, err = cpuref.DecodeBF16LE(data, shape...)
	case safetensors.F32:
		values, err = cpuref.DecodeF32LE(data, shape...)
	default:
		return nil, fmt.Errorf("llama: tensor %q has unsupported dtype %q", name, ref.DType)
	}
	if err != nil {
		return nil, fmt.Errorf("llama: decode tensor %q: %w", name, err)
	}
	return values, nil
}

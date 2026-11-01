package gptoss

import "fmt"

type layerWeights struct {
	qkv, qkvBias       []float32
	output, outputBias []float32
	sinks              []float32
	norm1, norm2       []float32
	router, routerBias []float32

	gateUp, gateUpBias expertTensor
	down, downBias     expertTensor
	gate, gateBias     expertTensor
	up, upBias         expertTensor
	fusedGateUp        bool
}

func (r *Runtime) loadResidentWeights() error {
	var err error
	if r.embeddings, err = r.source.read("token_embd.weight", r.config.HiddenSize, r.config.VocabSize); err != nil {
		return err
	}
	if r.finalNorm, err = r.source.read("output_norm.weight", r.config.HiddenSize); err != nil {
		return err
	}
	if _, ok := r.source.reader.Tensor("output.weight"); ok {
		r.lmHead, err = r.source.read("output.weight", r.config.HiddenSize, r.config.VocabSize)
	} else {
		r.lmHead = r.embeddings
	}
	return err
}

func (r *Runtime) loadDecoderLayer(index int) (DecoderLayer, error) {
	config := r.config
	prefix := fmt.Sprintf("blk.%d.", index)
	queryWidth := config.NumHeads * config.HeadDim
	kvWidth := config.NumKVHeads * config.HeadDim
	qkvWidth := queryWidth + 2*kvWidth
	weights := layerWeights{}
	var err error
	read := func(name string, destination *[]float32, shape ...int) bool {
		*destination, err = r.source.read(prefix+name, shape...)
		return err == nil
	}
	for _, item := range []struct {
		name        string
		destination *[]float32
		shape       []int
	}{
		{"attn_norm.weight", &weights.norm1, []int{config.HiddenSize}},
		{"ffn_gate_inp.weight", &weights.router, []int{config.HiddenSize, config.NumExperts}},
		{"ffn_gate_inp.bias", &weights.routerBias, []int{config.NumExperts}},
	} {
		if !read(item.name, item.destination, item.shape...) {
			return DecoderLayer{}, fmt.Errorf("gptoss: load layer %d: %w", index, err)
		}
	}
	if _, fused := r.source.reader.Tensor(prefix + "attn_qkv.weight"); fused {
		weights.qkv, err = r.source.read(prefix+"attn_qkv.weight", config.HiddenSize, qkvWidth)
		if err == nil {
			weights.qkvBias, err = r.source.read(prefix+"attn_qkv.bias", qkvWidth)
		}
	} else {
		weights.qkv, weights.qkvBias, err = r.source.readSeparateQKV(prefix, config.HiddenSize, queryWidth, kvWidth)
	}
	if err != nil {
		return DecoderLayer{}, fmt.Errorf("gptoss: load layer %d QKV: %w", index, err)
	}
	if weights.sinks, err = r.source.readAlias(prefix, []string{"attn_sinks.weight", "attn_sinks"}, config.NumHeads); err != nil {
		return DecoderLayer{}, fmt.Errorf("gptoss: load layer %d sinks: %w", index, err)
	}
	if weights.output, err = r.source.readAlias(prefix, []string{"attn_output.weight", "attn_out.weight"}, queryWidth, config.HiddenSize); err != nil {
		return DecoderLayer{}, fmt.Errorf("gptoss: load layer %d output: %w", index, err)
	}
	if weights.outputBias, err = r.source.readAlias(prefix, []string{"attn_output.bias", "attn_out.bias"}, config.HiddenSize); err != nil {
		return DecoderLayer{}, fmt.Errorf("gptoss: load layer %d output bias: %w", index, err)
	}
	if weights.norm2, err = r.source.readAlias(prefix, []string{"ffn_norm.weight", "post_attention_norm.weight"}, config.HiddenSize); err != nil {
		return DecoderLayer{}, fmt.Errorf("gptoss: load layer %d post-attention norm: %w", index, err)
	}

	if _, ok := r.source.reader.Tensor(prefix + "ffn_gate_up_exps.weight"); ok {
		weights.fusedGateUp = true
		weights.gateUp, err = r.source.expert(prefix+"ffn_gate_up_exps.weight", config.NumExperts, config.HiddenSize, 2*config.IntermediateSize)
		if err == nil {
			weights.gateUpBias, err = r.source.expert(prefix+"ffn_gate_up_exps.bias", config.NumExperts, 2*config.IntermediateSize)
		}
	} else {
		for _, item := range []struct {
			name        string
			destination *expertTensor
			shape       []int
		}{
			{"ffn_gate_exps.weight", &weights.gate, []int{config.HiddenSize, config.IntermediateSize}},
			{"ffn_gate_exps.bias", &weights.gateBias, []int{config.IntermediateSize}},
			{"ffn_up_exps.weight", &weights.up, []int{config.HiddenSize, config.IntermediateSize}},
			{"ffn_up_exps.bias", &weights.upBias, []int{config.IntermediateSize}},
		} {
			*item.destination, err = r.source.expert(prefix+item.name, config.NumExperts, item.shape...)
			if err != nil {
				break
			}
		}
	}
	if err != nil {
		return DecoderLayer{}, fmt.Errorf("gptoss: load layer %d expert gate/up metadata: %w", index, err)
	}
	weights.down, err = r.source.expert(prefix+"ffn_down_exps.weight", config.NumExperts, config.IntermediateSize, config.HiddenSize)
	if err == nil {
		weights.downBias, err = r.source.expert(prefix+"ffn_down_exps.bias", config.NumExperts, config.HiddenSize)
	}
	if err != nil {
		return DecoderLayer{}, fmt.Errorf("gptoss: load layer %d expert down metadata: %w", index, err)
	}

	window := 0
	if index%2 == 0 {
		window = config.SlidingWindow
	}
	return DecoderLayer{config: config, weights: weights, slidingWindow: window}, nil
}

func (s *tensorSource) readAlias(prefix string, names []string, shape ...int) ([]float32, error) {
	for _, name := range names {
		if _, ok := s.reader.Tensor(prefix + name); ok {
			return s.read(prefix+name, shape...)
		}
	}
	return nil, fmt.Errorf("gptoss: none of tensors %v found", names)
}

func (s *tensorSource) readSeparateQKV(prefix string, hidden, queryWidth, kvWidth int) ([]float32, []float32, error) {
	q, err := s.read(prefix+"attn_q.weight", hidden, queryWidth)
	if err != nil {
		return nil, nil, err
	}
	k, err := s.read(prefix+"attn_k.weight", hidden, kvWidth)
	if err != nil {
		return nil, nil, err
	}
	v, err := s.read(prefix+"attn_v.weight", hidden, kvWidth)
	if err != nil {
		return nil, nil, err
	}
	qBias, err := s.read(prefix+"attn_q.bias", queryWidth)
	if err != nil {
		return nil, nil, err
	}
	kBias, err := s.read(prefix+"attn_k.bias", kvWidth)
	if err != nil {
		return nil, nil, err
	}
	vBias, err := s.read(prefix+"attn_v.bias", kvWidth)
	if err != nil {
		return nil, nil, err
	}

	weights := make([]float32, 0, len(q)+len(k)+len(v))
	weights = append(weights, q...)
	weights = append(weights, k...)
	weights = append(weights, v...)
	biases := make([]float32, 0, len(qBias)+len(kBias)+len(vBias))
	biases = append(biases, qBias...)
	biases = append(biases, kBias...)
	biases = append(biases, vBias...)
	return weights, biases, nil
}

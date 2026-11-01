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
		{"attn_qkv.weight", &weights.qkv, []int{config.HiddenSize, qkvWidth}},
		{"attn_qkv.bias", &weights.qkvBias, []int{qkvWidth}},
		{"attn_sinks.weight", &weights.sinks, []int{config.NumHeads}},
		{"attn_output.weight", &weights.output, []int{queryWidth, config.HiddenSize}},
		{"attn_output.bias", &weights.outputBias, []int{config.HiddenSize}},
		{"ffn_norm.weight", &weights.norm2, []int{config.HiddenSize}},
		{"ffn_gate_inp.weight", &weights.router, []int{config.HiddenSize, config.NumExperts}},
		{"ffn_gate_inp.bias", &weights.routerBias, []int{config.NumExperts}},
	} {
		if !read(item.name, item.destination, item.shape...) {
			return DecoderLayer{}, fmt.Errorf("gptoss: load layer %d: %w", index, err)
		}
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

package gguf

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Convert converts a supported GGUF file into config.json, tokenizer.json, and
// one F16 model.safetensors file. Only one source tensor is held in memory at a
// time; no model-wide weight buffer is allocated.
func Convert(ctx context.Context, ggufPath, destDir string) (returnErr error) {
	if err := ctx.Err(); err != nil {
		return err
	}
	reader, err := Open(ggufPath)
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := reader.Close(); returnErr == nil && closeErr != nil {
			returnErr = fmt.Errorf("gguf: close input: %w", closeErr)
		}
	}()

	architecture, err := metadataString(reader.metadata, "general.architecture")
	if err != nil {
		return err
	}
	switch architecture {
	case "llama":
		return convertLlama(ctx, reader, destDir)
	case "gptoss", "gpt-oss":
		return convertGPTOSS(ctx, reader, destDir)
	default:
		return fmt.Errorf("gguf: unsupported architecture %q", architecture)
	}
}

func convertLlama(ctx context.Context, reader *Reader, destDir string) error {
	metadata, err := reader.LlamaMetadata()
	if err != nil {
		return err
	}
	tokens, merges, err := tokenizerParts(reader.metadata)
	if err != nil {
		return err
	}
	unknownID, err := unknownTokenID(reader.metadata, tokens)
	if err != nil {
		return err
	}
	tensors, tied, err := mapTensors(reader.Tensors())
	if err != nil {
		return err
	}
	if err := validateConvertedShapes(tensors, metadata, tied); err != nil {
		return err
	}
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return fmt.Errorf("gguf: create destination %q: %w", destDir, err)
	}

	if err := writeSafetensors(ctx, reader, tensors, filepath.Join(destDir, "model.safetensors")); err != nil {
		return err
	}
	config := map[string]any{
		"architectures":           []string{"LlamaForCausalLM"},
		"model_type":              "llama",
		"hidden_size":             metadata.HiddenSize,
		"intermediate_size":       metadata.IntermediateSize,
		"num_attention_heads":     metadata.NumHeads,
		"num_key_value_heads":     metadata.NumKVHeads,
		"num_hidden_layers":       metadata.NumLayers,
		"vocab_size":              metadata.VocabSize,
		"rms_norm_eps":            metadata.RMSNormEpsilon,
		"rope_theta":              metadata.RoPETheta,
		"head_dim":                metadata.HiddenSize / metadata.NumHeads,
		"max_position_embeddings": metadata.ContextLength,
		"tie_word_embeddings":     tied,
		"bos_token_id":            metadata.BOSTokenID,
		"eos_token_id":            metadata.EOSTokenID,
	}
	if err := writeJSONAtomic(filepath.Join(destDir, "config.json"), config); err != nil {
		return err
	}
	tokenizer := buildTokenizer(tokens, merges, int(metadata.BOSTokenID), int(metadata.EOSTokenID), unknownID)
	if err := writeJSONAtomic(filepath.Join(destDir, "tokenizer.json"), tokenizer); err != nil {
		return err
	}
	return nil
}

type convertedTensor struct {
	source TensorInfo
	name   string
	shape  []int64
	size   int64
}

func mapTensors(source []TensorInfo) ([]convertedTensor, bool, error) {
	result := make([]convertedTensor, 0, len(source))
	seen := make(map[string]string)
	hasOutput := false
	for _, tensor := range source {
		if tensor.Name == "rope_freqs.weight" || strings.HasSuffix(tensor.Name, ".rope_freqs.weight") {
			continue
		}
		name, ok := hfTensorName(tensor.Name)
		if !ok {
			return nil, false, fmt.Errorf("gguf: no Llama Hugging Face mapping for tensor %q", tensor.Name)
		}
		if previous, exists := seen[name]; exists {
			return nil, false, fmt.Errorf("gguf: tensors %q and %q both map to %q", previous, tensor.Name, name)
		}
		seen[name] = tensor.Name
		if name == "lm_head.weight" {
			hasOutput = true
		}
		elements, err := elementCount(tensor.Shape)
		if err != nil || elements > math.MaxInt64/2 {
			return nil, false, fmt.Errorf("gguf: tensor %q F16 output size overflows int64", tensor.Name)
		}
		shape := make([]int64, len(tensor.Shape))
		for i, dimension := range tensor.Shape {
			if dimension > math.MaxInt64 {
				return nil, false, fmt.Errorf("gguf: tensor %q shape dimension overflows int64", tensor.Name)
			}
			// GGUF stores dimensions in reverse order from Hugging Face.
			shape[len(shape)-1-i] = int64(dimension)
		}
		result = append(result, convertedTensor{source: tensor, name: name, shape: shape, size: int64(elements) * 2})
	}
	if len(result) == 0 {
		return nil, false, errors.New("gguf: no convertible tensors")
	}
	sort.Slice(result, func(i, j int) bool { return result[i].name < result[j].name })
	return result, !hasOutput, nil
}

func hfTensorName(name string) (string, bool) {
	switch name {
	case "token_embd.weight":
		return "model.embed_tokens.weight", true
	case "output_norm.weight":
		return "model.norm.weight", true
	case "output.weight":
		return "lm_head.weight", true
	}
	if !strings.HasPrefix(name, "blk.") {
		return "", false
	}
	parts := strings.Split(name, ".")
	if len(parts) != 4 || parts[0] != "blk" || parts[3] != "weight" {
		return "", false
	}
	if _, err := strconv.ParseUint(parts[1], 10, 31); err != nil {
		return "", false
	}
	mappedName, ok := map[string]string{
		"attn_norm":   "input_layernorm",
		"attn_q":      "self_attn.q_proj",
		"attn_k":      "self_attn.k_proj",
		"attn_v":      "self_attn.v_proj",
		"attn_output": "self_attn.o_proj",
		"ffn_norm":    "post_attention_layernorm",
		"ffn_gate":    "mlp.gate_proj",
		"ffn_up":      "mlp.up_proj",
		"ffn_down":    "mlp.down_proj",
	}[parts[2]]
	if !ok {
		return "", false
	}
	return "model.layers." + parts[1] + "." + mappedName + ".weight", true
}

func validateConvertedShapes(tensors []convertedTensor, metadata LlamaMetadata, tied bool) error {
	expected := map[string][]int64{
		"model.embed_tokens.weight": {int64(metadata.VocabSize), int64(metadata.HiddenSize)},
		"model.norm.weight":         {int64(metadata.HiddenSize)},
	}
	if !tied {
		expected["lm_head.weight"] = []int64{int64(metadata.VocabSize), int64(metadata.HiddenSize)}
	}
	hidden := int64(metadata.HiddenSize)
	kvWidth := int64(metadata.NumKVHeads * (metadata.HiddenSize / metadata.NumHeads))
	intermediate := int64(metadata.IntermediateSize)
	for layer := 0; layer < metadata.NumLayers; layer++ {
		prefix := fmt.Sprintf("model.layers.%d.", layer)
		expected[prefix+"self_attn.q_proj.weight"] = []int64{hidden, hidden}
		expected[prefix+"self_attn.k_proj.weight"] = []int64{kvWidth, hidden}
		expected[prefix+"self_attn.v_proj.weight"] = []int64{kvWidth, hidden}
		expected[prefix+"self_attn.o_proj.weight"] = []int64{hidden, hidden}
		expected[prefix+"mlp.gate_proj.weight"] = []int64{intermediate, hidden}
		expected[prefix+"mlp.up_proj.weight"] = []int64{intermediate, hidden}
		expected[prefix+"mlp.down_proj.weight"] = []int64{hidden, intermediate}
		expected[prefix+"input_layernorm.weight"] = []int64{hidden}
		expected[prefix+"post_attention_layernorm.weight"] = []int64{hidden}
	}
	actual := make(map[string][]int64, len(tensors))
	for _, tensor := range tensors {
		actual[tensor.name] = tensor.shape
	}
	for name, shape := range expected {
		got, ok := actual[name]
		if !ok {
			return fmt.Errorf("gguf: required tensor %q is missing", name)
		}
		if !sameShape(got, shape) {
			return fmt.Errorf("gguf: tensor %q has shape %v, want %v", name, got, shape)
		}
		delete(actual, name)
	}
	if len(actual) != 0 {
		names := make([]string, 0, len(actual))
		for name := range actual {
			names = append(names, name)
		}
		sort.Strings(names)
		return fmt.Errorf("gguf: tensors outside configured Llama architecture: %v", names)
	}
	return nil
}

func sameShape(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func elementCount(shape []uint64) (uint64, error) {
	result := uint64(1)
	for _, dimension := range shape {
		if dimension != 0 && result > math.MaxUint64/dimension {
			return 0, errors.New("element count overflow")
		}
		result *= dimension
	}
	return result, nil
}

type outputHeaderTensor struct {
	DType       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

func writeSafetensors(ctx context.Context, reader *Reader, tensors []convertedTensor, path string) (returnErr error) {
	header := make(map[string]any, len(tensors)+1)
	header["__metadata__"] = map[string]string{"format": "pt"}
	var offset int64
	for _, tensor := range tensors {
		if tensor.size > math.MaxInt64-offset {
			return errors.New("gguf: safetensors output size overflows int64")
		}
		header[tensor.name] = outputHeaderTensor{DType: "F16", Shape: tensor.shape, DataOffsets: [2]int64{offset, offset + tensor.size}}
		offset += tensor.size
	}
	headerData, err := json.Marshal(header)
	if err != nil {
		return fmt.Errorf("gguf: encode safetensors header: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	file, err := os.CreateTemp(filepath.Dir(path), ".model-*.safetensors")
	if err != nil {
		return fmt.Errorf("gguf: create safetensors: %w", err)
	}
	temporary := file.Name()
	defer func() {
		_ = file.Close()
		if returnErr != nil {
			_ = os.Remove(temporary)
		}
	}()
	var length [8]byte
	binary.LittleEndian.PutUint64(length[:], uint64(len(headerData)))
	if _, err := file.Write(length[:]); err != nil {
		return fmt.Errorf("gguf: write safetensors header length: %w", err)
	}
	if _, err := file.Write(headerData); err != nil {
		return fmt.Errorf("gguf: write safetensors header: %w", err)
	}
	for _, tensor := range tensors {
		if err := ctx.Err(); err != nil {
			return err
		}
		encoded, err := reader.ReadTensor(tensor.source.Name)
		if err != nil {
			return err
		}
		if err := writeF16Tensor(ctx, file, tensor.source, encoded); err != nil {
			return err
		}
	}
	if err := file.Sync(); err != nil {
		return fmt.Errorf("gguf: sync safetensors: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("gguf: close safetensors: %w", err)
	}
	if err := os.Rename(temporary, path); err != nil {
		return fmt.Errorf("gguf: install safetensors: %w", err)
	}
	return nil
}

func writeF16Tensor(ctx context.Context, destination io.Writer, tensor TensorInfo, source []byte) error {
	if int64(len(source)) != tensor.ByteSize {
		return fmt.Errorf("gguf: tensor %q has %d encoded bytes, want %d", tensor.Name, len(source), tensor.ByteSize)
	}
	buffer := make([]byte, 0, 64*1024)
	flush := func() error {
		if len(buffer) == 0 {
			return nil
		}
		if _, err := destination.Write(buffer); err != nil {
			return fmt.Errorf("gguf: write tensor %q: %w", tensor.Name, err)
		}
		buffer = buffer[:0]
		return nil
	}
	appendHalf := func(value uint16) error {
		if len(buffer)+2 > cap(buffer) {
			if err := flush(); err != nil {
				return err
			}
		}
		buffer = append(buffer, byte(value), byte(value>>8))
		return nil
	}
	appendFloat := func(value float32) error {
		return appendHalf(float32ToFloat16(value))
	}

	switch tensor.Type {
	case F16:
		for offset := 0; offset < len(source); offset += 2 {
			if offset%(64*1024) == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			if err := appendHalf(binary.LittleEndian.Uint16(source[offset:])); err != nil {
				return err
			}
		}
	case BF16:
		for offset := 0; offset < len(source); offset += 2 {
			if offset%(64*1024) == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			value := math.Float32frombits(uint32(binary.LittleEndian.Uint16(source[offset:])) << 16)
			if err := appendFloat(value); err != nil {
				return err
			}
		}
	case F32:
		for offset := 0; offset < len(source); offset += 4 {
			if offset%(64*1024) == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			value := math.Float32frombits(binary.LittleEndian.Uint32(source[offset:]))
			if err := appendFloat(value); err != nil {
				return err
			}
		}
	case MXFP4:
		for offset := 0; offset < len(source); offset += 17 {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := appendMXFP4Block(source[offset:offset+17], appendFloat); err != nil {
				return err
			}
		}
	case Q8_0:
		for offset := 0; offset < len(source); offset += 34 {
			if err := ctx.Err(); err != nil {
				return err
			}
			scale := float16ToFloat32(binary.LittleEndian.Uint16(source[offset:]))
			for i := 0; i < 32; i++ {
				if err := appendFloat(scale * float32(int8(source[offset+2+i]))); err != nil {
					return err
				}
			}
		}
	case Q4_0:
		for offset := 0; offset < len(source); offset += 18 {
			if err := ctx.Err(); err != nil {
				return err
			}
			scale := float16ToFloat32(binary.LittleEndian.Uint16(source[offset:]))
			for i := 0; i < 16; i++ {
				packed := source[offset+2+i]
				if err := appendFloat(scale * float32(int(packed&0x0f)-8)); err != nil {
					return err
				}
				if err := appendFloat(scale * float32(int(packed>>4)-8)); err != nil {
					return err
				}
			}
		}
	case Q6_K:
		for offset := 0; offset < len(source); offset += 210 {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := appendQ6KBlock(source[offset:offset+210], appendFloat); err != nil {
				return err
			}
		}
	case Q4_K, Q5_K:
		blockSize := 144
		if tensor.Type == Q5_K {
			blockSize = 176
		}
		for offset := 0; offset < len(source); offset += blockSize {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := appendQ45KBlock(tensor.Type, source[offset:offset+blockSize], appendFloat); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("gguf: tensor %q has unsupported type %s", tensor.Name, tensor.Type)
	}
	return flush()
}

var mxfp4Values = [16]float32{
	0, 0.5, 1, 1.5, 2, 3, 4, 6,
	float32(math.Copysign(0, -1)), -0.5, -1, -1.5, -2, -3, -4, -6,
}

func appendMXFP4Block(block []byte, appendValue func(float32) error) error {
	// E8M0 is an unsigned exponent with a 127 bias. 0xff denotes NaN.
	scale := float32(math.Ldexp(1, int(block[0])-127))
	if block[0] == 0xff {
		scale = float32(math.NaN())
	}
	for _, packed := range block[1:17] {
		if err := appendValue(scale * mxfp4Values[packed&0x0f]); err != nil {
			return err
		}
		if err := appendValue(scale * mxfp4Values[packed>>4]); err != nil {
			return err
		}
	}
	return nil
}

func appendQ6KBlock(block []byte, appendValue func(float32) error) error {
	d := float16ToFloat32(binary.LittleEndian.Uint16(block[208:]))
	var values [256]float32
	for part := 0; part < 4; part++ {
		ql := block[part*16:]
		qh := block[128+part*16:]
		scales := block[192+(part/2)*8+(part%2)*2:]
		for i := 0; i < 16; i++ {
			q1 := int(ql[i]&0x0f) | int(qh[i]&0x03)<<4
			q2 := int(ql[i+32]&0x0f) | int((qh[i]>>2)&0x03)<<4
			q3 := int(ql[i]>>4) | int((qh[i]>>4)&0x03)<<4
			q4 := int(ql[i+32]>>4) | int((qh[i]>>6)&0x03)<<4
			base := (part / 2) * 128
			local := (part % 2) * 16
			values[base+local+i] = d * float32(int8(scales[0])) * float32(q1-32)
			values[base+32+local+i] = d * float32(int8(scales[2])) * float32(q2-32)
			values[base+64+local+i] = d * float32(int8(scales[4])) * float32(q3-32)
			values[base+96+local+i] = d * float32(int8(scales[6])) * float32(q4-32)
		}
	}
	for _, value := range values {
		if err := appendValue(value); err != nil {
			return err
		}
	}
	return nil
}

func appendQ45KBlock(tensorType TensorType, block []byte, appendValue func(float32) error) error {
	d := float16ToFloat32(binary.LittleEndian.Uint16(block[0:]))
	dmin := float16ToFloat32(binary.LittleEndian.Uint16(block[2:]))
	qsOffset := 16
	if tensorType == Q5_K {
		qsOffset = 48
	}
	var values [256]float32
	for chunk := 0; chunk < 4; chunk++ {
		scaleIndex := chunk * 2
		scale1, min1 := kScaleMin(block[4:16], scaleIndex)
		scale2, min2 := kScaleMin(block[4:16], scaleIndex+1)
		qs := block[qsOffset+chunk*32:]
		for i := 0; i < 32; i++ {
			q1 := float32(qs[i] & 0x0f)
			q2 := float32(qs[i] >> 4)
			if tensorType == Q5_K {
				qh := block[16+i]
				q1 += float32((qh>>scaleIndex)&1) * 16
				q2 += float32((qh>>(scaleIndex+1))&1) * 16
			}
			base := chunk * 64
			values[base+i] = d*float32(scale1)*q1 - dmin*float32(min1)
			values[base+32+i] = d*float32(scale2)*q2 - dmin*float32(min2)
		}
	}
	for _, value := range values {
		if err := appendValue(value); err != nil {
			return err
		}
	}
	return nil
}

func kScaleMin(scales []byte, index int) (uint8, uint8) {
	if index < 4 {
		return scales[index] & 0x3f, scales[index+4] & 0x3f
	}
	scale := scales[index+4]&0x0f | (scales[index-4]>>6)<<4
	min := scales[index+4]>>4 | (scales[index]>>6)<<4
	return scale, min
}

func writeJSONAtomic(path string, value any) (returnErr error) {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("gguf: encode %s: %w", filepath.Base(path), err)
	}
	data = append(data, '\n')
	file, err := os.CreateTemp(filepath.Dir(path), ".gguf-json-*")
	if err != nil {
		return fmt.Errorf("gguf: create %s: %w", filepath.Base(path), err)
	}
	temporary := file.Name()
	defer func() {
		_ = file.Close()
		if returnErr != nil {
			_ = os.Remove(temporary)
		}
	}()
	if _, err := file.Write(data); err != nil {
		return fmt.Errorf("gguf: write %s: %w", filepath.Base(path), err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("gguf: close %s: %w", filepath.Base(path), err)
	}
	if err := os.Rename(temporary, path); err != nil {
		return fmt.Errorf("gguf: install %s: %w", filepath.Base(path), err)
	}
	return nil
}

// float32ToFloat16 implements IEEE-754 round-to-nearest-even conversion.
func float32ToFloat16(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16(bits >> 16 & 0x8000)
	exponent := int((bits >> 23) & 0xff)
	mantissa := bits & 0x7fffff
	if exponent == 0xff {
		if mantissa != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	}
	halfExponent := exponent - 127 + 15
	if halfExponent >= 31 {
		return sign | 0x7c00
	}
	if halfExponent <= 0 {
		if halfExponent < -10 {
			return sign
		}
		mantissa |= 0x800000
		shift := uint32(14 - halfExponent)
		half := mantissa >> shift
		remainder := mantissa & ((uint32(1) << shift) - 1)
		halfway := uint32(1) << (shift - 1)
		if remainder > halfway || remainder == halfway && half&1 != 0 {
			half++
		}
		return sign | uint16(half)
	}
	half := uint32(halfExponent)<<10 | mantissa>>13
	remainder := mantissa & 0x1fff
	if remainder > 0x1000 || remainder == 0x1000 && half&1 != 0 {
		half++
	}
	return sign | uint16(half)
}

func float16ToFloat32(value uint16) float32 {
	sign := uint32(value&0x8000) << 16
	exponent := uint32(value>>10) & 0x1f
	mantissa := uint32(value & 0x03ff)
	var bits uint32
	switch exponent {
	case 0:
		if mantissa == 0 {
			bits = sign
		} else {
			exponent = 1
			for mantissa&0x0400 == 0 {
				mantissa <<= 1
				exponent--
			}
			mantissa &= 0x03ff
			bits = sign | (exponent+127-15)<<23 | mantissa<<13
		}
	case 0x1f:
		bits = sign | 0x7f800000 | mantissa<<13
	default:
		bits = sign | (exponent+127-15)<<23 | mantissa<<13
	}
	return math.Float32frombits(bits)
}

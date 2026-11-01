// Package gguf reads GGUF models and converts Llama weights to safetensors.
package gguf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
)

// TensorType is a GGUF tensor encoding.
type TensorType uint32

const (
	F32   TensorType = 0
	F16   TensorType = 1
	Q4_0  TensorType = 2
	Q8_0  TensorType = 8
	Q4_K  TensorType = 12
	Q5_K  TensorType = 13
	Q6_K  TensorType = 14
	BF16  TensorType = 30
	MXFP4 TensorType = 39
)

func (t TensorType) String() string {
	switch t {
	case F32:
		return "F32"
	case F16:
		return "F16"
	case Q4_0:
		return "Q4_0"
	case Q8_0:
		return "Q8_0"
	case Q4_K:
		return "Q4_K"
	case Q5_K:
		return "Q5_K"
	case Q6_K:
		return "Q6_K"
	case BF16:
		return "BF16"
	case MXFP4:
		return "MXFP4"
	default:
		return fmt.Sprintf("GGUF_TYPE_%d", uint32(t))
	}
}

// TensorInfo describes one tensor. Offset is absolute in the GGUF file, while
// RelativeOffset is the offset stored in its tensor-info record.
type TensorInfo struct {
	Name           string
	Type           TensorType
	Shape          []uint64
	Offset         int64
	RelativeOffset uint64
	ByteSize       int64
}

// Reader is an open GGUF file. Tensor data is loaded only by ReadTensor.
type Reader struct {
	file       *os.File
	metadata   map[string]any
	tensors    map[string]TensorInfo
	dataOffset int64
}

// Open reads a little-endian GGUF v2 or v3 header and tensor index.
func Open(path string) (*Reader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: open %q: %w", path, err)
	}
	r, err := read(f)
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("gguf: open %q: %w", path, err)
	}
	return r, nil
}

// Close closes the underlying file. Metadata remains available afterwards.
func (r *Reader) Close() error {
	if r == nil || r.file == nil {
		return nil
	}
	return r.file.Close()
}

// Metadata returns a shallow copy of the GGUF key/value metadata. Array values
// returned in the map should be treated as read-only.
func (r *Reader) Metadata() map[string]any {
	result := make(map[string]any, len(r.metadata))
	for key, value := range r.metadata {
		result[key] = value
	}
	return result
}

// Architecture reads general.architecture without loading tensor data.
func Architecture(path string) (_ string, returnErr error) {
	reader, err := Open(path)
	if err != nil {
		return "", err
	}
	defer func() {
		if closeErr := reader.Close(); returnErr == nil && closeErr != nil {
			returnErr = fmt.Errorf("gguf: close input: %w", closeErr)
		}
	}()
	return metadataString(reader.metadata, "general.architecture")
}

// Tensors returns the tensor index sorted by name. Shapes are copies.
func (r *Reader) Tensors() []TensorInfo {
	result := make([]TensorInfo, 0, len(r.tensors))
	for _, info := range r.tensors {
		result = append(result, cloneTensorInfo(info))
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	return result
}

// Tensor returns one tensor's index entry.
func (r *Reader) Tensor(name string) (TensorInfo, bool) {
	info, ok := r.tensors[name]
	return cloneTensorInfo(info), ok
}

// ReadTensor reads exactly one tensor in its native GGUF encoding.
func (r *Reader) ReadTensor(name string) ([]byte, error) {
	info, ok := r.tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	if info.ByteSize > int64(maxInt()) {
		return nil, fmt.Errorf("gguf: tensor %q is too large for this platform", name)
	}
	data := make([]byte, int(info.ByteSize))
	if len(data) == 0 {
		return data, nil
	}
	n, err := r.file.ReadAt(data, info.Offset)
	if err != nil {
		return nil, fmt.Errorf("gguf: read tensor %q: %w", name, err)
	}
	if n != len(data) {
		return nil, fmt.Errorf("gguf: read tensor %q: %w", name, io.ErrUnexpectedEOF)
	}
	return data, nil
}

// LlamaMetadata contains the dense Llama-like configuration stored in GGUF metadata.
type LlamaMetadata struct {
	Architecture            string
	HiddenSize              int
	NumHeads                int
	NumKVHeads              int
	NumLayers               int
	IntermediateSize        int
	VocabSize               int
	RMSNormEpsilon          float64
	RoPETheta               float64
	ContextLength           int
	HeadDim                 int
	SlidingWindow           int
	SlidingWindowPattern    int
	QueryPreAttentionScalar float64
	AttnLogitSoftcapping    float64
	FinalLogitSoftcapping   float64
	EOSTokenID              int64
	BOSTokenID              int64
}

// LlamaMetadata extracts and validates dense Llama-family architecture fields.
func (r *Reader) LlamaMetadata() (LlamaMetadata, error) {
	var result LlamaMetadata
	var err error
	if result.Architecture, err = metadataString(r.metadata, "general.architecture"); err != nil {
		return LlamaMetadata{}, err
	}
	switch result.Architecture {
	case "llama", "qwen2", "gemma", "gemma2":
	default:
		return LlamaMetadata{}, fmt.Errorf("gguf: unsupported architecture %q for dense Llama-family conversion", result.Architecture)
	}
	prefix := result.Architecture
	intFields := []struct {
		key         string
		destination *int
	}{
		{prefix + ".embedding_length", &result.HiddenSize},
		{prefix + ".attention.head_count", &result.NumHeads},
		{prefix + ".attention.head_count_kv", &result.NumKVHeads},
		{prefix + ".block_count", &result.NumLayers},
		{prefix + ".feed_forward_length", &result.IntermediateSize},
		{prefix + ".context_length", &result.ContextLength},
	}
	for _, field := range intFields {
		value, err := metadataInt(r.metadata, field.key)
		if err != nil {
			return LlamaMetadata{}, err
		}
		if value <= 0 || value > int64(maxInt()) {
			return LlamaMetadata{}, fmt.Errorf("gguf: metadata %q must be a positive platform-sized integer, got %d", field.key, value)
		}
		*field.destination = int(value)
	}
	if result.RMSNormEpsilon, err = metadataFloat(r.metadata, prefix+".attention.layer_norm_rms_epsilon"); err != nil {
		return LlamaMetadata{}, err
	}
	if result.RMSNormEpsilon <= 0 || math.IsNaN(result.RMSNormEpsilon) || math.IsInf(result.RMSNormEpsilon, 0) {
		return LlamaMetadata{}, fmt.Errorf("gguf: invalid RMS norm epsilon %v", result.RMSNormEpsilon)
	}
	if value, ok := r.metadata[prefix+".rope.freq_base"]; ok {
		if result.RoPETheta, err = numberFloat(value); err != nil || result.RoPETheta <= 0 || math.IsNaN(result.RoPETheta) || math.IsInf(result.RoPETheta, 0) {
			return LlamaMetadata{}, fmt.Errorf("gguf: invalid %s.rope.freq_base %v", prefix, value)
		}
	} else {
		result.RoPETheta = 10000
	}
	result.HeadDim = result.HiddenSize / result.NumHeads
	for _, key := range []string{prefix + ".attention.key_length", prefix + ".attention.value_length"} {
		if raw, ok := r.metadata[key]; ok {
			value, parseErr := numberInt(raw)
			if parseErr != nil || value <= 0 || value > int64(maxInt()) {
				return LlamaMetadata{}, fmt.Errorf("gguf: invalid %s %v", key, raw)
			}
			result.HeadDim = int(value)
		}
	}
	for _, field := range []struct {
		key         string
		destination *int
	}{
		{prefix + ".attention.sliding_window", &result.SlidingWindow},
		{prefix + ".attention.sliding_window_pattern", &result.SlidingWindowPattern},
	} {
		if raw, ok := r.metadata[field.key]; ok {
			value, parseErr := numberInt(raw)
			if parseErr != nil || value < 0 || value > int64(maxInt()) {
				return LlamaMetadata{}, fmt.Errorf("gguf: invalid %s %v", field.key, raw)
			}
			*field.destination = int(value)
		}
	}
	for _, field := range []struct {
		key         string
		destination *float64
	}{
		{prefix + ".attention.query_pre_attn_scalar", &result.QueryPreAttentionScalar},
		{prefix + ".attention.logit_softcapping", &result.AttnLogitSoftcapping},
		{prefix + ".final_logit_softcapping", &result.FinalLogitSoftcapping},
	} {
		if raw, ok := r.metadata[field.key]; ok {
			value, parseErr := numberFloat(raw)
			if parseErr != nil || value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
				return LlamaMetadata{}, fmt.Errorf("gguf: invalid %s %v", field.key, raw)
			}
			*field.destination = value
		}
	}
	if result.EOSTokenID, err = metadataInt(r.metadata, "tokenizer.ggml.eos_token_id"); err != nil {
		return LlamaMetadata{}, err
	}
	if raw, ok := r.metadata["tokenizer.ggml.bos_token_id"]; ok {
		if result.BOSTokenID, err = numberInt(raw); err != nil {
			return LlamaMetadata{}, fmt.Errorf("gguf: invalid tokenizer.ggml.bos_token_id %v", raw)
		}
	} else if result.Architecture == "qwen2" {
		result.BOSTokenID = result.EOSTokenID
	} else {
		return LlamaMetadata{}, errors.New("gguf: missing metadata \"tokenizer.ggml.bos_token_id\"")
	}
	tokens, ok := r.metadata["tokenizer.ggml.tokens"].([]string)
	if !ok || len(tokens) == 0 {
		return LlamaMetadata{}, errors.New("gguf: tokenizer.ggml.tokens must be a non-empty string array")
	}
	result.VocabSize = len(tokens)
	gemma := result.Architecture == "gemma" || result.Architecture == "gemma2"
	if !gemma && result.HiddenSize%result.NumHeads != 0 || result.HeadDim%2 != 0 || result.NumKVHeads > result.NumHeads || result.NumHeads%result.NumKVHeads != 0 {
		return LlamaMetadata{}, fmt.Errorf("gguf: inconsistent %s attention head dimensions", result.Architecture)
	}
	if result.EOSTokenID < 0 || result.EOSTokenID >= int64(result.VocabSize) || result.BOSTokenID < 0 || result.BOSTokenID >= int64(result.VocabSize) {
		return LlamaMetadata{}, errors.New("gguf: BOS or EOS token ID is outside the vocabulary")
	}
	return result, nil
}

func cloneTensorInfo(info TensorInfo) TensorInfo {
	info.Shape = append([]uint64(nil), info.Shape...)
	return info
}

func read(f *os.File) (*Reader, error) {
	stat, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat: %w", err)
	}
	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic[:]) != "GGUF" {
		return nil, fmt.Errorf("invalid magic %q", magic)
	}
	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if version != 2 && version != 3 {
		return nil, fmt.Errorf("unsupported version %d: want 2 or 3", version)
	}
	var tensorCount, metadataCount uint64
	if err := binary.Read(f, binary.LittleEndian, &tensorCount); err != nil {
		return nil, fmt.Errorf("read tensor count: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &metadataCount); err != nil {
		return nil, fmt.Errorf("read metadata count: %w", err)
	}
	if tensorCount > uint64(maxInt()) || metadataCount > uint64(maxInt()) {
		return nil, errors.New("header count exceeds platform capacity")
	}

	metadata := make(map[string]any, int(metadataCount))
	for i := uint64(0); i < metadataCount; i++ {
		key, err := readString(f)
		if err != nil {
			return nil, fmt.Errorf("read metadata key %d: %w", i, err)
		}
		if _, exists := metadata[key]; exists {
			return nil, fmt.Errorf("duplicate metadata key %q", key)
		}
		value, err := readMetadataValue(f)
		if err != nil {
			return nil, fmt.Errorf("read metadata %q: %w", key, err)
		}
		metadata[key] = value
	}

	r := &Reader{file: f, metadata: metadata, tensors: make(map[string]TensorInfo, int(tensorCount))}
	for i := uint64(0); i < tensorCount; i++ {
		name, err := readString(f)
		if err != nil {
			return nil, fmt.Errorf("read tensor %d name: %w", i, err)
		}
		if _, exists := r.tensors[name]; exists {
			return nil, fmt.Errorf("duplicate tensor %q", name)
		}
		var dimensions uint32
		if err := binary.Read(f, binary.LittleEndian, &dimensions); err != nil {
			return nil, fmt.Errorf("tensor %q dimensions: %w", name, err)
		}
		if dimensions > 64 {
			return nil, fmt.Errorf("tensor %q has unreasonable dimension count %d", name, dimensions)
		}
		shape := make([]uint64, int(dimensions))
		for j := range shape {
			if err := binary.Read(f, binary.LittleEndian, &shape[j]); err != nil {
				return nil, fmt.Errorf("tensor %q shape: %w", name, err)
			}
		}
		var tensorType uint32
		var offset uint64
		if err := binary.Read(f, binary.LittleEndian, &tensorType); err != nil {
			return nil, fmt.Errorf("tensor %q type: %w", name, err)
		}
		if err := binary.Read(f, binary.LittleEndian, &offset); err != nil {
			return nil, fmt.Errorf("tensor %q offset: %w", name, err)
		}
		size, err := encodedSize(TensorType(tensorType), shape)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", name, err)
		}
		r.tensors[name] = TensorInfo{Name: name, Type: TensorType(tensorType), Shape: shape, RelativeOffset: offset, ByteSize: size}
	}

	alignment := uint64(32)
	if raw, ok := metadata["general.alignment"]; ok {
		alignment, err = numberUint(raw)
		if err != nil || alignment == 0 || alignment > math.MaxInt64 {
			return nil, fmt.Errorf("invalid general.alignment %v", raw)
		}
	}
	headerEnd, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, fmt.Errorf("find tensor data: %w", err)
	}
	dataOffset := align(headerEnd, int64(alignment))
	if dataOffset < headerEnd || dataOffset > stat.Size() {
		return nil, errors.New("tensor data offset exceeds file size")
	}
	r.dataOffset = dataOffset
	for name, info := range r.tensors {
		if info.RelativeOffset > math.MaxInt64 || int64(info.RelativeOffset) > math.MaxInt64-dataOffset || info.ByteSize > math.MaxInt64-dataOffset-int64(info.RelativeOffset) {
			return nil, fmt.Errorf("tensor %q data range overflows int64", name)
		}
		info.Offset = dataOffset + int64(info.RelativeOffset)
		if info.Offset+info.ByteSize > stat.Size() {
			return nil, fmt.Errorf("tensor %q data range exceeds file size", name)
		}
		r.tensors[name] = info
	}
	return r, nil
}

func align(value, alignment int64) int64 {
	if alignment <= 0 || value > math.MaxInt64-(alignment-1) {
		return -1
	}
	return ((value + alignment - 1) / alignment) * alignment
}

func encodedSize(tensorType TensorType, shape []uint64) (int64, error) {
	elements := uint64(1)
	for _, dimension := range shape {
		if dimension != 0 && elements > math.MaxUint64/dimension {
			return 0, errors.New("shape element count overflows uint64")
		}
		elements *= dimension
	}
	if elements > math.MaxInt64 {
		return 0, errors.New("tensor is too large")
	}
	var size uint64
	switch tensorType {
	case F32:
		if elements > math.MaxInt64/4 {
			return 0, errors.New("encoded tensor size overflows int64")
		}
		size = elements * 4
	case F16, BF16:
		if elements > math.MaxInt64/2 {
			return 0, errors.New("encoded tensor size overflows int64")
		}
		size = elements * 2
	case MXFP4:
		if elements%32 != 0 {
			return 0, errors.New("MXFP4 element count must be divisible by 32")
		}
		if elements/32 > math.MaxInt64/17 {
			return 0, errors.New("encoded tensor size overflows int64")
		}
		size = elements / 32 * 17
	case Q8_0:
		if elements%32 != 0 {
			return 0, errors.New("Q8_0 element count must be divisible by 32")
		}
		if elements/32 > math.MaxInt64/34 {
			return 0, errors.New("encoded tensor size overflows int64")
		}
		size = elements / 32 * 34
	case Q4_0:
		if elements%32 != 0 {
			return 0, errors.New("Q4_0 element count must be divisible by 32")
		}
		if elements/32 > math.MaxInt64/18 {
			return 0, errors.New("encoded tensor size overflows int64")
		}
		size = elements / 32 * 18
	case Q4_K, Q5_K, Q6_K:
		if elements%256 != 0 {
			return 0, fmt.Errorf("%s element count must be divisible by 256", tensorType)
		}
		blockSize := uint64(144)
		if tensorType == Q5_K {
			blockSize = 176
		} else if tensorType == Q6_K {
			blockSize = 210
		}
		if elements/256 > math.MaxInt64/blockSize {
			return 0, errors.New("encoded tensor size overflows int64")
		}
		size = elements / 256 * blockSize
	default:
		return 0, fmt.Errorf("unsupported tensor type %s", tensorType)
	}
	return int64(size), nil
}

const (
	valueUint8   = 0
	valueInt8    = 1
	valueUint16  = 2
	valueInt16   = 3
	valueUint32  = 4
	valueInt32   = 5
	valueFloat32 = 6
	valueBool    = 7
	valueString  = 8
	valueArray   = 9
	valueUint64  = 10
	valueInt64   = 11
	valueFloat64 = 12
)

func readMetadataValue(r io.Reader) (any, error) {
	var valueType uint32
	if err := binary.Read(r, binary.LittleEndian, &valueType); err != nil {
		return nil, err
	}
	return readTypedValue(r, valueType)
}

func readTypedValue(r io.Reader, valueType uint32) (any, error) {
	var value any
	switch valueType {
	case valueUint8:
		value = new(uint8)
	case valueInt8:
		value = new(int8)
	case valueUint16:
		value = new(uint16)
	case valueInt16:
		value = new(int16)
	case valueUint32:
		value = new(uint32)
	case valueInt32:
		value = new(int32)
	case valueFloat32:
		value = new(float32)
	case valueUint64:
		value = new(uint64)
	case valueInt64:
		value = new(int64)
	case valueFloat64:
		value = new(float64)
	case valueBool:
		var raw uint8
		if err := binary.Read(r, binary.LittleEndian, &raw); err != nil {
			return nil, err
		}
		if raw > 1 {
			return nil, fmt.Errorf("invalid boolean value %d", raw)
		}
		return raw == 1, nil
	case valueString:
		return readString(r)
	case valueArray:
		return readArray(r)
	default:
		return nil, fmt.Errorf("unknown metadata value type %d", valueType)
	}
	if err := binary.Read(r, binary.LittleEndian, value); err != nil {
		return nil, err
	}
	switch value := value.(type) {
	case *uint8:
		return *value, nil
	case *int8:
		return *value, nil
	case *uint16:
		return *value, nil
	case *int16:
		return *value, nil
	case *uint32:
		return *value, nil
	case *int32:
		return *value, nil
	case *float32:
		return *value, nil
	case *uint64:
		return *value, nil
	case *int64:
		return *value, nil
	case *float64:
		return *value, nil
	default:
		panic("unreachable GGUF scalar type")
	}
}

func readArray(r io.Reader) (any, error) {
	var elementType uint32
	var count uint64
	if err := binary.Read(r, binary.LittleEndian, &elementType); err != nil {
		return nil, err
	}
	if elementType == valueArray {
		return nil, errors.New("nested arrays are not valid GGUF metadata")
	}
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, err
	}
	if count > uint64(maxInt()) {
		return nil, errors.New("array exceeds platform capacity")
	}
	if elementType == valueString {
		values := make([]string, int(count))
		for i := range values {
			value, err := readString(r)
			if err != nil {
				return nil, fmt.Errorf("array element %d: %w", i, err)
			}
			values[i] = value
		}
		return values, nil
	}
	values := make([]any, int(count))
	for i := range values {
		value, err := readTypedValue(r, elementType)
		if err != nil {
			return nil, fmt.Errorf("array element %d: %w", i, err)
		}
		values[i] = value
	}
	return values, nil
}

func readString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > uint64(maxInt()) {
		return "", errors.New("string exceeds platform capacity")
	}
	data := make([]byte, int(length))
	if _, err := io.ReadFull(r, data); err != nil {
		return "", err
	}
	return string(data), nil
}

func metadataString(metadata map[string]any, key string) (string, error) {
	value, ok := metadata[key].(string)
	if !ok {
		return "", fmt.Errorf("gguf: metadata %q must be a string", key)
	}
	return value, nil
}

func metadataInt(metadata map[string]any, key string) (int64, error) {
	value, ok := metadata[key]
	if !ok {
		return 0, fmt.Errorf("gguf: missing metadata %q", key)
	}
	return numberInt(value)
}

func metadataFloat(metadata map[string]any, key string) (float64, error) {
	value, ok := metadata[key]
	if !ok {
		return 0, fmt.Errorf("gguf: missing metadata %q", key)
	}
	result, err := numberFloat(value)
	if err != nil {
		return 0, fmt.Errorf("gguf: metadata %q: %w", key, err)
	}
	return result, nil
}

func numberInt(value any) (int64, error) {
	switch value := value.(type) {
	case uint8:
		return int64(value), nil
	case int8:
		return int64(value), nil
	case uint16:
		return int64(value), nil
	case int16:
		return int64(value), nil
	case uint32:
		return int64(value), nil
	case int32:
		return int64(value), nil
	case uint64:
		if value > math.MaxInt64 {
			return 0, fmt.Errorf("integer %d overflows int64", value)
		}
		return int64(value), nil
	case int64:
		return value, nil
	default:
		return 0, fmt.Errorf("%v (%T) is not an integer", value, value)
	}
}

func numberUint(value any) (uint64, error) {
	switch value := value.(type) {
	case uint8:
		return uint64(value), nil
	case uint16:
		return uint64(value), nil
	case uint32:
		return uint64(value), nil
	case uint64:
		return value, nil
	default:
		integer, err := numberInt(value)
		if err != nil || integer < 0 {
			return 0, fmt.Errorf("%v is not an unsigned integer", value)
		}
		return uint64(integer), nil
	}
}

func numberFloat(value any) (float64, error) {
	switch value := value.(type) {
	case float32:
		return float64(value), nil
	case float64:
		return value, nil
	default:
		return 0, fmt.Errorf("%v (%T) is not a float", value, value)
	}
}

func maxInt() int {
	return int(^uint(0) >> 1)
}

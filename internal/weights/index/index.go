// Package index provides indexed, per-tensor access to a model's safetensors weights.
package index

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/lutikk/airllm/internal/weights/safetensors"
)

const (
	indexFileName  = "model.safetensors.index.json"
	singleFileName = "model.safetensors"
)

// TensorRef describes a tensor and the shard that contains it. Size is the
// tensor's encoded size in bytes.
type TensorRef struct {
	Name  string
	Shard string
	DType safetensors.DType
	Shape []int64
	Size  int64
}

// Model is an open set of model shards. LayerNames returns the available
// groups in model order; Layer selects a group by its position in that list.
type Model struct {
	files      map[string]*safetensors.File
	tensors    map[string]TensorRef
	layerNames []string
	layers     [][]TensorRef
	closed     bool
}

type weightIndex struct {
	WeightMap map[string]string `json:"weight_map"`
}

// OpenModel opens a sharded or single-file safetensors model in dir. Only
// shard headers are read; tensor data remains on disk until ReadTensor.
func OpenModel(dir string) (*Model, error) {
	indexPath := filepath.Join(dir, indexFileName)
	if _, err := os.Stat(indexPath); err == nil {
		return openSharded(dir, indexPath)
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("weights: stat model index %q: %w", indexPath, err)
	}

	singlePath := filepath.Join(dir, singleFileName)
	if _, err := os.Stat(singlePath); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("weights: no model found in %q: expected %s or %s", dir, indexFileName, singleFileName)
		}
		return nil, fmt.Errorf("weights: stat model file %q: %w", singlePath, err)
	}
	return openSingle(dir, singleFileName)
}

func openSharded(dir, indexPath string) (_ *Model, err error) {
	contents, err := os.ReadFile(indexPath)
	if err != nil {
		return nil, fmt.Errorf("weights: read model index %q: %w", indexPath, err)
	}
	var encoded weightIndex
	if err := json.Unmarshal(contents, &encoded); err != nil {
		return nil, fmt.Errorf("weights: decode model index %q: %w", indexPath, err)
	}
	if len(encoded.WeightMap) == 0 {
		return nil, fmt.Errorf("weights: model index %q has an empty or missing weight_map", indexPath)
	}

	shards := make(map[string]struct{})
	for name, shard := range encoded.WeightMap {
		if name == "" {
			return nil, fmt.Errorf("weights: model index %q contains an empty tensor name", indexPath)
		}
		if shard == "" {
			return nil, fmt.Errorf("weights: tensor %q has an empty shard name", name)
		}
		shards[shard] = struct{}{}
	}

	model := newModel()
	defer func() {
		if err != nil {
			_ = model.Close()
		}
	}()
	for shard := range shards {
		path := filepath.Join(dir, shard)
		file, openErr := safetensors.Open(path)
		if openErr != nil {
			return nil, fmt.Errorf("weights: open shard %q: %w", path, openErr)
		}
		model.files[shard] = file
	}

	for name, shard := range encoded.WeightMap {
		tensor, ok := model.files[shard].TensorByName(name)
		if !ok {
			return nil, fmt.Errorf("weights: tensor %q listed in model index is missing from shard %q", name, shard)
		}
		model.tensors[name] = makeRef(tensor, shard)
	}
	model.buildLayers()
	return model, nil
}

func openSingle(dir, shard string) (*Model, error) {
	file, err := safetensors.Open(filepath.Join(dir, shard))
	if err != nil {
		return nil, fmt.Errorf("weights: open model: %w", err)
	}
	model := newModel()
	model.files[shard] = file
	for _, tensor := range file.Tensors() {
		model.tensors[tensor.Name] = makeRef(tensor, shard)
	}
	model.buildLayers()
	return model, nil
}

func newModel() *Model {
	return &Model{
		files:   make(map[string]*safetensors.File),
		tensors: make(map[string]TensorRef),
	}
}

func makeRef(tensor safetensors.Tensor, shard string) TensorRef {
	return TensorRef{
		Name:  tensor.Name,
		Shard: shard,
		DType: tensor.DType,
		Shape: append([]int64(nil), tensor.Shape...),
		Size:  tensor.ByteSize,
	}
}

func (m *Model) buildLayers() {
	groups := make(map[string][]TensorRef)
	layerNumbers := make(map[int]struct{})
	for name, ref := range m.tensors {
		switch {
		case strings.HasPrefix(name, "model.embed_tokens."):
			groups["embedding"] = append(groups["embedding"], ref)
		case strings.HasPrefix(name, "model.layers."):
			number, ok := layerNumber(name)
			if ok {
				group := "model.layers." + strconv.Itoa(number)
				groups[group] = append(groups[group], ref)
				layerNumbers[number] = struct{}{}
			}
		case strings.HasPrefix(name, "model.norm."):
			groups["model.norm"] = append(groups["model.norm"], ref)
		case strings.HasPrefix(name, "lm_head."):
			groups["lm_head"] = append(groups["lm_head"], ref)
		}
	}

	if _, ok := groups["embedding"]; ok {
		m.layerNames = append(m.layerNames, "embedding")
	}
	numbers := make([]int, 0, len(layerNumbers))
	for number := range layerNumbers {
		numbers = append(numbers, number)
	}
	sort.Ints(numbers)
	for _, number := range numbers {
		m.layerNames = append(m.layerNames, "model.layers."+strconv.Itoa(number))
	}
	for _, name := range []string{"model.norm", "lm_head"} {
		if _, ok := groups[name]; ok {
			m.layerNames = append(m.layerNames, name)
		}
	}

	m.layers = make([][]TensorRef, len(m.layerNames))
	for i, name := range m.layerNames {
		refs := groups[name]
		sort.Slice(refs, func(i, j int) bool { return refs[i].Name < refs[j].Name })
		m.layers[i] = refs
	}
}

func layerNumber(name string) (int, bool) {
	remainder := strings.TrimPrefix(name, "model.layers.")
	dot := strings.IndexByte(remainder, '.')
	if dot <= 0 {
		return 0, false
	}
	number, err := strconv.Atoi(remainder[:dot])
	if err != nil || number < 0 || strconv.Itoa(number) != remainder[:dot] {
		return 0, false
	}
	return number, true
}

// LayerNames returns available layer groups in model order. A tied lm_head
// normally has no separate tensor and therefore no lm_head group.
func (m *Model) LayerNames() []string {
	return append([]string(nil), m.layerNames...)
}

// Layer returns the tensors in the layer at position i in LayerNames.
func (m *Model) Layer(i int) ([]TensorRef, error) {
	if i < 0 || i >= len(m.layers) {
		return nil, fmt.Errorf("weights: layer index %d out of range [0, %d)", i, len(m.layers))
	}
	return cloneRefs(m.layers[i]), nil
}

// Tensor returns metadata for name. The returned shape is a copy.
func (m *Model) Tensor(name string) (TensorRef, bool) {
	ref, ok := m.tensors[name]
	if !ok {
		return TensorRef{}, false
	}
	return cloneRef(ref), true
}

// ReadTensor reads only the requested tensor's bytes from its shard.
func (m *Model) ReadTensor(name string) ([]byte, error) {
	if m.closed {
		return nil, errors.New("weights: read tensor: model is closed")
	}
	ref, ok := m.tensors[name]
	if !ok {
		return nil, fmt.Errorf("weights: tensor %q not found", name)
	}
	data, err := m.files[ref.Shard].ReadTensor(name)
	if err != nil {
		return nil, fmt.Errorf("weights: read tensor %q from shard %q: %w", name, ref.Shard, err)
	}
	return data, nil
}

// Close closes every open shard. It is safe to call Close more than once.
func (m *Model) Close() error {
	if m == nil || m.closed {
		return nil
	}
	m.closed = true

	shards := make([]string, 0, len(m.files))
	for shard := range m.files {
		shards = append(shards, shard)
	}
	sort.Strings(shards)

	var errs []error
	for _, shard := range shards {
		if err := m.files[shard].Close(); err != nil {
			errs = append(errs, fmt.Errorf("close shard %q: %w", shard, err))
		}
	}
	return errors.Join(errs...)
}

func cloneRefs(refs []TensorRef) []TensorRef {
	result := make([]TensorRef, len(refs))
	for i, ref := range refs {
		result[i] = cloneRef(ref)
	}
	return result
}

func cloneRef(ref TensorRef) TensorRef {
	ref.Shape = append([]int64(nil), ref.Shape...)
	return ref
}

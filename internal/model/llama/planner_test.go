package llama

import (
	"math"
	"testing"

	"github.com/lutikk/airllm/pkg/memory"
)

func TestLayerPlannerTinyLlama(t *testing.T) {
	planner := newTestPlanner(t, tinyLlamaConfig, 2)
	if got, want := planner.MaxLayerBytes(), uint64(154_148_864); got != want {
		t.Fatalf("MaxLayerBytes() = %d, want %d", got, want)
	}
	if got, want := planner.KVCacheBytes(2048), uint64(46_137_344); got != want {
		t.Fatalf("KVCacheBytes() = %d, want %d", got, want)
	}
	estimate := planner.Estimate(2048)
	if got, want := estimate.ResidentWeights, uint64(262_148_096); got != want {
		t.Fatalf("ResidentWeights = %d, want %d", got, want)
	}
	if estimate.MaxLayer != planner.MaxLayerBytes() || estimate.KVCache != planner.KVCacheBytes(2048) {
		t.Fatalf("Estimate() is inconsistent: %+v", estimate)
	}
}

func TestLayerPlannerMistralTiedEmbeddings(t *testing.T) {
	input := `{
	  "architectures": ["MistralForCausalLM"],
	  "hidden_size": 4096,
	  "intermediate_size": 14336,
	  "num_attention_heads": 32,
	  "num_key_value_heads": 8,
	  "num_hidden_layers": 32,
	  "vocab_size": 32000,
	  "rms_norm_eps": 0.00001,
	  "rope_theta": 10000,
	  "head_dim": 128,
	  "max_position_embeddings": 32768,
	  "sliding_window": 4096,
	  "tie_word_embeddings": true,
	  "eos_token_id": 2
	}`
	planner := newTestPlanner(t, input, 2)
	if got, want := planner.MaxLayerBytes(), uint64(436_224_000); got != want {
		t.Fatalf("MaxLayerBytes() = %d, want %d", got, want)
	}
	estimate := planner.Estimate(4096)
	if got, want := estimate.ResidentWeights, uint64(262_152_192); got != want {
		t.Fatalf("ResidentWeights = %d, want %d", got, want)
	}
	if got, want := estimate.KVCache, uint64(536_870_912); got != want {
		t.Fatalf("KVCache = %d, want %d", got, want)
	}
}

func TestLayerPlannerUsesMemoryBudget(t *testing.T) {
	planner := newTestPlanner(t, tinyLlamaConfig, 2)
	estimate := planner.Estimate(2048)
	budget := memory.Budget{
		HostBytes:   estimate.ResidentWeights,
		DeviceBytes: estimate.MaxLayer + estimate.KVCache,
	}
	if result := planner.Plan(2048, budget); !result.Fits {
		t.Fatalf("Plan() did not fit exact budget: %+v", result)
	}
	budget.DeviceBytes--
	if result := planner.Plan(2048, budget); result.Fits || result.DeviceFits {
		t.Fatalf("Plan() fit an insufficient device budget: %+v", result)
	}
}

func TestNewLayerPlannerRejectsInvalidInput(t *testing.T) {
	config := loadTestConfig(t, tinyLlamaConfig)
	if _, err := NewLayerPlanner(config, 0); err == nil {
		t.Fatal("NewLayerPlanner() accepted zero bytes per element")
	}
	config.NumHiddenLayers = 0
	if _, err := NewLayerPlanner(config, 2); err == nil {
		t.Fatal("NewLayerPlanner() accepted an invalid config")
	}
}

func TestLayerPlannerSaturatesOnOverflow(t *testing.T) {
	config := loadTestConfig(t, tinyLlamaConfig)
	planner, err := NewLayerPlanner(config, math.MaxUint64)
	if err != nil {
		t.Fatal(err)
	}
	if got := planner.MaxLayerBytes(); got != math.MaxUint64 {
		t.Fatalf("MaxLayerBytes() = %d, want MaxUint64", got)
	}
	if got := planner.Estimate(2048).ResidentWeights; got != math.MaxUint64 {
		t.Fatalf("ResidentWeights = %d, want MaxUint64", got)
	}
}

func newTestPlanner(t *testing.T, input string, bytesPerElement uint64) LayerPlanner {
	t.Helper()
	planner, err := NewLayerPlanner(loadTestConfig(t, input), bytesPerElement)
	if err != nil {
		t.Fatalf("NewLayerPlanner() error = %v", err)
	}
	return planner
}

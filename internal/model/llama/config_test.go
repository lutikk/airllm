package llama

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const tinyLlamaConfig = `{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 2048,
  "intermediate_size": 11008,
  "num_attention_heads": 32,
  "num_key_value_heads": 4,
  "num_hidden_layers": 22,
  "vocab_size": 32000,
  "rms_norm_eps": 0.00001,
  "max_position_embeddings": 2048,
  "tie_word_embeddings": false,
  "eos_token_id": 2,
  "model_type": "llama",
  "torch_dtype": "float16"
}`

func TestLoadConfigTinyLlamaDefaults(t *testing.T) {
	config := loadTestConfig(t, tinyLlamaConfig)
	if config.RopeTheta != 10000 {
		t.Fatalf("RopeTheta = %v, want 10000", config.RopeTheta)
	}
	if config.HeadDim != 64 {
		t.Fatalf("HeadDim = %d, want 64", config.HeadDim)
	}
	if len(config.EOSTokenIDs) != 1 || config.EOSTokenIDs[0] != 2 {
		t.Fatalf("EOSTokenIDs = %v, want [2]", config.EOSTokenIDs)
	}
	if config.SlidingWindow != 0 {
		t.Fatalf("SlidingWindow = %d, want 0", config.SlidingWindow)
	}
}

func TestLoadConfigMistral(t *testing.T) {
	input := `{
	  "architectures": ["MistralForCausalLM"],
	  "hidden_size": 4096,
	  "intermediate_size": 14336,
	  "num_attention_heads": 32,
	  "num_key_value_heads": 8,
	  "num_hidden_layers": 32,
	  "vocab_size": 32000,
	  "rms_norm_eps": 0.00001,
	  "rope_theta": 1000000,
	  "head_dim": 128,
	  "max_position_embeddings": 32768,
	  "sliding_window": 4096,
	  "tie_word_embeddings": true,
	  "eos_token_id": [2, 32000]
	}`
	// Keep the final token inside the configured vocabulary.
	input = strings.Replace(input, "32000]", "31999]", 1)
	config := loadTestConfig(t, input)
	if config.HeadDim != 128 || config.RopeTheta != 1_000_000 {
		t.Fatalf("loaded rope config = head_dim %d, theta %v", config.HeadDim, config.RopeTheta)
	}
	if config.SlidingWindow != 4096 || !config.TieWordEmbeddings {
		t.Fatalf("loaded Mistral options = %+v", config)
	}
	if len(config.EOSTokenIDs) != 2 || config.EOSTokenIDs[1] != 31999 {
		t.Fatalf("EOSTokenIDs = %v, want [2 31999]", config.EOSTokenIDs)
	}
}

func TestLoadConfigRejectsInvalidConfigs(t *testing.T) {
	tests := []struct {
		name    string
		old     string
		new     string
		wantErr string
	}{
		{"unsupported architecture", "LlamaForCausalLM", "Qwen3ForCausalLM", "unsupported architectures"},
		{"missing layers", `"num_hidden_layers": 22`, `"num_hidden_layers": 0`, "num_hidden_layers"},
		{"negative dimensions", `"hidden_size": 2048`, `"hidden_size": -2048`, "hidden_size"},
		{"indivisible attention heads", `"num_attention_heads": 32`, `"num_attention_heads": 30`, "divisible"},
		{"incompatible head dim", `"max_position_embeddings": 2048`, `"head_dim": 128, "max_position_embeddings": 2048`, "head_dim"},
		{"invalid grouped query attention", `"num_key_value_heads": 4`, `"num_key_value_heads": 5`, "num_key_value_heads"},
		{"invalid epsilon", `"rms_norm_eps": 0.00001`, `"rms_norm_eps": 0`, "rms_norm_eps"},
		{"invalid rope theta", `"max_position_embeddings": 2048`, `"rope_theta": -1, "max_position_embeddings": 2048`, "rope_theta"},
		{"negative sliding window", `"max_position_embeddings": 2048`, `"sliding_window": -1, "max_position_embeddings": 2048`, "sliding_window"},
		{"missing eos", `"eos_token_id": 2`, `"eos_token_id": []`, "eos_token_id"},
		{"eos outside vocabulary", `"eos_token_id": 2`, `"eos_token_id": 32000`, "outside vocabulary"},
		{"fractional eos", `"eos_token_id": 2`, `"eos_token_id": 2.5`, "integer"},
		{"wrong field type", `"vocab_size": 32000`, `"vocab_size": "32000"`, "vocab_size"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			input := strings.Replace(tinyLlamaConfig, test.old, test.new, 1)
			_, err := LoadConfig(writeTestConfig(t, input))
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("LoadConfig() error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

func TestLoadConfigRejectsMalformedJSONAndMissingFile(t *testing.T) {
	if _, err := LoadConfig(writeTestConfig(t, `{"architectures":`)); err == nil {
		t.Fatal("LoadConfig() accepted malformed JSON")
	}
	if _, err := LoadConfig(filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Fatal("LoadConfig() accepted a missing file")
	}
}

func loadTestConfig(t *testing.T, input string) Config {
	t.Helper()
	config, err := LoadConfig(writeTestConfig(t, input))
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}
	return config
}

func writeTestConfig(t *testing.T, input string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(input), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

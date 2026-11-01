package ollama

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestPullBearerChallengeManifestBlobsAndProgress(t *testing.T) {
	model := []byte("GGUF small test weights")
	config := []byte(`{"format":"gguf"}`)
	license := []byte("test license")
	manifest := makeManifest(t, model, config, license)
	manifestData, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}

	var server *httptest.Server
	var mu sync.Mutex
	manifestRequests := 0
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/token":
			if r.URL.Query().Get("service") != "ollama-registry" || r.URL.Query().Get("scope") != "repository:library/qwen2.5:pull" {
				t.Errorf("token query = %s", r.URL.RawQuery)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"token":"registry-token"}`))
		case "/v2/library/qwen2.5/manifests/0.5b":
			mu.Lock()
			manifestRequests++
			mu.Unlock()
			if r.Header.Get("Authorization") == "" {
				w.Header().Set("WWW-Authenticate", fmt.Sprintf(`Bearer realm="%s/token",service="ollama-registry",scope="repository:library/qwen2.5:pull"`, server.URL))
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
			assertBearer(t, r)
			_, _ = w.Write(manifestData)
		case "/v2/library/qwen2.5/blobs/" + manifest.Layers[0].Digest:
			assertBearer(t, r)
			_, _ = w.Write(model)
		case "/v2/library/qwen2.5/blobs/" + manifest.Config.Digest:
			assertBearer(t, r)
			_, _ = w.Write(config)
		case "/v2/library/qwen2.5/blobs/" + manifest.Layers[1].Digest:
			assertBearer(t, r)
			_, _ = w.Write(license)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
	destDir := t.TempDir()
	var progress []Progress
	err = client.Pull(context.Background(), "qwen2.5:0.5b", destDir, func(p Progress) {
		progress = append(progress, p)
	})
	if err != nil {
		t.Fatalf("Pull: %v", err)
	}
	assertFile(t, filepath.Join(destDir, "model.gguf"), model)
	assertFile(t, filepath.Join(destDir, "config.json"), config)
	assertFile(t, filepath.Join(destDir, "LICENSE"), license)
	assertFile(t, filepath.Join(destDir, "manifest.json"), manifestData)
	if len(progress) == 0 {
		t.Fatal("Pull emitted no progress")
	}
	var modelProgress []Progress
	for _, p := range progress {
		if p.Filename == "model.gguf" {
			modelProgress = append(modelProgress, p)
		}
	}
	if len(modelProgress) < 2 {
		t.Fatalf("model progress = %v", modelProgress)
	}
	last := modelProgress[len(modelProgress)-1]
	if last.Downloaded != int64(len(model)) || last.Total != int64(len(model)) {
		t.Fatalf("final model progress = %+v", last)
	}

	requestCountBefore := manifestRequests
	if err := client.Pull(context.Background(), "library/qwen2.5:0.5b", destDir, nil); err != nil {
		t.Fatalf("cached Pull: %v", err)
	}
	if manifestRequests != requestCountBefore+2 {
		t.Fatalf("manifest requests = %d, want %d", manifestRequests, requestCountBefore+2)
	}
}

func TestPullResumesModelBlob(t *testing.T) {
	model := []byte("hello GGUF world")
	manifest := makeManifest(t, model, nil, nil)
	manifestData, _ := json.Marshal(manifest)
	server := newManifestServer(t, manifestData, func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Range"); got != "bytes=6-" {
			t.Errorf("Range = %q, want bytes=6-", got)
		}
		w.Header().Set("Content-Range", fmt.Sprintf("bytes 6-%d/%d", len(model)-1, len(model)))
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write(model[6:])
	})
	defer server.Close()

	destDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(destDir, "model.gguf.part"), model[:6], 0o644); err != nil {
		t.Fatal(err)
	}
	client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
	if err := client.Pull(context.Background(), "qwen2.5", destDir, nil); err != nil {
		t.Fatalf("Pull: %v", err)
	}
	assertFile(t, filepath.Join(destDir, "model.gguf"), model)
	if _, err := os.Stat(filepath.Join(destDir, "model.gguf.part")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("partial file remains: %v", err)
	}
}

func TestPullAcceptsDockerConfigMediaType(t *testing.T) {
	model := []byte("tinyllama GGUF")
	config := []byte(`{"architecture":"llama"}`)
	manifest := makeManifest(t, model, config, nil)
	manifest.Config.MediaType = "application/vnd.docker.container.image.v1+json"
	manifestData, _ := json.Marshal(manifest)
	server := newManifestServer(t, manifestData, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path[strings.LastIndex(r.URL.Path, "/")+1:] {
		case manifest.Layers[0].Digest:
			_, _ = w.Write(model)
		case manifest.Config.Digest:
			_, _ = w.Write(config)
		default:
			http.NotFound(w, r)
		}
	})
	defer server.Close()

	destDir := t.TempDir()
	client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
	if err := client.Pull(context.Background(), "tinyllama:latest", destDir, nil); err != nil {
		t.Fatalf("Pull with Docker config media type: %v", err)
	}
	assertFile(t, filepath.Join(destDir, "model.gguf"), model)
	assertFile(t, filepath.Join(destDir, "config.json"), config)
	assertFile(t, filepath.Join(destDir, "manifest.json"), manifestData)
}

func TestPullRestartsWhenServerIgnoresRange(t *testing.T) {
	model := []byte("fresh complete model")
	manifest := makeManifest(t, model, nil, nil)
	manifestData, _ := json.Marshal(manifest)
	server := newManifestServer(t, manifestData, func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Range"); got != "bytes=5-" {
			t.Errorf("Range = %q, want bytes=5-", got)
		}
		_, _ = w.Write(model)
	})
	defer server.Close()

	destDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(destDir, "model.gguf.part"), []byte("stale"), 0o644); err != nil {
		t.Fatal(err)
	}
	client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
	if err := client.Pull(context.Background(), "qwen2.5:latest", destDir, nil); err != nil {
		t.Fatalf("Pull: %v", err)
	}
	assertFile(t, filepath.Join(destDir, "model.gguf"), model)
}

func TestPullErrorsPreservePartialModel(t *testing.T) {
	model := []byte("expected model")
	manifest := makeManifest(t, model, nil, nil)
	manifestData, _ := json.Marshal(manifest)

	t.Run("digest mismatch", func(t *testing.T) {
		server := newManifestServer(t, manifestData, func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte("invalid model!"))
		})
		defer server.Close()
		destDir := t.TempDir()
		client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
		err := client.Pull(context.Background(), "qwen2.5", destDir, nil)
		if err == nil || !strings.Contains(err.Error(), "digest mismatch") {
			t.Fatalf("Pull error = %v", err)
		}
		assertFile(t, filepath.Join(destDir, "model.gguf.part"), []byte("invalid model!"))
		assertNotExists(t, filepath.Join(destDir, "model.gguf"))
		assertNotExists(t, filepath.Join(destDir, "manifest.json"))
	})

	t.Run("invalid resume range", func(t *testing.T) {
		server := newManifestServer(t, manifestData, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Range", fmt.Sprintf("bytes 0-%d/%d", len(model)-6, len(model)))
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write(model[5:])
		})
		defer server.Close()
		destDir := t.TempDir()
		if err := os.WriteFile(filepath.Join(destDir, "model.gguf.part"), model[:5], 0o644); err != nil {
			t.Fatal(err)
		}
		client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
		err := client.Pull(context.Background(), "qwen2.5", destDir, nil)
		if err == nil || !strings.Contains(err.Error(), "invalid Content-Range") {
			t.Fatalf("Pull error = %v", err)
		}
		assertFile(t, filepath.Join(destDir, "model.gguf.part"), model[:5])
	})
}

func TestPullRegistryAndManifestErrors(t *testing.T) {
	t.Run("manifest status", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte("registry unavailable"))
		}))
		defer server.Close()
		client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
		err := client.Pull(context.Background(), "qwen2.5", t.TempDir(), nil)
		var statusErr *StatusError
		if !errors.As(err, &statusErr) || statusErr.StatusCode != http.StatusInternalServerError {
			t.Fatalf("Pull error = %T %v", err, err)
		}
	})

	t.Run("missing GGUF layer", func(t *testing.T) {
		manifestData := []byte(`{"schemaVersion":2,"layers":[]}`)
		server := newManifestServer(t, manifestData, nil)
		defer server.Close()
		client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
		err := client.Pull(context.Background(), "qwen2.5", t.TempDir(), nil)
		if err == nil || !strings.Contains(err.Error(), "no GGUF model layer") {
			t.Fatalf("Pull error = %v", err)
		}
	})

	t.Run("token status", func(t *testing.T) {
		var server *httptest.Server
		server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/token" {
				w.WriteHeader(http.StatusForbidden)
				return
			}
			w.Header().Set("WWW-Authenticate", fmt.Sprintf(`Bearer realm="%s/token"`, server.URL))
			w.WriteHeader(http.StatusUnauthorized)
		}))
		defer server.Close()
		client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
		err := client.Pull(context.Background(), "qwen2.5", t.TempDir(), nil)
		var statusErr *StatusError
		if !errors.As(err, &statusErr) || statusErr.StatusCode != http.StatusForbidden {
			t.Fatalf("Pull error = %T %v", err, err)
		}
	})
}

func TestParseReference(t *testing.T) {
	tests := []struct {
		input      string
		repository string
		tag        string
	}{
		{"qwen2.5:0.5b", "library/qwen2.5", "0.5b"},
		{"library/qwen2.5:0.5b", "library/qwen2.5", "0.5b"},
		{"qwen2.5", "library/qwen2.5", "latest"},
		{"registry.ollama.ai/library/qwen2.5:0.5b", "library/qwen2.5", "0.5b"},
		{"https://registry.ollama.ai/library/qwen2.5:0.5b", "library/qwen2.5", "0.5b"},
		{"https://registry.ollama.ai/v2/library/qwen2.5/manifests/0.5b", "library/qwen2.5", "0.5b"},
	}
	for _, test := range tests {
		t.Run(test.input, func(t *testing.T) {
			ref, err := parseReference(test.input, "")
			if err != nil {
				t.Fatalf("parseReference: %v", err)
			}
			if ref.repository != test.repository || ref.tag != test.tag || ref.baseURL != defaultRegistry {
				t.Fatalf("reference = %+v", ref)
			}
		})
	}
	for _, input := range []string{"", "../model", "org/model/extra", "qwen2.5@", "qwen2.5:"} {
		t.Run("invalid "+input, func(t *testing.T) {
			if _, err := parseReference(input, ""); err == nil {
				t.Fatalf("parseReference(%q) succeeded", input)
			}
		})
	}
}

func makeManifest(t *testing.T, model, config, license []byte) Manifest {
	t.Helper()
	manifest := Manifest{
		SchemaVersion: 2,
		MediaType:     "application/vnd.oci.image.manifest.v1+json",
		Layers: []Layer{{
			MediaType: modelMediaType,
			Digest:    testDigest(model),
			Size:      int64(len(model)),
		}},
	}
	if config != nil {
		manifest.Config = Layer{MediaType: configMediaType, Digest: testDigest(config), Size: int64(len(config))}
	}
	if license != nil {
		manifest.Layers = append(manifest.Layers, Layer{MediaType: licenseMediaType, Digest: testDigest(license), Size: int64(len(license))})
	}
	return manifest
}

func newManifestServer(t *testing.T, manifest []byte, blob http.HandlerFunc) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/manifests/") {
			_, _ = w.Write(manifest)
			return
		}
		if strings.Contains(r.URL.Path, "/blobs/") && blob != nil {
			blob(w, r)
			return
		}
		http.NotFound(w, r)
	}))
}

func testDigest(data []byte) string {
	return fmt.Sprintf("sha256:%x", sha256.Sum256(data))
}

func assertBearer(t *testing.T, r *http.Request) {
	t.Helper()
	if got := r.Header.Get("Authorization"); got != "Bearer registry-token" {
		t.Errorf("Authorization = %q", got)
	}
}

func assertFile(t *testing.T, filename string, want []byte) {
	t.Helper()
	got, err := os.ReadFile(filename)
	if err != nil {
		t.Fatalf("ReadFile(%s): %v", filename, err)
	}
	if string(got) != string(want) {
		t.Fatalf("%s = %q, want %q", filename, got, want)
	}
}

func assertNotExists(t *testing.T, filename string) {
	t.Helper()
	if _, err := os.Stat(filename); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("%s exists or stat failed: %v", filename, err)
	}
}

func TestInspectResolvesManifestModelAndConfigWithoutWeights(t *testing.T) {
	model := []byte("large GGUF contents must not be downloaded")
	config := []byte(`{"architecture":"llama","parameter_size":"1.1B"}`)
	manifest := makeManifest(t, model, config, nil)
	manifestData, _ := json.Marshal(manifest)
	modelRequests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v2/library/tinyllama/manifests/latest":
			_, _ = w.Write(manifestData)
		case "/v2/library/tinyllama/blobs/" + manifest.Config.Digest:
			_, _ = w.Write(config)
		case "/v2/library/tinyllama/blobs/" + manifest.Layers[0].Digest:
			modelRequests++
			_, _ = w.Write(model)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
	inspection, err := client.Inspect(context.Background(), "tinyllama")
	if err != nil {
		t.Fatalf("Inspect: %v", err)
	}
	if inspection.Manifest.MediaType != manifest.MediaType {
		t.Errorf("manifest media type = %q, want %q", inspection.Manifest.MediaType, manifest.MediaType)
	}
	if inspection.Model.MediaType != modelMediaType || inspection.Model.Digest != manifest.Layers[0].Digest {
		t.Errorf("model layer = %+v", inspection.Model)
	}
	if inspection.ModelSize != int64(len(model)) {
		t.Errorf("model size = %d, want %d", inspection.ModelSize, len(model))
	}
	if inspection.Manifest.Config.MediaType != configMediaType {
		t.Errorf("config media type = %q", inspection.Manifest.Config.MediaType)
	}
	if string(inspection.Config) != string(config) {
		t.Errorf("config = %s, want %s", inspection.Config, config)
	}
	if modelRequests != 0 {
		t.Fatalf("Inspect downloaded model blob %d times", modelRequests)
	}
}

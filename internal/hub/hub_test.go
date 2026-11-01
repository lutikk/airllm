package hub

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
)

func TestEnsureModelSingleFileAndCache(t *testing.T) {
	files := map[string]string{
		"config.json":            `{"model_type":"test"}`,
		"tokenizer.json":         `{"version":"1.0"}`,
		"tokenizer_config.json":  `{}`,
		"generation_config.json": `{"max_length":20}`,
		"model.safetensors":      "single-file-weights",
	}
	var mu sync.Mutex
	var requests []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer secret" {
			t.Errorf("Authorization = %q", got)
		}
		mu.Lock()
		requests = append(requests, r.URL.EscapedPath())
		mu.Unlock()
		name := strings.TrimPrefix(r.URL.Path, "/org/model/resolve/main/")
		if body, ok := files[name]; ok {
			_, _ = w.Write([]byte(body))
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	client := &Client{BaseURL: server.URL, HTTPClient: server.Client(), Token: "secret"}
	cacheDir := t.TempDir()
	localDir, err := client.EnsureModel(context.Background(), "org/model", cacheDir)
	if err != nil {
		t.Fatalf("EnsureModel: %v", err)
	}
	wantDir := filepath.Join(cacheDir, "org", "model")
	if localDir != wantDir {
		t.Fatalf("localDir = %q, want %q", localDir, wantDir)
	}
	for name, want := range files {
		assertFileContents(t, filepath.Join(localDir, name), want)
		if _, err := os.Stat(filepath.Join(localDir, name) + ".part"); !errors.Is(err, os.ErrNotExist) {
			t.Errorf("partial file remains for %s: %v", name, err)
		}
	}

	mu.Lock()
	requestCount := len(requests)
	mu.Unlock()
	if _, err := client.EnsureModel(context.Background(), "org/model", cacheDir); err != nil {
		t.Fatalf("cached EnsureModel: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(requests) != requestCount {
		t.Fatalf("cached files were requested again: %v", requests[requestCount:])
	}
}

func TestEnsureModelShardedAndOptionalGenerationConfig(t *testing.T) {
	index := `{"metadata":{"total_size":12},"weight_map":{"b":"shards/model-00002-of-00002.safetensors","a":"shards/model-00001-of-00002.safetensors","c":"shards/model-00001-of-00002.safetensors"}}`
	files := map[string]string{
		"config.json":                             `{}`,
		"tokenizer.json":                          `{}`,
		"tokenizer_config.json":                   `{}`,
		"model.safetensors.index.json":            index,
		"shards/model-00001-of-00002.safetensors": "first",
		"shards/model-00002-of-00002.safetensors": "second",
	}
	var mu sync.Mutex
	var requested []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name := strings.TrimPrefix(r.URL.Path, "/org/model/resolve/main/")
		mu.Lock()
		requested = append(requested, name)
		mu.Unlock()
		if body, ok := files[name]; ok {
			_, _ = w.Write([]byte(body))
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
	localDir, err := client.EnsureModel(context.Background(), "org/model", t.TempDir())
	if err != nil {
		t.Fatalf("EnsureModel: %v", err)
	}
	for name, want := range files {
		assertFileContents(t, filepath.Join(localDir, filepath.FromSlash(name)), want)
	}

	mu.Lock()
	defer mu.Unlock()
	sort.Strings(requested)
	if countString(requested, "shards/model-00001-of-00002.safetensors") != 1 {
		t.Fatalf("shared shard was not deduplicated: %v", requested)
	}
	if !containsString(requested, "generation_config.json") {
		t.Fatalf("optional generation config was not attempted: %v", requested)
	}
}

func TestDownloadResumesPartialFile(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Range"); got != "bytes=6-" {
			t.Errorf("Range = %q, want bytes=6-", got)
		}
		w.Header().Set("Content-Range", "bytes 6-10/11")
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write([]byte("world"))
	}))
	defer server.Close()

	destination := filepath.Join(t.TempDir(), "model.bin")
	if err := os.WriteFile(destination+".part", []byte("hello "), 0o644); err != nil {
		t.Fatal(err)
	}
	client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
	if err := client.Download(context.Background(), "org/model", "model.bin", destination); err != nil {
		t.Fatalf("Download: %v", err)
	}
	assertFileContents(t, destination, "hello world")
}

func TestDownloadRestartsWhenRangeIgnored(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Range"); got != "bytes=5-" {
			t.Errorf("Range = %q, want bytes=5-", got)
		}
		_, _ = w.Write([]byte("fresh-content"))
	}))
	defer server.Close()

	destination := filepath.Join(t.TempDir(), "model.bin")
	if err := os.WriteFile(destination+".part", []byte("stale"), 0o644); err != nil {
		t.Fatal(err)
	}
	client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
	if err := client.Download(context.Background(), "org/model", "model.bin", destination); err != nil {
		t.Fatalf("Download: %v", err)
	}
	assertFileContents(t, destination, "fresh-content")
}

func TestDownloadCommitsCompletePartialFileAfter416(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Range", "bytes */11")
		w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
	}))
	defer server.Close()

	destination := filepath.Join(t.TempDir(), "model.bin")
	if err := os.WriteFile(destination+".part", []byte("hello world"), 0o644); err != nil {
		t.Fatal(err)
	}
	client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
	if err := client.Download(context.Background(), "org/model", "model.bin", destination); err != nil {
		t.Fatalf("Download: %v", err)
	}
	assertFileContents(t, destination, "hello world")
}

func TestDownloadRejectsInvalidContentRange(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Range", "bytes 0-4/11")
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write([]byte("world"))
	}))
	defer server.Close()

	destination := filepath.Join(t.TempDir(), "model.bin")
	if err := os.WriteFile(destination+".part", []byte("hello "), 0o644); err != nil {
		t.Fatal(err)
	}
	client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
	err := client.Download(context.Background(), "org/model", "model.bin", destination)
	if err == nil || !strings.Contains(err.Error(), "invalid Content-Range") {
		t.Fatalf("Download error = %v", err)
	}
	assertFileContents(t, destination+".part", "hello ")
}

func TestEnsureModelRejectsTraversalInIndex(t *testing.T) {
	files := map[string]string{
		"config.json":                  `{}`,
		"tokenizer.json":               `{}`,
		"tokenizer_config.json":        `{}`,
		"model.safetensors.index.json": `{"weight_map":{"x":"../../outside.safetensors"}}`,
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name := strings.TrimPrefix(r.URL.Path, "/org/model/resolve/main/")
		if body, ok := files[name]; ok {
			_, _ = w.Write([]byte(body))
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
	_, err := client.EnsureModel(context.Background(), "org/model", t.TempDir())
	if err == nil || !strings.Contains(err.Error(), "invalid shard") {
		t.Fatalf("EnsureModel error = %v", err)
	}
}

func TestResolveRejectsUnsafePaths(t *testing.T) {
	client := &Client{BaseURL: "http://example.invalid"}
	tests := []struct {
		repoID   string
		filename string
	}{
		{"../model", "config.json"},
		{"org/../model", "config.json"},
		{"org/model/extra", "config.json"},
		{"org/model", "../config.json"},
		{"org/model", "/config.json"},
		{"org/model", `..\config.json`},
		{"org/model", "nested/../../config.json"},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%s/%s", test.repoID, test.filename), func(t *testing.T) {
			if _, err := client.Resolve(context.Background(), test.repoID, test.filename); err == nil {
				t.Fatal("Resolve accepted an unsafe path")
			}
		})
	}
}

func TestResolveStatusError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("upstream unavailable"))
	}))
	defer server.Close()

	client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
	_, err := client.Resolve(context.Background(), "org/model", "config.json")
	var statusErr *StatusError
	if !errors.As(err, &statusErr) {
		t.Fatalf("Resolve error = %T %v", err, err)
	}
	if statusErr.StatusCode != http.StatusInternalServerError || !strings.Contains(statusErr.Body, "upstream unavailable") {
		t.Fatalf("StatusError = %#v", statusErr)
	}
}

func TestResolveHonorsContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
	_, err := client.Resolve(ctx, "org/model", "config.json")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Resolve error = %v, want context.Canceled", err)
	}
}

func TestResolveEscapesPathSegments(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.EscapedPath()
		_, _ = w.Write([]byte("{}"))
	}))
	defer server.Close()

	client := &Client{BaseURL: server.URL, HTTPClient: server.Client()}
	resp, err := client.Resolve(context.Background(), "org/model", "nested/a b.json")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	resp.Body.Close()
	if gotPath != "/org/model/resolve/main/nested/a%20b.json" {
		t.Fatalf("path = %q", gotPath)
	}
}

func assertFileContents(t *testing.T, filename, want string) {
	t.Helper()
	got, err := os.ReadFile(filename)
	if err != nil {
		t.Fatalf("ReadFile(%s): %v", filename, err)
	}
	if string(got) != want {
		t.Fatalf("%s = %q, want %q", filename, got, want)
	}
}

func containsString(values []string, want string) bool {
	return countString(values, want) > 0
}

func countString(values []string, want string) int {
	count := 0
	for _, value := range values {
		if value == want {
			count++
		}
	}
	return count
}

func TestRequiredFilesRemainStable(t *testing.T) {
	want := []string{"config.json", "tokenizer.json", "tokenizer_config.json"}
	if !reflect.DeepEqual(requiredFiles, want) {
		t.Fatalf("requiredFiles = %v", requiredFiles)
	}
}

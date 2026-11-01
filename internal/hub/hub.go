// Package hub downloads model files from Hugging Face Hub.
package hub

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const defaultBaseURL = "https://huggingface.co"

var requiredFiles = []string{
	"config.json",
	"tokenizer.json",
	"tokenizer_config.json",
}

// Client is a Hugging Face Hub client. The zero value uses the public Hub,
// http.DefaultTransport-compatible timeouts, and HF_TOKEN from the environment.
type Client struct {
	// BaseURL overrides the Hub endpoint. It is primarily useful for tests or mirrors.
	BaseURL string
	// HTTPClient overrides the default HTTP client. A client-wide Timeout should be
	// avoided for large model downloads; use context cancellation instead.
	HTTPClient *http.Client
	// Token overrides HF_TOKEN. An empty token means anonymous access.
	Token string
}

// StatusError reports a non-successful HTTP response from the Hub.
type StatusError struct {
	RepoID     string
	Filename   string
	StatusCode int
	Status     string
	Header     http.Header
	Body       string
}

func (e *StatusError) Error() string {
	if e.Body != "" {
		return fmt.Sprintf("hub: resolve %s/%s: %s: %s", e.RepoID, e.Filename, e.Status, e.Body)
	}
	return fmt.Sprintf("hub: resolve %s/%s: %s", e.RepoID, e.Filename, e.Status)
}

// EnsureModel downloads a model into cacheDir/repoID and returns its local
// directory. Existing files are reused. Large files are streamed to disk.
func EnsureModel(ctx context.Context, repoID, cacheDir string) (string, error) {
	return (&Client{}).EnsureModel(ctx, repoID, cacheDir)
}

// EnsureModel downloads a model into cacheDir/repoID and returns its local
// directory. Existing files are reused. Large files are streamed to disk.
func (c *Client) EnsureModel(ctx context.Context, repoID, cacheDir string) (string, error) {
	if err := validateRepoID(repoID); err != nil {
		return "", err
	}
	if cacheDir == "" {
		return "", errors.New("hub: cache directory is empty")
	}

	localDir := filepath.Join(cacheDir, filepath.FromSlash(repoID))
	if err := os.MkdirAll(localDir, 0o755); err != nil {
		return "", fmt.Errorf("hub: create model cache: %w", err)
	}

	for _, name := range requiredFiles {
		if err := c.Download(ctx, repoID, name, filepath.Join(localDir, filepath.FromSlash(name))); err != nil {
			return "", err
		}
	}

	generationPath := filepath.Join(localDir, "generation_config.json")
	if err := c.Download(ctx, repoID, "generation_config.json", generationPath); err != nil && !isNotFound(err) {
		return "", err
	}

	modelPath := filepath.Join(localDir, "model.safetensors")
	err := c.Download(ctx, repoID, "model.safetensors", modelPath)
	if err == nil {
		return localDir, nil
	}
	if !isNotFound(err) {
		return "", err
	}

	indexPath := filepath.Join(localDir, "model.safetensors.index.json")
	if err := c.Download(ctx, repoID, "model.safetensors.index.json", indexPath); err != nil {
		return "", fmt.Errorf("hub: neither model.safetensors nor a sharded model index could be downloaded: %w", err)
	}

	indexFile, err := os.Open(indexPath)
	if err != nil {
		return "", fmt.Errorf("hub: open sharded model index: %w", err)
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	decodeErr := json.NewDecoder(indexFile).Decode(&index)
	closeErr := indexFile.Close()
	if decodeErr != nil {
		return "", fmt.Errorf("hub: decode sharded model index: %w", decodeErr)
	}
	if closeErr != nil {
		return "", fmt.Errorf("hub: close sharded model index: %w", closeErr)
	}
	if len(index.WeightMap) == 0 {
		return "", errors.New("hub: sharded model index has an empty weight_map")
	}

	uniqueShards := make(map[string]struct{}, len(index.WeightMap))
	for tensor, shard := range index.WeightMap {
		if err := validateFilename(shard); err != nil {
			return "", fmt.Errorf("hub: invalid shard for tensor %q: %w", tensor, err)
		}
		uniqueShards[shard] = struct{}{}
	}
	shards := make([]string, 0, len(uniqueShards))
	for shard := range uniqueShards {
		shards = append(shards, shard)
	}
	sort.Strings(shards)
	for _, shard := range shards {
		if err := c.Download(ctx, repoID, shard, filepath.Join(localDir, filepath.FromSlash(shard))); err != nil {
			return "", err
		}
	}

	return localDir, nil
}

// Resolve opens a model file from the repository's main revision. The caller
// must close the returned response body.
func (c *Client) Resolve(ctx context.Context, repoID, filename string) (*http.Response, error) {
	endpoint, err := c.resolveURL(repoID, filename)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("hub: create resolve request: %w", err)
	}
	token := c.Token
	if token == "" {
		token = os.Getenv("HF_TOKEN")
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("Accept", "*/*")

	resp, err := c.httpClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("hub: resolve %s/%s: %w", repoID, filename, err)
	}
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return resp, nil
	}
	defer resp.Body.Close()
	body, readErr := io.ReadAll(io.LimitReader(resp.Body, 4096))
	if readErr != nil {
		return nil, fmt.Errorf("hub: resolve %s/%s: %s (read error response: %v)", repoID, filename, resp.Status, readErr)
	}
	return nil, &StatusError{
		RepoID:     repoID,
		Filename:   filename,
		StatusCode: resp.StatusCode,
		Status:     resp.Status,
		Header:     resp.Header.Clone(),
		Body:       strings.TrimSpace(string(body)),
	}
}

// Download streams a repository file to destination using destination.part.
// A partial file is resumed when the server honors Range requests. The final
// rename is atomic within the destination directory.
func (c *Client) Download(ctx context.Context, repoID, filename, destination string) error {
	if err := validateFilename(filename); err != nil {
		return err
	}
	if destination == "" {
		return errors.New("hub: download destination is empty")
	}
	if info, err := os.Stat(destination); err == nil {
		if !info.Mode().IsRegular() {
			return fmt.Errorf("hub: cached file %s is not a regular file", destination)
		}
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("hub: inspect cached file: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		return fmt.Errorf("hub: create download directory: %w", err)
	}

	partPath := destination + ".part"
	part, err := os.OpenFile(partPath, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return fmt.Errorf("hub: open partial download: %w", err)
	}
	offset, err := part.Seek(0, io.SeekEnd)
	if err != nil {
		part.Close()
		return fmt.Errorf("hub: inspect partial download: %w", err)
	}

	resp, err := c.resolveWithRange(ctx, repoID, filename, offset)
	if err != nil {
		var statusErr *StatusError
		if offset > 0 && errors.As(err, &statusErr) && statusErr.StatusCode == http.StatusRequestedRangeNotSatisfiable && partialFileComplete(statusErr.Header.Get("Content-Range"), offset) {
			if closeErr := part.Close(); closeErr != nil {
				return fmt.Errorf("hub: close completed partial download: %w", closeErr)
			}
			if renameErr := os.Rename(partPath, destination); renameErr != nil {
				return fmt.Errorf("hub: commit download: %w", renameErr)
			}
			return nil
		}
		part.Close()
		return err
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		if offset > 0 {
			if err := part.Truncate(0); err != nil {
				part.Close()
				return fmt.Errorf("hub: restart unsupported resume: %w", err)
			}
		}
		if _, err := part.Seek(0, io.SeekStart); err != nil {
			part.Close()
			return fmt.Errorf("hub: rewind partial download: %w", err)
		}
	case http.StatusPartialContent:
		start, ok := contentRangeStart(resp.Header.Get("Content-Range"))
		if !ok || start != offset {
			part.Close()
			return fmt.Errorf("hub: invalid Content-Range %q for resume at byte %d", resp.Header.Get("Content-Range"), offset)
		}
		if _, err := part.Seek(offset, io.SeekStart); err != nil {
			part.Close()
			return fmt.Errorf("hub: resume partial download: %w", err)
		}
	default:
		part.Close()
		return fmt.Errorf("hub: unexpected download status %s", resp.Status)
	}

	buffer := make([]byte, 128*1024)
	if _, err := io.CopyBuffer(part, resp.Body, buffer); err != nil {
		part.Close()
		return fmt.Errorf("hub: download %s/%s: %w", repoID, filename, err)
	}
	if err := part.Sync(); err != nil {
		part.Close()
		return fmt.Errorf("hub: sync download: %w", err)
	}
	if err := part.Close(); err != nil {
		return fmt.Errorf("hub: close download: %w", err)
	}
	if err := os.Rename(partPath, destination); err != nil {
		return fmt.Errorf("hub: commit download: %w", err)
	}
	return nil
}

func (c *Client) resolveWithRange(ctx context.Context, repoID, filename string, offset int64) (*http.Response, error) {
	if offset == 0 {
		return c.Resolve(ctx, repoID, filename)
	}

	endpoint, err := c.resolveURL(repoID, filename)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("hub: create resume request: %w", err)
	}
	token := c.Token
	if token == "" {
		token = os.Getenv("HF_TOKEN")
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	resp, err := c.httpClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("hub: resume %s/%s: %w", repoID, filename, err)
	}
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return resp, nil
	}
	defer resp.Body.Close()
	body, readErr := io.ReadAll(io.LimitReader(resp.Body, 4096))
	if readErr != nil {
		return nil, fmt.Errorf("hub: resume %s/%s: %s (read error response: %v)", repoID, filename, resp.Status, readErr)
	}
	return nil, &StatusError{
		RepoID:     repoID,
		Filename:   filename,
		StatusCode: resp.StatusCode,
		Status:     resp.Status,
		Header:     resp.Header.Clone(),
		Body:       strings.TrimSpace(string(body)),
	}
}

func (c *Client) resolveURL(repoID, filename string) (string, error) {
	if err := validateRepoID(repoID); err != nil {
		return "", err
	}
	if err := validateFilename(filename); err != nil {
		return "", err
	}
	base := c.BaseURL
	if base == "" {
		base = defaultBaseURL
	}
	parsed, err := url.Parse(base)
	if err != nil {
		return "", fmt.Errorf("hub: invalid base URL: %w", err)
	}
	if (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", fmt.Errorf("hub: invalid base URL %q", base)
	}
	escapedRepo := escapePath(repoID)
	escapedFilename := escapePath(filename)
	return strings.TrimRight(base, "/") + "/" + escapedRepo + "/resolve/main/" + escapedFilename, nil
}

var defaultHTTPClient = &http.Client{
	Transport: &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		ForceAttemptHTTP2:     true,
		TLSHandshakeTimeout:   30 * time.Second,
		ResponseHeaderTimeout: 60 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	},
}

func (c *Client) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return defaultHTTPClient
}

func validateRepoID(repoID string) error {
	if repoID == "" {
		return errors.New("hub: repository ID is empty")
	}
	parts := strings.Split(repoID, "/")
	if len(parts) < 1 || len(parts) > 2 {
		return fmt.Errorf("hub: invalid repository ID %q", repoID)
	}
	for _, part := range parts {
		if err := validatePathSegment(part); err != nil {
			return fmt.Errorf("hub: invalid repository ID %q: %w", repoID, err)
		}
	}
	return nil
}

func validateFilename(filename string) error {
	if filename == "" || strings.HasPrefix(filename, "/") || strings.Contains(filename, "\\") {
		return fmt.Errorf("hub: invalid repository filename %q", filename)
	}
	if path.Clean(filename) != filename {
		return fmt.Errorf("hub: unsafe repository filename %q", filename)
	}
	for _, part := range strings.Split(filename, "/") {
		if err := validatePathSegment(part); err != nil {
			return fmt.Errorf("hub: unsafe repository filename %q: %w", filename, err)
		}
	}
	return nil
}

func validatePathSegment(segment string) error {
	if segment == "" || segment == "." || segment == ".." {
		return errors.New("empty or traversal path segment")
	}
	if strings.ContainsAny(segment, "\\\x00") {
		return errors.New("invalid path character")
	}
	for _, r := range segment {
		if r < 0x20 || r == 0x7f {
			return errors.New("control character in path")
		}
	}
	return nil
}

func escapePath(value string) string {
	parts := strings.Split(value, "/")
	for i, part := range parts {
		parts[i] = url.PathEscape(part)
	}
	return strings.Join(parts, "/")
}

func isNotFound(err error) bool {
	var statusErr *StatusError
	return errors.As(err, &statusErr) && statusErr.StatusCode == http.StatusNotFound
}

func contentRangeStart(value string) (int64, bool) {
	if !strings.HasPrefix(value, "bytes ") {
		return 0, false
	}
	rangeAndSize := strings.TrimPrefix(value, "bytes ")
	rangePart, _, found := strings.Cut(rangeAndSize, "/")
	if !found {
		return 0, false
	}
	startText, _, found := strings.Cut(rangePart, "-")
	if !found {
		return 0, false
	}
	start, err := strconv.ParseInt(startText, 10, 64)
	return start, err == nil && start >= 0
}

func partialFileComplete(value string, size int64) bool {
	prefix := "bytes */"
	if !strings.HasPrefix(value, prefix) {
		return false
	}
	total, err := strconv.ParseInt(strings.TrimPrefix(value, prefix), 10, 64)
	return err == nil && total == size
}

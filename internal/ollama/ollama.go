// Package ollama downloads Ollama models directly from an OCI-compatible
// registry. It does not require an Ollama installation or daemon.
package ollama

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	defaultRegistry  = "https://registry.ollama.ai"
	manifestLimit    = 16 << 20
	modelMediaType   = "application/vnd.ollama.image.model"
	configMediaType  = "application/vnd.ollama.image.config"
	licenseMediaType = "application/vnd.ollama.image.license"
)

// Progress describes the state of the file currently being downloaded.
type Progress struct {
	Filename   string
	Downloaded int64
	Total      int64
}

// Layer is a blob referenced by an Ollama manifest.
type Layer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
}

// Manifest is the subset of an OCI manifest used by Ollama.
type Manifest struct {
	SchemaVersion int     `json:"schemaVersion"`
	MediaType     string  `json:"mediaType"`
	Config        Layer   `json:"config"`
	Layers        []Layer `json:"layers"`
}

// StatusError reports a non-successful registry response.
type StatusError struct {
	URL        string
	StatusCode int
	Status     string
	Header     http.Header
	Body       string
}

func (e *StatusError) Error() string {
	if e.Body != "" {
		return fmt.Sprintf("ollama: GET %s: %s: %s", e.URL, e.Status, e.Body)
	}
	return fmt.Sprintf("ollama: GET %s: %s", e.URL, e.Status)
}

// Client downloads models from an Ollama registry. The zero value uses
// registry.ollama.ai and anonymous Bearer-challenge authentication.
type Client struct {
	// BaseURL overrides the registry endpoint. It is useful for tests and mirrors.
	BaseURL string
	// HTTPClient overrides the default HTTP client. Avoid a client-wide Timeout
	// for large downloads; use context cancellation instead.
	HTTPClient *http.Client
	// Token optionally supplies a Bearer token before the first challenge.
	Token string
}

// Pull downloads modelRef into destDir using the public Ollama registry.
func Pull(ctx context.Context, modelRef, destDir string, progress func(Progress)) error {
	return (&Client{}).Pull(ctx, modelRef, destDir, progress)
}

// Inspection describes a resolved model without downloading its GGUF weights.
type Inspection struct {
	Manifest  Manifest
	Model     Layer
	ModelSize int64
	Config    json.RawMessage
}

type resolvedModel struct {
	ref          reference
	session      *session
	manifest     Manifest
	manifestData []byte
	model        Layer
}

// Inspect resolves a model reference and downloads only its manifest and small
// config blob. Model contains the selected GGUF layer, including its media type
// and size; Config is nil when the manifest has no config descriptor.
func (c *Client) Inspect(ctx context.Context, modelRef string) (Inspection, error) {
	resolved, err := c.resolve(ctx, modelRef)
	if err != nil {
		return Inspection{}, err
	}
	inspection := Inspection{
		Manifest:  resolved.manifest,
		Model:     resolved.model,
		ModelSize: resolved.model.Size,
	}
	if resolved.manifest.Config.Digest != "" {
		config, err := c.readConfig(ctx, resolved)
		if err != nil {
			return Inspection{}, err
		}
		inspection.Config = config
	}
	return inspection, nil
}

// Pull downloads a manifest, its GGUF model layer, and any config or license
// layers. The files are named model.gguf, config.json, LICENSE, and
// manifest.json. Large blobs stream through .part files and are renamed only
// after their size and SHA-256 digest have been verified.
func (c *Client) Pull(ctx context.Context, modelRef, destDir string, progress func(Progress)) error {
	if destDir == "" {
		return errors.New("ollama: destination directory is empty")
	}
	resolved, err := c.resolve(ctx, modelRef)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return fmt.Errorf("ollama: create destination directory: %w", err)
	}

	manifest := resolved.manifest
	if err := c.downloadLayer(ctx, resolved.session, resolved.ref, resolved.model, filepath.Join(destDir, "model.gguf"), progress); err != nil {
		return err
	}
	if manifest.Config.Digest != "" {
		if err := c.downloadLayer(ctx, resolved.session, resolved.ref, manifest.Config, filepath.Join(destDir, "config.json"), progress); err != nil {
			return err
		}
	}
	licenseNumber := 0
	for _, layer := range manifest.Layers {
		if layer.MediaType != licenseMediaType {
			continue
		}
		licenseNumber++
		name := "LICENSE"
		if licenseNumber > 1 {
			name = fmt.Sprintf("LICENSE-%d", licenseNumber)
		}
		if err := c.downloadLayer(ctx, resolved.session, resolved.ref, layer, filepath.Join(destDir, name), progress); err != nil {
			return err
		}
	}
	if err := atomicWrite(filepath.Join(destDir, "manifest.json"), resolved.manifestData); err != nil {
		return fmt.Errorf("ollama: save manifest: %w", err)
	}
	return nil
}

func (c *Client) resolve(ctx context.Context, modelRef string) (resolvedModel, error) {
	ref, err := parseReference(modelRef, c.BaseURL)
	if err != nil {
		return resolvedModel{}, err
	}
	s := &session{client: c, token: c.Token}
	manifestURL := ref.baseURL + "/v2/" + escapePath(ref.repository) + "/manifests/" + url.PathEscape(ref.tag)
	resp, err := s.get(ctx, manifestURL, "")
	if err != nil {
		return resolvedModel{}, fmt.Errorf("ollama: download manifest: %w", err)
	}
	manifestData, readErr := io.ReadAll(io.LimitReader(resp.Body, manifestLimit+1))
	closeErr := resp.Body.Close()
	if readErr != nil {
		return resolvedModel{}, fmt.Errorf("ollama: read manifest: %w", readErr)
	}
	if closeErr != nil {
		return resolvedModel{}, fmt.Errorf("ollama: close manifest response: %w", closeErr)
	}
	if len(manifestData) > manifestLimit {
		return resolvedModel{}, errors.New("ollama: manifest exceeds 16 MiB")
	}
	var manifest Manifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		return resolvedModel{}, fmt.Errorf("ollama: decode manifest: %w", err)
	}
	model, err := selectModel(manifest.Layers)
	if err != nil {
		return resolvedModel{}, err
	}
	if model.Size < 0 {
		return resolvedModel{}, fmt.Errorf("ollama: invalid negative size for %s", model.Digest)
	}
	if manifest.Config.Digest != "" && manifest.Config.MediaType != "" && manifest.Config.MediaType != configMediaType && manifest.Config.MediaType != "application/vnd.docker.container.image.v1+json" {
		return resolvedModel{}, fmt.Errorf("ollama: unsupported config media type %q", manifest.Config.MediaType)
	}
	return resolvedModel{ref: ref, session: s, manifest: manifest, manifestData: manifestData, model: model}, nil
}

func (c *Client) readConfig(ctx context.Context, resolved resolvedModel) (json.RawMessage, error) {
	layer := resolved.manifest.Config
	if layer.Size < 0 || layer.Size > manifestLimit {
		return nil, fmt.Errorf("ollama: invalid config size %d", layer.Size)
	}
	if _, err := parseDigest(layer.Digest); err != nil {
		return nil, err
	}
	blobURL := resolved.ref.baseURL + "/v2/" + escapePath(resolved.ref.repository) + "/blobs/" + layer.Digest
	resp, err := resolved.session.get(ctx, blobURL, "")
	if err != nil {
		return nil, fmt.Errorf("ollama: download config: %w", err)
	}
	data, readErr := io.ReadAll(io.LimitReader(resp.Body, manifestLimit+1))
	closeErr := resp.Body.Close()
	if readErr != nil {
		return nil, fmt.Errorf("ollama: read config: %w", readErr)
	}
	if closeErr != nil {
		return nil, fmt.Errorf("ollama: close config response: %w", closeErr)
	}
	if int64(len(data)) != layer.Size {
		return nil, fmt.Errorf("ollama: config size mismatch: got %d bytes, want %d", len(data), layer.Size)
	}
	digest := sha256.Sum256(data)
	if got := "sha256:" + hex.EncodeToString(digest[:]); got != strings.ToLower(layer.Digest) {
		return nil, fmt.Errorf("ollama: config digest mismatch: got %s", got)
	}
	if !json.Valid(data) {
		return nil, errors.New("ollama: config is not valid JSON")
	}
	return json.RawMessage(data), nil
}

func selectModel(layers []Layer) (Layer, error) {
	for _, layer := range layers {
		if layer.MediaType == modelMediaType {
			return layer, nil
		}
	}
	return Layer{}, errors.New("ollama: manifest has no GGUF model layer")
}

func (c *Client) downloadLayer(ctx context.Context, s *session, ref reference, layer Layer, destination string, progress func(Progress)) error {
	digest, err := parseDigest(layer.Digest)
	if err != nil {
		return err
	}
	if layer.Size < 0 {
		return fmt.Errorf("ollama: invalid negative size for %s", layer.Digest)
	}
	if valid, err := fileMatches(destination, layer.Size, digest); err != nil {
		return err
	} else if valid {
		report(progress, filepath.Base(destination), layer.Size, layer.Size)
		return nil
	}

	partPath := destination + ".part"
	part, err := os.OpenFile(partPath, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return fmt.Errorf("ollama: open partial download: %w", err)
	}
	offset, err := part.Seek(0, io.SeekEnd)
	if err != nil {
		part.Close()
		return fmt.Errorf("ollama: inspect partial download: %w", err)
	}
	if offset > layer.Size {
		if err := part.Truncate(0); err != nil {
			part.Close()
			return fmt.Errorf("ollama: truncate oversized partial download: %w", err)
		}
		offset = 0
	}
	if offset == layer.Size {
		if err := part.Close(); err != nil {
			return fmt.Errorf("ollama: close completed partial download: %w", err)
		}
		valid, err := fileMatches(partPath, layer.Size, digest)
		if err != nil {
			return err
		}
		if valid {
			report(progress, filepath.Base(destination), offset, layer.Size)
			return commitPart(partPath, destination)
		}
		part, err = os.OpenFile(partPath, os.O_RDWR|os.O_TRUNC, 0o644)
		if err != nil {
			return fmt.Errorf("ollama: restart invalid partial download: %w", err)
		}
		offset = 0
	}

	blobURL := ref.baseURL + "/v2/" + escapePath(ref.repository) + "/blobs/" + layer.Digest
	resp, err := s.get(ctx, blobURL, rangeHeader(offset))
	if err != nil {
		part.Close()
		var statusErr *StatusError
		if offset > 0 && errors.As(err, &statusErr) && statusErr.StatusCode == http.StatusRequestedRangeNotSatisfiable && partialComplete(statusErr.Header.Get("Content-Range"), offset) {
			valid, validateErr := fileMatches(partPath, layer.Size, digest)
			if validateErr != nil {
				return validateErr
			}
			if valid {
				report(progress, filepath.Base(destination), offset, layer.Size)
				return commitPart(partPath, destination)
			}
		}
		return fmt.Errorf("ollama: download %s: %w", layer.Digest, err)
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		if offset > 0 {
			if err := part.Truncate(0); err != nil {
				part.Close()
				return fmt.Errorf("ollama: restart unsupported resume: %w", err)
			}
		}
		offset = 0
	case http.StatusPartialContent:
		start, ok := contentRangeStart(resp.Header.Get("Content-Range"))
		if !ok || start != offset {
			part.Close()
			return fmt.Errorf("ollama: invalid Content-Range %q for resume at byte %d", resp.Header.Get("Content-Range"), offset)
		}
	default:
		part.Close()
		return fmt.Errorf("ollama: unexpected blob status %s", resp.Status)
	}
	if _, err := part.Seek(offset, io.SeekStart); err != nil {
		part.Close()
		return fmt.Errorf("ollama: seek partial download: %w", err)
	}

	hash := sha256.New()
	if offset > 0 {
		if _, err := part.Seek(0, io.SeekStart); err != nil {
			part.Close()
			return fmt.Errorf("ollama: rewind partial download: %w", err)
		}
		if _, err := io.CopyN(hash, part, offset); err != nil {
			part.Close()
			return fmt.Errorf("ollama: hash partial download: %w", err)
		}
	}
	report(progress, filepath.Base(destination), offset, layer.Size)
	downloaded, copyErr := copyWithProgress(part, resp.Body, hash, offset, layer.Size, func(n int64) {
		report(progress, filepath.Base(destination), n, layer.Size)
	})
	if copyErr != nil {
		part.Close()
		return fmt.Errorf("ollama: download %s: %w", layer.Digest, copyErr)
	}
	if downloaded != layer.Size {
		part.Close()
		return fmt.Errorf("ollama: size mismatch for %s: got %d bytes, want %d", layer.Digest, downloaded, layer.Size)
	}
	if got := hex.EncodeToString(hash.Sum(nil)); got != digest {
		part.Close()
		return fmt.Errorf("ollama: digest mismatch for %s: got sha256:%s", layer.Digest, got)
	}
	if err := part.Sync(); err != nil {
		part.Close()
		return fmt.Errorf("ollama: sync download: %w", err)
	}
	if err := part.Close(); err != nil {
		return fmt.Errorf("ollama: close download: %w", err)
	}
	if err := commitPart(partPath, destination); err != nil {
		return err
	}
	return nil
}

func copyWithProgress(dst io.Writer, src io.Reader, hash io.Writer, downloaded, total int64, progress func(int64)) (int64, error) {
	buffer := make([]byte, 128*1024)
	for {
		n, readErr := src.Read(buffer)
		if n > 0 {
			if total >= 0 && downloaded+int64(n) > total {
				return downloaded, errors.New("blob exceeds manifest size")
			}
			written, writeErr := dst.Write(buffer[:n])
			if written > 0 {
				if _, err := hash.Write(buffer[:written]); err != nil {
					return downloaded, err
				}
				downloaded += int64(written)
				progress(downloaded)
			}
			if writeErr != nil {
				return downloaded, writeErr
			}
			if written != n {
				return downloaded, io.ErrShortWrite
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				return downloaded, nil
			}
			return downloaded, readErr
		}
	}
}

func report(progress func(Progress), filename string, downloaded, total int64) {
	if progress != nil {
		progress(Progress{Filename: filename, Downloaded: downloaded, Total: total})
	}
}

func commitPart(partPath, destination string) error {
	if err := os.Rename(partPath, destination); err != nil {
		return fmt.Errorf("ollama: commit download: %w", err)
	}
	return nil
}

func fileMatches(filename string, size int64, digest string) (bool, error) {
	file, err := os.Open(filename)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("ollama: open cached file: %w", err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return false, fmt.Errorf("ollama: inspect cached file: %w", err)
	}
	if !info.Mode().IsRegular() || info.Size() != size {
		return false, nil
	}
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return false, fmt.Errorf("ollama: hash cached file: %w", err)
	}
	return hex.EncodeToString(hash.Sum(nil)) == digest, nil
}

func atomicWrite(destination string, data []byte) error {
	file, err := os.CreateTemp(filepath.Dir(destination), ".manifest-*")
	if err != nil {
		return err
	}
	temporary := file.Name()
	defer os.Remove(temporary)
	if err := file.Chmod(0o644); err != nil {
		file.Close()
		return err
	}
	if _, err := file.Write(data); err != nil {
		file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return os.Rename(temporary, destination)
}

type session struct {
	client *Client
	token  string
}

func (s *session) get(ctx context.Context, endpoint, byteRange string) (*http.Response, error) {
	resp, err := s.do(ctx, endpoint, byteRange)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusUnauthorized {
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return resp, nil
		}
		return nil, responseError(resp)
	}

	challenge := resp.Header.Get("WWW-Authenticate")
	resp.Body.Close()
	realm, service, scope, ok := bearerChallenge(challenge)
	if !ok {
		return nil, fmt.Errorf("ollama: unsupported authentication challenge %q", challenge)
	}
	token, err := s.fetchToken(ctx, realm, service, scope)
	if err != nil {
		return nil, err
	}
	s.token = token
	resp, err = s.do(ctx, endpoint, byteRange)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return resp, nil
	}
	return nil, responseError(resp)
}

func (s *session) do(ctx context.Context, endpoint, byteRange string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("ollama: create request: %w", err)
	}
	if s.token != "" {
		req.Header.Set("Authorization", "Bearer "+s.token)
	}
	if byteRange != "" {
		req.Header.Set("Range", byteRange)
	}
	req.Header.Set("Accept", strings.Join([]string{
		"application/vnd.oci.image.manifest.v1+json",
		"application/vnd.docker.distribution.manifest.v2+json",
		"application/octet-stream",
	}, ", "))
	resp, err := s.client.httpClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama: GET %s: %w", endpoint, err)
	}
	return resp, nil
}

func (s *session) fetchToken(ctx context.Context, realm, service, scope string) (string, error) {
	tokenURL, err := url.Parse(realm)
	if err != nil || (tokenURL.Scheme != "http" && tokenURL.Scheme != "https") || tokenURL.Host == "" {
		return "", fmt.Errorf("ollama: invalid Bearer realm %q", realm)
	}
	query := tokenURL.Query()
	if service != "" {
		query.Set("service", service)
	}
	if scope != "" {
		query.Set("scope", scope)
	}
	tokenURL.RawQuery = query.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, tokenURL.String(), nil)
	if err != nil {
		return "", fmt.Errorf("ollama: create token request: %w", err)
	}
	resp, err := s.client.httpClient().Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama: request Bearer token: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", responseError(resp)
	}
	defer resp.Body.Close()
	var tokenResponse struct {
		Token       string `json:"token"`
		AccessToken string `json:"access_token"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&tokenResponse); err != nil {
		return "", fmt.Errorf("ollama: decode Bearer token: %w", err)
	}
	token := tokenResponse.Token
	if token == "" {
		token = tokenResponse.AccessToken
	}
	if token == "" {
		return "", errors.New("ollama: Bearer token response is empty")
	}
	return token, nil
}

func bearerChallenge(header string) (realm, service, scope string, ok bool) {
	const prefix = "bearer "
	if len(header) < len(prefix) || !strings.EqualFold(header[:len(prefix)], prefix) {
		return "", "", "", false
	}
	parameters := make(map[string]string)
	for _, field := range splitChallenge(header[len(prefix):]) {
		key, value, found := strings.Cut(field, "=")
		if !found {
			continue
		}
		value = strings.TrimSpace(value)
		if len(value) >= 2 && value[0] == '"' && value[len(value)-1] == '"' {
			if unquoted, err := strconv.Unquote(value); err == nil {
				value = unquoted
			}
		}
		parameters[strings.ToLower(strings.TrimSpace(key))] = value
	}
	realm = parameters["realm"]
	return realm, parameters["service"], parameters["scope"], realm != ""
}

func splitChallenge(value string) []string {
	var fields []string
	start := 0
	quoted, escaped := false, false
	for i, r := range value {
		switch {
		case escaped:
			escaped = false
		case quoted && r == '\\':
			escaped = true
		case r == '"':
			quoted = !quoted
		case r == ',' && !quoted:
			fields = append(fields, strings.TrimSpace(value[start:i]))
			start = i + 1
		}
	}
	fields = append(fields, strings.TrimSpace(value[start:]))
	return fields
}

func responseError(resp *http.Response) error {
	defer resp.Body.Close()
	body, readErr := io.ReadAll(io.LimitReader(resp.Body, 4096))
	if readErr != nil {
		return fmt.Errorf("ollama: GET %s: %s (read error response: %v)", resp.Request.URL, resp.Status, readErr)
	}
	return &StatusError{
		URL:        resp.Request.URL.String(),
		StatusCode: resp.StatusCode,
		Status:     resp.Status,
		Header:     resp.Header.Clone(),
		Body:       strings.TrimSpace(string(body)),
	}
}

type reference struct {
	baseURL    string
	repository string
	tag        string
}

func parseReference(modelRef, baseOverride string) (reference, error) {
	modelRef = strings.TrimSpace(modelRef)
	if modelRef == "" {
		return reference{}, errors.New("ollama: model reference is empty")
	}
	baseURL := baseOverride
	if baseURL == "" {
		baseURL = defaultRegistry
	}
	name := modelRef
	if strings.Contains(modelRef, "://") {
		parsed, err := url.Parse(modelRef)
		if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.RawQuery != "" || parsed.Fragment != "" {
			return reference{}, fmt.Errorf("ollama: invalid model reference %q", modelRef)
		}
		baseURL = parsed.Scheme + "://" + parsed.Host
		name = strings.TrimPrefix(parsed.Path, "/")
		if strings.HasPrefix(name, "v2/") {
			parts := strings.Split(name, "/")
			if len(parts) < 5 || parts[len(parts)-2] != "manifests" {
				return reference{}, fmt.Errorf("ollama: invalid registry URL %q", modelRef)
			}
			name = strings.Join(parts[1:len(parts)-2], "/") + ":" + parts[len(parts)-1]
		}
	} else {
		parts := strings.Split(modelRef, "/")
		if len(parts) > 1 && parts[0] != "." && parts[0] != ".." && (strings.Contains(parts[0], ".") || parts[0] == "localhost" || strings.HasPrefix(parts[0], "[") || strings.Contains(parts[0], ":")) {
			baseURL = "https://" + parts[0]
			name = strings.Join(parts[1:], "/")
		}
	}
	base, err := url.Parse(baseURL)
	if err != nil || base.Host == "" || (base.Scheme != "http" && base.Scheme != "https") || base.RawQuery != "" || base.Fragment != "" {
		return reference{}, fmt.Errorf("ollama: invalid registry base URL %q", baseURL)
	}
	baseURL = strings.TrimRight(baseURL, "/")

	if strings.Contains(name, "@") {
		return reference{}, fmt.Errorf("ollama: digest references are not supported: %q", modelRef)
	}
	lastSlash := strings.LastIndex(name, "/")
	lastColon := strings.LastIndex(name, ":")
	tag := "latest"
	if lastColon > lastSlash {
		tag = name[lastColon+1:]
		name = name[:lastColon]
	}
	if !strings.Contains(name, "/") {
		name = "library/" + name
	}
	parts := strings.Split(name, "/")
	if len(parts) != 2 || tag == "" {
		return reference{}, fmt.Errorf("ollama: invalid model reference %q", modelRef)
	}
	for _, part := range append(parts, tag) {
		if part == "" || part == "." || part == ".." || strings.ContainsAny(part, "\\\x00") {
			return reference{}, fmt.Errorf("ollama: invalid model reference %q", modelRef)
		}
		for _, r := range part {
			if r < 0x20 || r == 0x7f {
				return reference{}, fmt.Errorf("ollama: invalid model reference %q", modelRef)
			}
		}
	}
	return reference{baseURL: baseURL, repository: name, tag: tag}, nil
}

func parseDigest(value string) (string, error) {
	algorithm, encoded, found := strings.Cut(value, ":")
	if !found || algorithm != "sha256" || len(encoded) != sha256.Size*2 {
		return "", fmt.Errorf("ollama: unsupported or invalid digest %q", value)
	}
	decoded, err := hex.DecodeString(encoded)
	if err != nil || len(decoded) != sha256.Size {
		return "", fmt.Errorf("ollama: invalid digest %q", value)
	}
	return strings.ToLower(encoded), nil
}

func rangeHeader(offset int64) string {
	if offset == 0 {
		return ""
	}
	return fmt.Sprintf("bytes=%d-", offset)
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

func partialComplete(value string, size int64) bool {
	total, err := strconv.ParseInt(strings.TrimPrefix(value, "bytes */"), 10, 64)
	return strings.HasPrefix(value, "bytes */") && err == nil && total == size
}

func escapePath(value string) string {
	parts := strings.Split(value, "/")
	for i, part := range parts {
		parts[i] = url.PathEscape(part)
	}
	return strings.Join(parts, "/")
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

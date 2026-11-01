package main

import (
	"context"
	"path/filepath"
	"testing"

	"fyne.io/fyne/v2/test"
)

func TestSavedModelDirUsesPreferences(t *testing.T) {
	application := test.NewApp()
	defer application.Quit()
	preferences := application.Preferences()

	fallback := savedModelDir(preferences)
	cacheDir, err := defaultModelDir()
	if err != nil {
		t.Fatal(err)
	}
	if fallback != cacheDir {
		t.Fatalf("savedModelDir() = %q; want default %q", fallback, cacheDir)
	}

	custom := filepath.Join(t.TempDir(), "models")
	preferences.SetString(modelDirPreference, custom)
	if got := savedModelDir(preferences); got != custom {
		t.Fatalf("savedModelDir() = %q; want saved %q", got, custom)
	}
}

func TestResolveModelDirKeepsLocalModelPath(t *testing.T) {
	modelDir := t.TempDir()
	got, err := resolveModelDir(context.Background(), modelDir, filepath.Join(t.TempDir(), "unused"))
	if err != nil {
		t.Fatal(err)
	}
	if got != modelDir {
		t.Fatalf("resolveModelDir() = %q; want %q", got, modelDir)
	}
}

func TestDiskFreeBytesForExistingAndFutureDirectory(t *testing.T) {
	futureDir := filepath.Join(t.TempDir(), "airllm", "models")
	free, err := diskFreeBytes(futureDir)
	if err != nil {
		t.Fatal(err)
	}
	if free == 0 {
		t.Fatal("diskFreeBytes() returned zero for an available test disk")
	}
}

func TestFormatByteSize(t *testing.T) {
	if got := formatByteSize(512 << 20); got != "512.0 МиБ" {
		t.Fatalf("formatByteSize() = %q; want MiB", got)
	}
	if got := formatByteSize(12 << 30); got != "12.0 ГиБ" {
		t.Fatalf("formatByteSize() = %q; want GiB", got)
	}
}

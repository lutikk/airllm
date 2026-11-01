//go:build !windows

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"syscall"
)

func diskFreeBytes(path string) (uint64, error) {
	current, err := filepath.Abs(path)
	if err != nil {
		return 0, fmt.Errorf("определение абсолютного пути: %w", err)
	}
	for {
		if info, statErr := os.Stat(current); statErr == nil {
			if !info.IsDir() {
				current = filepath.Dir(current)
			}
			break
		} else if !os.IsNotExist(statErr) {
			return 0, fmt.Errorf("проверка каталога: %w", statErr)
		}
		parent := filepath.Dir(current)
		if parent == current {
			return 0, fmt.Errorf("каталог не существует: %s", path)
		}
		current = parent
	}

	var stats syscall.Statfs_t
	if err := syscall.Statfs(current, &stats); err != nil {
		return 0, fmt.Errorf("определение свободного места: %w", err)
	}
	return stats.Bavail * uint64(stats.Bsize), nil
}

//go:build windows

package main

import (
	"fmt"
	"path/filepath"

	"golang.org/x/sys/windows"
)

func diskFreeBytes(path string) (uint64, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return 0, fmt.Errorf("определение абсолютного пути: %w", err)
	}
	root := filepath.VolumeName(absolute) + string(filepath.Separator)
	var available uint64
	if err := windows.GetDiskFreeSpaceEx(windows.StringToUTF16Ptr(root), &available, nil, nil); err != nil {
		return 0, fmt.Errorf("определение свободного места: %w", err)
	}
	return available, nil
}

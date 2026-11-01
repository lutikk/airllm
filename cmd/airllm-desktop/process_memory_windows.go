//go:build windows

package main

import (
	"fmt"
	"syscall"
	"unsafe"
)

type processMemoryCounters struct {
	cb                         uint32
	pageFaultCount             uint32
	peakWorkingSetSize         uintptr
	workingSetSize             uintptr
	quotaPeakPagedPoolUsage    uintptr
	quotaPagedPoolUsage        uintptr
	quotaPeakNonPagedPoolUsage uintptr
	quotaNonPagedPoolUsage     uintptr
	pagefileUsage              uintptr
	peakPagefileUsage          uintptr
}

func processRSS() (uint64, error) {
	psapi := syscall.NewLazyDLL("psapi.dll")
	getProcessMemoryInfo := psapi.NewProc("GetProcessMemoryInfo")
	handle, err := syscall.GetCurrentProcess()
	if err != nil {
		return 0, err
	}
	var counters processMemoryCounters
	counters.cb = uint32(unsafe.Sizeof(counters))
	result, _, callErr := getProcessMemoryInfo.Call(uintptr(handle), uintptr(unsafe.Pointer(&counters)), uintptr(counters.cb))
	if result == 0 {
		return 0, fmt.Errorf("GetProcessMemoryInfo: %w", callErr)
	}
	return uint64(counters.workingSetSize), nil
}

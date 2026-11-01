//go:build darwin

package main

import "syscall"

// processRSS returns the peak resident size of the whole application on macOS.
func processRSS() (uint64, error) {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return 0, err
	}
	return uint64(usage.Maxrss), nil
}

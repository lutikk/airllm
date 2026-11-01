package hardware

import (
	"runtime"
	"testing"
)

func TestProbeAlwaysReportsCPU(t *testing.T) {
	snapshot := Probe()
	if !snapshot.Capabilities.CPU {
		t.Fatal("CPU backend must always be available")
	}
	if snapshot.CPU.Cores <= 0 {
		t.Fatalf("CPU cores = %d, want a positive value", snapshot.CPU.Cores)
	}
	if snapshot.CPU.Arch != runtime.GOARCH {
		t.Fatalf("CPU architecture = %q, want %q", snapshot.CPU.Arch, runtime.GOARCH)
	}
	if snapshot.CPU.OS != runtime.GOOS {
		t.Fatalf("operating system = %q, want %q", snapshot.CPU.OS, runtime.GOOS)
	}
}

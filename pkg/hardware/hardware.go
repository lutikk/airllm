// Package hardware discovers the compute and memory resources available to AirLLM.
package hardware

import (
	"context"
	"os/exec"
	"runtime"
	"time"
)

const probeTimeout = 2 * time.Second

// CPU describes the host processor and operating system.
type CPU struct {
	Model    string `json:"model"`
	Cores    int    `json:"cores"`
	Arch     string `json:"arch"`
	OS       string `json:"os"`
	RAMBytes uint64 `json:"ram_bytes"`
}

// GPU describes a graphics processor. VRAMBytes is zero when the driver does
// not report a fixed amount, as is common for unified-memory GPUs.
type GPU struct {
	Name      string `json:"name"`
	VRAMBytes uint64 `json:"vram_bytes,omitempty"`
}

// Capabilities describes which inference backends the host can support.
type Capabilities struct {
	Metal bool `json:"metal"`
	CUDA  bool `json:"cuda"`
	CPU   bool `json:"cpu"`
}

// Snapshot is a best-effort view of the current host. Fields that cannot be
// discovered are left at their zero values.
type Snapshot struct {
	CPU          CPU          `json:"cpu"`
	GPU          *GPU         `json:"gpu,omitempty"`
	Capabilities Capabilities `json:"capabilities"`
}

// Probe inspects the local machine without requiring privileges. Missing
// files, drivers, and system utilities are expected and do not cause a panic.
func Probe() Snapshot {
	return probePlatform()
}

func baseSnapshot() Snapshot {
	return Snapshot{
		CPU: CPU{
			Cores: runtime.NumCPU(),
			Arch:  runtime.GOARCH,
			OS:    runtime.GOOS,
		},
		Capabilities: Capabilities{CPU: true},
	}
}

func commandOutput(name string, args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), probeTimeout)
	defer cancel()
	return exec.CommandContext(ctx, name, args...).Output()
}

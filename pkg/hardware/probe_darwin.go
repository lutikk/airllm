//go:build darwin

package hardware

import (
	"encoding/json"
	"strconv"
	"strings"
)

func probePlatform() Snapshot {
	snapshot := baseSnapshot()
	snapshot.Capabilities.Metal = true

	if model, err := commandOutput("sysctl", "-n", "machdep.cpu.brand_string"); err == nil {
		snapshot.CPU.Model = strings.TrimSpace(string(model))
	}
	if cores, err := commandOutput("sysctl", "-n", "hw.ncpu"); err == nil {
		if value, err := strconv.Atoi(strings.TrimSpace(string(cores))); err == nil && value > 0 {
			snapshot.CPU.Cores = value
		}
	}
	if memory, err := commandOutput("sysctl", "-n", "hw.memsize"); err == nil {
		snapshot.CPU.RAMBytes, _ = strconv.ParseUint(strings.TrimSpace(string(memory)), 10, 64)
	}

	if gpu := probeSystemProfiler(); gpu != nil {
		snapshot.GPU = gpu
	}
	return snapshot
}

type displayDevice struct {
	Name     string          `json:"_name"`
	PCIModel string          `json:"sppci_model"`
	VRAM     string          `json:"spdisplays_vram"`
	Displays []displayDevice `json:"spdisplays_ndrvs"`
}

func probeSystemProfiler() *GPU {
	output, err := commandOutput("system_profiler", "-json", "SPDisplaysDataType")
	if err != nil {
		return nil
	}
	gpu, _ := parseSystemProfiler(output)
	return gpu
}

func parseSystemProfiler(output []byte) (*GPU, error) {
	var devices []displayDevice
	if err := json.Unmarshal(output, &devices); err != nil {
		return nil, err
	}
	for _, device := range devices {
		if gpu := gpuFromDisplay(device); gpu != nil {
			return gpu, nil
		}
		for _, child := range device.Displays {
			if gpu := gpuFromDisplay(child); gpu != nil {
				return gpu, nil
			}
		}
	}
	return nil, nil
}

func gpuFromDisplay(device displayDevice) *GPU {
	name := strings.TrimSpace(device.Name)
	if name == "" {
		name = strings.TrimSpace(device.PCIModel)
	}
	if name == "" {
		return nil
	}
	return &GPU{Name: name, VRAMBytes: parseVRAM(device.VRAM)}
}

func parseVRAM(value string) uint64 {
	fields := strings.Fields(value)
	if len(fields) == 0 {
		return 0
	}
	amount, err := strconv.ParseFloat(fields[0], 64)
	if err != nil || amount <= 0 {
		return 0
	}
	multiplier := float64(1)
	if len(fields) > 1 {
		switch strings.ToUpper(fields[1]) {
		case "KB":
			multiplier = 1e3
		case "MB":
			multiplier = 1e6
		case "GB":
			multiplier = 1e9
		case "TB":
			multiplier = 1e12
		case "KIB":
			multiplier = 1 << 10
		case "MIB":
			multiplier = 1 << 20
		case "GIB":
			multiplier = 1 << 30
		case "TIB":
			multiplier = 1 << 40
		}
	}
	return uint64(amount * multiplier)
}

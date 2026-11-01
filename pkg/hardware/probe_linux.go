//go:build linux

package hardware

import (
	"bufio"
	"bytes"
	"strconv"
	"strings"
)

func probePlatform() Snapshot {
	snapshot := baseSnapshot()
	if cpuinfo, err := readFile("/proc/cpuinfo"); err == nil {
		snapshot.CPU.Model = parseCPUModel(cpuinfo)
	}
	if meminfo, err := readFile("/proc/meminfo"); err == nil {
		snapshot.CPU.RAMBytes = parseMemTotal(meminfo)
	}
	if output, err := commandOutput("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"); err == nil {
		if gpu, ok := parseNVIDIAOutput(output); ok {
			snapshot.GPU = gpu
			snapshot.Capabilities.CUDA = true
		}
	}
	return snapshot
}

func parseCPUModel(input []byte) string {
	scanner := bufio.NewScanner(bytes.NewReader(input))
	for scanner.Scan() {
		key, value, ok := strings.Cut(scanner.Text(), ":")
		if !ok {
			continue
		}
		switch strings.TrimSpace(key) {
		case "model name", "Hardware", "Processor":
			if model := strings.TrimSpace(value); model != "" {
				return model
			}
		}
	}
	return ""
}

func parseMemTotal(input []byte) uint64 {
	scanner := bufio.NewScanner(bytes.NewReader(input))
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 || fields[0] != "MemTotal:" {
			continue
		}
		kibibytes, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil || kibibytes > ^uint64(0)/1024 {
			return 0
		}
		return kibibytes * 1024
	}
	return 0
}

func parseNVIDIAOutput(output []byte) (*GPU, bool) {
	line, _, _ := strings.Cut(string(output), "\n")
	name, memory, ok := strings.Cut(strings.TrimSpace(line), ",")
	if !ok {
		return nil, false
	}
	name = strings.TrimSpace(name)
	mib, err := strconv.ParseFloat(strings.TrimSpace(memory), 64)
	if name == "" || err != nil || mib <= 0 || mib > float64(^uint64(0))/(1<<20) {
		return nil, false
	}
	return &GPU{Name: name, VRAMBytes: uint64(mib * (1 << 20))}, true
}

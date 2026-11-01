// Package memory parses memory quantities and checks inference memory budgets.
package memory

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"strings"
)

var byteMultipliers = map[string]uint64{
	"B":   1,
	"KB":  1e3,
	"MB":  1e6,
	"GB":  1e9,
	"TB":  1e12,
	"KIB": 1 << 10,
	"MIB": 1 << 20,
	"GIB": 1 << 30,
	"TIB": 1 << 40,
}

// ParseBytes converts a human-readable quantity into bytes. Units are
// case-insensitive; SI units use powers of 1000 and IEC units powers of 1024.
// A missing unit means bytes. Fractional values are accepted when they denote
// a whole number of bytes.
func ParseBytes(input string) (uint64, error) {
	input = strings.TrimSpace(input)
	if input == "" {
		return 0, errors.New("memory quantity is empty")
	}

	numberEnd := 0
	for numberEnd < len(input) {
		char := input[numberEnd]
		if (char < '0' || char > '9') && char != '.' && char != '+' && char != '-' {
			break
		}
		numberEnd++
	}
	number := input[:numberEnd]
	unit := strings.TrimSpace(input[numberEnd:])
	if unit == "" {
		unit = "B"
	}
	multiplier, ok := byteMultipliers[strings.ToUpper(unit)]
	if !ok {
		return 0, fmt.Errorf("unknown memory unit %q", unit)
	}

	value, ok := new(big.Rat).SetString(number)
	if !ok || value.Sign() < 0 {
		return 0, fmt.Errorf("invalid memory quantity %q", number)
	}
	value.Mul(value, new(big.Rat).SetInt(new(big.Int).SetUint64(multiplier)))
	if !value.IsInt() || !value.Num().IsUint64() {
		return 0, fmt.Errorf("memory quantity %q is not a whole number of bytes or overflows uint64", input)
	}
	return value.Num().Uint64(), nil
}

// Budget contains the memory available on the host and compute device.
type Budget struct {
	HostBytes   uint64
	DeviceBytes uint64
}

// Estimate contains the memory required by an inference configuration.
type Estimate struct {
	ResidentWeights uint64
	MaxLayer        uint64
	KVCache         uint64
	Workspace       uint64
	Prefetch        bool
}

// Result describes the calculated requirements and any remediation advice.
type Result struct {
	HostRequiredBytes   uint64
	DeviceRequiredBytes uint64
	HostFits            bool
	DeviceFits          bool
	Fits                bool
	Explanation         []string
	Suggestions         []string
}

// Plan compares an estimate with a budget. Prefetch reserves a second layer
// buffer on the device so the next layer can be transferred during execution.
func Plan(budget Budget, estimate Estimate) Result {
	result := Result{
		HostRequiredBytes: estimate.ResidentWeights,
		DeviceRequiredBytes: add(
			add(estimate.MaxLayer, estimate.KVCache),
			estimate.Workspace,
		),
	}
	if estimate.Prefetch {
		result.DeviceRequiredBytes = add(result.DeviceRequiredBytes, estimate.MaxLayer)
	}
	result.HostFits = result.HostRequiredBytes <= budget.HostBytes
	result.DeviceFits = result.DeviceRequiredBytes <= budget.DeviceBytes
	result.Fits = result.HostFits && result.DeviceFits

	if result.HostFits {
		result.Explanation = append(result.Explanation, fmt.Sprintf(
			"host memory fits: %d bytes required of %d available",
			result.HostRequiredBytes, budget.HostBytes,
		))
	} else {
		result.Explanation = append(result.Explanation, fmt.Sprintf(
			"host memory is insufficient: %d bytes required of %d available",
			result.HostRequiredBytes, budget.HostBytes,
		))
		result.Suggestions = append(result.Suggestions, "reduce resident weights or increase the host memory budget")
	}

	if result.DeviceFits {
		result.Explanation = append(result.Explanation, fmt.Sprintf(
			"device memory fits: %d bytes required of %d available",
			result.DeviceRequiredBytes, budget.DeviceBytes,
		))
	} else {
		result.Explanation = append(result.Explanation, fmt.Sprintf(
			"device memory is insufficient: %d bytes required of %d available",
			result.DeviceRequiredBytes, budget.DeviceBytes,
		))
		if estimate.Prefetch {
			result.Suggestions = append(result.Suggestions, "disable prefetch to release one layer buffer")
		}
		if estimate.KVCache > 0 {
			result.Suggestions = append(result.Suggestions, "reduce context length to shrink the KV cache")
		}
		result.Suggestions = append(result.Suggestions, "reduce max layer or workspace memory")
	}
	return result
}

// Validate returns a detailed error when the estimate does not fit the budget.
func Validate(budget Budget, estimate Estimate) error {
	return Plan(budget, estimate).Validate()
}

// Validate returns a detailed error when the plan does not fit.
func (result Result) Validate() error {
	if result.Fits {
		return nil
	}
	parts := append([]string{}, result.Explanation...)
	parts = append(parts, result.Suggestions...)
	return errors.New(strings.Join(parts, "; "))
}

// LlamaKVCacheBytes returns 2 * layers * contextLength * kvHeads * headDim *
// bytesPerElement. The leading two accounts for separate key and value caches.
// Overflow saturates at MaxUint64 rather than wrapping to a deceptively small
// allocation.
func LlamaKVCacheBytes(layers, contextLength, kvHeads, headDim, bytesPerElement uint64) uint64 {
	result := uint64(2)
	for _, factor := range []uint64{layers, contextLength, kvHeads, headDim, bytesPerElement} {
		result = multiply(result, factor)
	}
	return result
}

func add(left, right uint64) uint64 {
	if math.MaxUint64-left < right {
		return math.MaxUint64
	}
	return left + right
}

func multiply(left, right uint64) uint64 {
	if left != 0 && right > math.MaxUint64/left {
		return math.MaxUint64
	}
	return left * right
}

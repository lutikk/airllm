package cpuref

import (
	"math"
	"testing"
)

func TestLinearGolden(t *testing.T) {
	input := []float32{1, -2, 0.5}
	weight := []float32{
		0.25, 0.5, -1,
		-2, 1, 0.75,
	}
	got, err := Linear(input, weight, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, got, []float32{-1.25, -3.625})
}

func TestRMSNormGolden(t *testing.T) {
	got, err := RMSNorm(
		[]float32{1, -2, 2},
		[]float32{0.5, 1.5, -1},
		1e-5,
	)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, got, []float32{0.28867465, -1.7320479, -1.1546986})
}

func TestSiLUGolden(t *testing.T) {
	got := SiLU([]float32{-2, -0.5, 0, 0.5, 2})
	assertClose(t, got, []float32{-0.23840584, -0.18877034, 0, 0.31122967, 1.7615942})
}

func TestMultiplyGolden(t *testing.T) {
	got, err := Multiply([]float32{1.5, -2, 0.25}, []float32{-4, 0.5, 8})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, got, []float32{-6, -1, 2})
}

func TestSiLUGateGolden(t *testing.T) {
	got, err := SiLUGate([]float32{0.5, -1, 2}, []float32{2, 3, -0.5})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, got, []float32{0.62245935, -0.80682427, -0.8807971})
}

func TestPrimitiveValidation(t *testing.T) {
	if _, err := Linear([]float32{1}, []float32{1}, 2, 1); err == nil {
		t.Error("Linear accepted an incorrectly sized input")
	}
	if _, err := RMSNorm([]float32{1}, []float32{1}, 0); err == nil {
		t.Error("RMSNorm accepted zero epsilon")
	}
	if _, err := Multiply([]float32{1}, nil); err == nil {
		t.Error("Multiply accepted mismatched lengths")
	}
	if _, err := SiLUGate([]float32{1}, nil); err == nil {
		t.Error("SiLUGate accepted mismatched lengths")
	}
}

func assertClose(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		difference := math.Abs(float64(got[i] - want[i]))
		if difference > 2e-6 {
			t.Errorf("element %d = %.9g, want %.9g (difference %.3g)", i, got[i], want[i], difference)
		}
	}
}

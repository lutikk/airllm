package cpuref

import "testing"

func TestRoPEGoldenWithPositionOffset(t *testing.T) {
	input := []float32{
		1, 2, 3, 4,
		-1, 0.5, 2, -2,
	}
	got, err := RoPE(input, 4, 2, 10000)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{
		-3.1440392, 1.9196054, -0.3391431, 4.0391974,
		0.7077525, 0.559766, -2.121105, -1.9841024,
	}
	assertClose(t, got, want)
}

func TestRoPEZeroOffsetLeavesZeroVectorUnchanged(t *testing.T) {
	got, err := RoPE([]float32{0, 0, 0, 0}, 4, 0, 10000)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, got, []float32{0, 0, 0, 0})
}

func TestRoPEValidation(t *testing.T) {
	if _, err := RoPE([]float32{1, 2, 3}, 3, 0, 10000); err == nil {
		t.Error("RoPE accepted an odd head dimension")
	}
	if _, err := RoPE([]float32{1, 2}, 2, -1, 10000); err == nil {
		t.Error("RoPE accepted a negative position offset")
	}
}

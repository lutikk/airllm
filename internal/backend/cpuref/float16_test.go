package cpuref

import (
	"math"
	"testing"
)

func TestDecodeF16LEGolden(t *testing.T) {
	// 0, -0, 1, -2, 0.5, smallest positive subnormal, +Inf.
	data := []byte{
		0x00, 0x00,
		0x00, 0x80,
		0x00, 0x3c,
		0x00, 0xc0,
		0x00, 0x38,
		0x01, 0x00,
		0x00, 0x7c,
	}
	got, err := DecodeF16LE(data, 7)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 0, 1, -2, 0.5, 0x1p-24, float32(math.Inf(1))}
	assertClose(t, got, want)
	if !math.Signbit(float64(got[1])) {
		t.Error("F16 negative zero lost its sign")
	}
	if !math.IsInf(float64(got[6]), 1) {
		t.Errorf("F16 infinity = %v", got[6])
	}
}

func TestDecodeBF16LEGolden(t *testing.T) {
	// 1, -2, 0.5, 3.1415927 rounded to BF16, -Inf.
	data := []byte{
		0x80, 0x3f,
		0x00, 0xc0,
		0x00, 0x3f,
		0x49, 0x40,
		0x80, 0xff,
	}
	got, err := DecodeBF16LE(data, 1, 5)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, got[:4], []float32{1, -2, 0.5, 3.140625})
	if !math.IsInf(float64(got[4]), -1) {
		t.Errorf("BF16 infinity = %v", got[4])
	}
}

func TestFloat16ShapeValidation(t *testing.T) {
	if _, err := DecodeF16LE([]byte{0, 0}, 2); err == nil {
		t.Error("DecodeF16LE accepted a shape/data length mismatch")
	}
	if _, err := DecodeBF16LE([]byte{0, 0}, 2); err == nil {
		t.Error("DecodeBF16LE accepted a shape/data length mismatch")
	}
	if _, err := DecodeF16LE(nil, 0); err == nil {
		t.Error("DecodeF16LE accepted a zero dimension")
	}
	if _, err := DecodeBF16LE(nil); err == nil {
		t.Error("DecodeBF16LE accepted an empty shape")
	}
	if _, err := DecodeF16LE(nil, int(^uint(0)>>1), 2); err == nil {
		t.Error("DecodeF16LE accepted an overflowing shape")
	}
}

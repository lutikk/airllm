package cpuref

import "testing"

func TestDecodeF32LEGolden(t *testing.T) {
	data := []byte{
		0x00, 0x00, 0x80, 0x3f,
		0x00, 0x00, 0x00, 0xc0,
		0x00, 0x00, 0x00, 0x3f,
	}
	got, err := DecodeF32LE(data, 3)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, got, []float32{1, -2, 0.5})
	if _, err := DecodeF32LE(data, 2); err == nil {
		t.Error("DecodeF32LE accepted a shape/data length mismatch")
	}
}

package gguf

import (
	"bytes"
	"context"
	"encoding/binary"
	"testing"
)

func TestKQuantGoldenBlocksAndShape(t *testing.T) {
	tests := []struct {
		name       string
		tensorType TensorType
		block      []byte
		checks     map[int]float32
	}{
		{
			name: "Q4_K", tensorType: Q4_K, block: goldenQ45KBlock(false),
			checks: map[int]float32{0: 0.5, 32: 6.5},
		},
		{
			name: "Q5_K", tensorType: Q5_K, block: goldenQ45KBlock(true),
			checks: map[int]float32{0: 32.5, 32: 6.5},
		},
		{
			name: "Q6_K", tensorType: Q6_K, block: goldenQ6KBlock(),
			checks: map[int]float32{0: -31, 32: -15 * 3, 64: 2 * 5, 96: 18 * 7, 128: -31 * 9, 160: -16 * 11, 192: 2 * 13, 224: 16 * 15},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			info := TensorInfo{Name: "output_norm.weight", Type: test.tensorType, Shape: []uint64{256}, ByteSize: int64(len(test.block))}
			var converted bytes.Buffer
			if err := writeF16Tensor(context.Background(), &converted, info, test.block); err != nil {
				t.Fatal(err)
			}
			if converted.Len() != 256*2 {
				t.Fatalf("decoded byte length = %d, want %d", converted.Len(), 256*2)
			}
			mapped, _, err := mapTensors([]TensorInfo{info})
			if err != nil {
				t.Fatal(err)
			}
			if len(mapped) != 1 || !sameShape(mapped[0].shape, []int64{256}) {
				t.Fatalf("shape did not survive conversion: %+v", mapped)
			}
			data := converted.Bytes()
			for index, want := range test.checks {
				got := float16ToFloat32(binary.LittleEndian.Uint16(data[index*2:]))
				if got != want {
					t.Errorf("element %d = %v, want %v", index, got, want)
				}
			}
		})
	}
}

func TestKQuantRequiresWholeSuperBlocks(t *testing.T) {
	for _, tensorType := range []TensorType{Q4_K, Q5_K, Q6_K} {
		if _, err := encodedSize(tensorType, []uint64{128}); err == nil {
			t.Errorf("%s accepted a partial 256-element super-block", tensorType)
		}
	}
}

func goldenQ45KBlock(q5 bool) []byte {
	size := 144
	qsOffset := 16
	if q5 {
		size = 176
		qsOffset = 48
	}
	block := make([]byte, size)
	binary.LittleEndian.PutUint16(block[0:], float32ToFloat16(2))
	binary.LittleEndian.PutUint16(block[2:], float32ToFloat16(0.5))
	block[4] = 1 // scale 0
	block[5] = 2 // scale 1
	block[8] = 3 // min 0
	block[9] = 3 // min 1
	block[qsOffset] = 0x21
	if q5 {
		block[16] = 1 // high bit for the first low-nibble value only
	}
	return block
}

func goldenQ6KBlock() []byte {
	block := make([]byte, 210)
	binary.LittleEndian.PutUint16(block[208:], float32ToFloat16(1))
	for i := 0; i < 16; i++ {
		block[192+i] = byte(i + 1)
	}
	block[0], block[32] = 0x21, 0x21
	block[128], block[160] = 0xe4, 0xe4
	return block
}

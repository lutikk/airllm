//go:build linux

package hardware

import "os"

func readFile(name string) ([]byte, error) {
	return os.ReadFile(name)
}

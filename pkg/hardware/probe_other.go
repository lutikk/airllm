//go:build !darwin && !linux

package hardware

func probePlatform() Snapshot {
	return baseSnapshot()
}

package npz

import (
	"os/exec"
	"strings"
)

func runPython(script string, args ...string) (string, error) {
	cmd := exec.Command("python3", append([]string{"-c", script}, args...)...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

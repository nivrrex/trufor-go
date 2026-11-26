//go:build amd64 && !trufor_scalar

package kernels

import (
	"os"
	"strings"
	"sync"
)

var (
	simdOnce sync.Once
	simdOK   bool
)

// simdAvailable 判断当前 CPU/OS 是否支持 AVX2+FMA。
//
// 判定方式为解析 /proc/cpuinfo（Devuan/Linux 目标环境）；其它系统或文件不可读时
// 返回 false，自动退回纯 Go 实现（结果相同，性能较低）。
func simdAvailable() bool {
	simdOnce.Do(func() {
		blob, err := os.ReadFile("/proc/cpuinfo")
		if err != nil {
			return
		}
		flags := parseCPUFlags(string(blob))
		simdOK = flags["avx2"] && flags["fma"] && flags["avx"]
	})
	return simdOK
}

// parseCPUFlags 收集 cpuinfo 中所有 flags 行里的标记。
func parseCPUFlags(cpuinfo string) map[string]bool {
	flags := make(map[string]bool, 64)
	for _, line := range strings.Split(cpuinfo, "\n") {
		idx := strings.Index(line, ":")
		if idx < 0 {
			continue
		}
		key := strings.TrimSpace(line[:idx])
		if key != "flags" && key != "Features" {
			continue
		}
		for _, f := range strings.Fields(line[idx+1:]) {
			flags[f] = true
		}
	}
	return flags
}

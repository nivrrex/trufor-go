//go:build trufor_ort && cgo && linux && amd64

package ort

import (
	"fmt"
	"math"
	"os"
	"sync"
	"testing"
)

func TestSessionRun(t *testing.T) {
	model := os.Getenv("TRUFOR_ORT_TEST_MODEL")
	if model == "" {
		t.Skip("设置 TRUFOR_ORT_TEST_MODEL 后运行 ORT 集成测试")
	}
	session, err := Open(model, 2)
	if err != nil {
		t.Fatalf("Open() = %v", err)
	}
	input := make([]float32, 3*64*64)
	outputs, err := session.Run(input, 64, 64)
	if err != nil {
		t.Fatalf("Run() = %v", err)
	}
	if len(outputs[0].Data) != 2*64*64 || len(outputs[1].Data) != 64*64 || len(outputs[2].Data) != 1 {
		t.Fatalf("Run() 输出长度 = [%d,%d,%d]，期望 [8192,4096,1]",
			len(outputs[0].Data), len(outputs[1].Data), len(outputs[2].Data))
	}
	for i, output := range outputs {
		for j, value := range output.Data {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("Run() 输出 %d[%d] 非有限值 %v", i, j, value)
			}
		}
	}
	const parallel = 4
	var wg sync.WaitGroup
	errCh := make(chan error, parallel)
	for i := 0; i < parallel; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := session.Run(input, 64, 64)
			if err != nil {
				errCh <- err
				return
			}
			for j := range got[0].Data {
				if math.Float32bits(got[0].Data[j]) != math.Float32bits(outputs[0].Data[j]) {
					errCh <- fmt.Errorf("并发 Run() 输出 %d 不一致", j)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Errorf("并发 Run(): %v", err)
	}
	if err := session.Close(); err != nil {
		t.Fatalf("Close() = %v", err)
	}
	if _, err := session.Run(input, 64, 64); err == nil {
		t.Fatal("Close() 后 Run() 应返回错误")
	}
}

func TestOpenMissingModel(t *testing.T) {
	if _, err := Open(t.TempDir()+"/missing.onnx", 1); err == nil {
		t.Fatal("Open() 对不存在的模型应返回错误")
	}
}

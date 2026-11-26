package ops

import "sync"

// parallelFor 将 [0,n) 切成连续区间并行处理；小任务保持串行。
// fn 必须只写入各自区间对应的目标数据，不得修改共享输入。
func parallelFor(n, workers, minWork int, fn func(lo, hi int)) {
	if workers <= 1 || n < 2 || n < minWork {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * n / workers
		hi := (w + 1) * n / workers
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

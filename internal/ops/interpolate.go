package ops

import (
	"fmt"
	"math"

	"github.com/nivrrex/trufor-go/internal/tensor"
)

// ResizeBilinear 实现 PyTorch F.interpolate(mode='bilinear', align_corners=False)：
// 源坐标 src = (dst+0.5)*scale - 0.5，并在两端 clamp 到 [0, size-1]，再在相邻两行/列间线性插值。
func ResizeBilinear(x *tensor.Tensor, outH, outW int) (*tensor.Tensor, error) {
	if outH <= 0 || outW <= 0 {
		return nil, fmt.Errorf("resize: 目标尺寸非法 %dx%d", outH, outW)
	}
	if outH == x.H && outW == x.W {
		return x.Clone(), nil
	}
	out := tensor.New(x.C, outH, outW)
	inPlane := x.H * x.W
	outPlane := outH * outW
	scaleH := float64(x.H) / float64(outH)
	scaleW := float64(x.W) / float64(outW)

	// 预计算每一行/列的源坐标与权重
	y0 := make([]int, outH)
	y1 := make([]int, outH)
	wy := make([]float64, outH)
	for oy := 0; oy < outH; oy++ {
		src := (float64(oy)+0.5)*scaleH - 0.5
		if src < 0 {
			src = 0
		}
		if src > float64(x.H-1) {
			src = float64(x.H - 1)
		}
		i0 := int(math.Floor(src))
		if i0 > x.H-1 {
			i0 = x.H - 1
		}
		i1 := i0 + 1
		if i1 > x.H-1 {
			i1 = x.H - 1
		}
		y0[oy], y1[oy], wy[oy] = i0, i1, src-float64(i0)
	}
	x0 := make([]int, outW)
	x1 := make([]int, outW)
	wx := make([]float64, outW)
	for ox := 0; ox < outW; ox++ {
		src := (float64(ox)+0.5)*scaleW - 0.5
		if src < 0 {
			src = 0
		}
		if src > float64(x.W-1) {
			src = float64(x.W - 1)
		}
		i0 := int(math.Floor(src))
		if i0 > x.W-1 {
			i0 = x.W - 1
		}
		i1 := i0 + 1
		if i1 > x.W-1 {
			i1 = x.W - 1
		}
		x0[ox], x1[ox], wx[ox] = i0, i1, src-float64(i0)
	}

	for c := 0; c < x.C; c++ {
		srcPlane := x.Data[c*inPlane : (c+1)*inPlane]
		dstPlane := out.Data[c*outPlane : (c+1)*outPlane]
		for oy := 0; oy < outH; oy++ {
			rowA := srcPlane[y0[oy]*x.W : (y0[oy]+1)*x.W]
			rowB := srcPlane[y1[oy]*x.W : (y1[oy]+1)*x.W]
			dy := wy[oy]
			dstRow := dstPlane[oy*outW : (oy+1)*outW]
			for ox := 0; ox < outW; ox++ {
				dx := wx[ox]
				top := float64(rowA[x0[ox]])*(1-dx) + float64(rowA[x1[ox]])*dx
				bottom := float64(rowB[x0[ox]])*(1-dx) + float64(rowB[x1[ox]])*dx
				dstRow[ox] = float32(top*(1-dy) + bottom*dy)
			}
		}
	}
	return out, nil
}

// LogSigmoid 返回 log(sigmoid(x)) = -softplus(-x)，逐元素写入 dst。
func LogSigmoid(src, dst []float32) {
	for i, v := range src {
		x := float64(v)
		if x >= 0 {
			dst[i] = float32(-math.Log1p(math.Exp(-x)))
		} else {
			dst[i] = float32(x - math.Log1p(math.Exp(x)))
		}
	}
}

// weightedStatistics 对应上游 weighted_statistics_pooling 在单通道（C=1）下的结果：
// 返回 [x_min, x_max, x_avg, x_msq]。
// logW 为 nil 时等价于均匀权重（log_w = 0）。
func weightedStatistics(x []float32, logW []float32) [4]float64 {
	n := len(x)
	if n == 0 {
		return [4]float64{}
	}
	// log_softmax(logW)
	lw := make([]float64, n)
	if logW == nil {
		// 均匀权重时 log_softmax(0) = -log(n)
		v := -math.Log(float64(n))
		for i := range lw {
			lw[i] = v
		}
	} else {
		var maxV float64 = math.Inf(-1)
		for _, v := range logW {
			if f := float64(v); f > maxV {
				maxV = f
			}
		}
		var sum float64
		for _, v := range logW {
			sum += math.Exp(float64(v) - maxV)
		}
		logSum := maxV + math.Log(sum)
		for i := range logW {
			lw[i] = float64(logW[i]) - logSum
		}
	}

	// x_min = -logsumexp(lw - x)；x_max = logsumexp(lw + x)
	maxNeg := math.Inf(-1)
	maxPos := math.Inf(-1)
	for i := 0; i < n; i++ {
		f := float64(x[i])
		if v := lw[i] - f; v > maxNeg {
			maxNeg = v
		}
		if v := lw[i] + f; v > maxPos {
			maxPos = v
		}
	}
	var sumNeg, sumPos float64
	var avg, msq float64
	for i := 0; i < n; i++ {
		f := float64(x[i])
		sumNeg += math.Exp(lw[i] - f - maxNeg)
		sumPos += math.Exp(lw[i] + f - maxPos)
		w := math.Exp(lw[i])
		avg += w * f
		msq += w * f * f
	}
	logSumNeg := maxNeg + math.Log(sumNeg)
	logSumPos := maxPos + math.Log(sumPos)
	return [4]float64{-logSumNeg, logSumPos, avg, msq}
}

// WeightedStatisticsPooling 是 weighted_statistics_pooling 的单通道封装：
// x 与 logW 长度相同（H*W），logW 为 nil 时用均匀权重。
func WeightedStatisticsPooling(x []float32, logW []float32) [4]float32 {
	s := weightedStatistics(x, logW)
	return [4]float32{float32(s[0]), float32(s[1]), float32(s[2]), float32(s[3])}
}

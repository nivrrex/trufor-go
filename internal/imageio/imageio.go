// Package imageio 负责把图像文件解码成推理输入张量。
//
// 输入语义与上游一致：PIL `Image.open(...).convert("RGB")` 得到的 uint8 RGB，
// 再除以 256.0（注意不是 255）。16 位 PNG 按高字节截断（与 PIL 的 I;16→L 一致），
// 带 alpha 的图像直接丢弃 alpha（不合成）。
//
// 已知差异：JPEG 走 Go 标准库解码（最近邻色度上采样 + 各自 IDCT），与 PIL/libturbojpeg
// 的 fancy upsampling 结果不同，实测最大像素差 11，会导致 score 出现 ~0.07 量级偏差。
// PNG 路径与 PIL 逐像素一致（已实测）。需要 JPEG 精确复现时应改用 cgo + libjpeg-turbo。
package imageio

import (
	"fmt"
	"image"
	"image/draw"
	_ "image/gif" // 注册解码器
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/nivrrex/trufor-go/internal/tensor"
)

// LoadRGB 解码图像并返回 [0,1] 范围的 3 通道张量。
func LoadRGB(path string) (*tensor.Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("打开图像 %s: %w", path, err)
	}
	defer f.Close()
	img, format, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("解码图像 %s: %w", path, err)
	}
	return FromImage(img, format)
}

// FromImage 把已解码的图像转换为 [0,1] 的 3 通道张量。
func FromImage(img image.Image, format string) (*tensor.Tensor, error) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return nil, fmt.Errorf("图像尺寸非法: %dx%d", w, h)
	}
	out := tensor.New(3, h, w)
	plane := h * w

	// 快速路径：YCbCr（JPEG 常见）直接取亮度与色度
	if ycc, ok := img.(*image.YCbCr); ok {
		for y := 0; y < h; y++ {
			row := (b.Min.Y + y - ycc.Rect.Min.Y) * ycc.YStride
			for x := 0; x < w; x++ {
				xi := b.Min.X + x - ycc.Rect.Min.X
				yi := row + xi
				ci := ycc.COffset(xi, b.Min.Y+y)
				r, g, bl := colorYCbCrToRGB(ycc.Y[yi], ycc.Cb[ci], ycc.Cr[ci])
				idx := y*w + x
				out.Data[idx] = float32(r) / 256.0
				out.Data[plane+idx] = float32(g) / 256.0
				out.Data[2*plane+idx] = float32(bl) / 256.0
			}
		}
		return out, nil
	}

	// 通用路径：转换到 NRGBA 后只取 RGB（等价于丢弃 alpha）
	nrgba := image.NewNRGBA(image.Rect(0, 0, w, h))
	draw.Draw(nrgba, nrgba.Bounds(), img, b.Min, draw.Src)
	for y := 0; y < h; y++ {
		src := nrgba.Pix[y*nrgba.Stride : y*nrgba.Stride+w*4]
		for x := 0; x < w; x++ {
			idx := y*w + x
			out.Data[idx] = float32(src[x*4]) / 256.0
			out.Data[plane+idx] = float32(src[x*4+1]) / 256.0
			out.Data[2*plane+idx] = float32(src[x*4+2]) / 256.0
		}
	}
	return out, nil
}

func colorYCbCrToRGB(y, cb, cr uint8) (uint8, uint8, uint8) {
	const (
		yy1 = 1 << 15
	)
	yy := int32(y) * 0x10101
	cbv := int32(cb) - 128
	crv := int32(cr) - 128
	r := yy + 91881*crv
	g := yy - 22554*cbv - 46802*crv
	b := yy + 116130*cbv
	return clamp8((r + yy1) >> 16), clamp8((g + yy1) >> 16), clamp8((b + yy1) >> 16)
}

func clamp8(v int32) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

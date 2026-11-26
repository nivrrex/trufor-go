package trufor

import (
	"fmt"
	"math"

	"github.com/nivrrex/trufor-go/internal/ort"
	"github.com/nivrrex/trufor-go/internal/tensor"
)

type onnxBackend struct {
	session *ort.Session
	workers int
}

func newONNXBackend(weights string, workers int) (*onnxBackend, error) {
	session, err := ort.Open(weights, workers)
	if err != nil {
		return nil, err
	}
	return &onnxBackend{session: session, workers: workers}, nil
}

func (b *onnxBackend) detect(x *tensor.Tensor) (*Result, error) {
	outputs, err := b.session.Run(x.Data, x.H, x.W)
	if err != nil {
		return nil, err
	}
	logits, confLogits, det := outputs[0].Data, outputs[1].Data, outputs[2].Data
	npix := x.H * x.W
	if len(logits) < 2*npix || len(confLogits) < npix || len(det) < 1 {
		return nil, fmt.Errorf("trufor: ONNX 输出形状不完整")
	}
	mapData := make([]float32, npix)
	for i := 0; i < npix; i++ {
		o0 := float64(logits[i])
		o1 := float64(logits[npix+i])
		mx := o0
		if o1 > mx {
			mx = o1
		}
		e0 := math.Exp(o0 - mx)
		e1 := math.Exp(o1 - mx)
		mapData[i] = float32(e1 / (e0 + e1))
	}
	confData := make([]float32, npix)
	for i := 0; i < npix; i++ {
		confData[i] = float32(1 / (1 + math.Exp(-float64(confLogits[i]))))
	}
	score := float32(1 / (1 + math.Exp(-float64(det[0]))))
	return &Result{
		Map:    mapData,
		Conf:   confData,
		Score:  score,
		Width:  x.W,
		Height: x.H,
	}, nil
}

func (b *onnxBackend) close() error {
	return b.session.Close()
}

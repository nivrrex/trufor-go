# ONNX Runtime C API headers

- Upstream: https://github.com/microsoft/onnxruntime
- Version: v1.29.0
- Files:
  - `onnxruntime_c_api.h`
  - `onnxruntime_error_code.h`
  - `onnxruntime_ep_c_api.h`
- Retrieved from jsDelivr mirrors of the v1.29.0 tag:
  - https://cdn.jsdelivr.net/gh/microsoft/onnxruntime@v1.29.0/include/onnxruntime/core/session/onnxruntime_c_api.h
  - https://cdn.jsdelivr.net/gh/microsoft/onnxruntime@v1.29.0/include/onnxruntime/core/session/onnxruntime_error_code.h
  - https://cdn.jsdelivr.net/gh/microsoft/onnxruntime@v1.29.0/include/onnxruntime/core/session/onnxruntime_ep_c_api.h
- License: MIT, see `LICENSE`.

The Go wrapper requests the API version defined by these headers. The runtime itself is
not vendored or linked at build time; it is loaded with `dlopen` at runtime.

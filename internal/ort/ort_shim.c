//go:build trufor_ort && cgo && linux && amd64

#include "ort_shim.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/onnxruntime_c_api.h"

typedef struct tr_ort_session {
  void *library;
  const OrtApi *api;
  OrtEnv *env;
  OrtSessionOptions *options;
  OrtSession *session;
  OrtMemoryInfo *memory_info;
} tr_ort_session;

static void set_error(char *error, size_t error_size, const char *message) {
  if (error == NULL || error_size == 0) {
    return;
  }
  (void)snprintf(error, error_size, "%s", message);
}

static void set_status_error(const OrtApi *api, OrtStatus *status,
                             char *error, size_t error_size) {
  if (status == NULL) {
    return;
  }
  if (error != NULL && error_size != 0) {
    (void)snprintf(error, error_size, "%s", api->GetErrorMessage(status));
  }
  api->ReleaseStatus(status);
}

void *tr_ort_open(const char *model_path, int threads, char *error, size_t error_size) {
  if (model_path == NULL || model_path[0] == '\0' || threads < 1) {
    set_error(error, error_size, "tr_ort_open 参数非法");
    return NULL;
  }
  tr_ort_session *handle = calloc(1, sizeof(*handle));
  if (handle == NULL) {
    set_error(error, error_size, "分配 ORT session 失败");
    return NULL;
  }

  const char *configured = getenv("TRUFOR_ORT_LIBRARY");
  const char *candidates[3] = {
      configured != NULL && configured[0] != '\0' ? configured : "libonnxruntime.so.1",
      "libonnxruntime.so.1",
      "libonnxruntime.so",
  };
  for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
    handle->library = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
    if (handle->library != NULL) {
      break;
    }
  }
  if (handle->library == NULL) {
    if (error != NULL && error_size != 0) {
      (void)snprintf(error, error_size,
                     "无法加载 ONNX Runtime（设置 TRUFOR_ORT_LIBRARY 指定库路径）: %s",
                     dlerror());
    }
    free(handle);
    return NULL;
  }

  void *symbol = dlsym(handle->library, "OrtGetApiBase");
  const OrtApiBase *(*get_api_base)(void) = NULL;
  _Static_assert(sizeof(get_api_base) == sizeof(symbol),
                 "dlsym function pointer size mismatch");
  memcpy(&get_api_base, &symbol, sizeof(get_api_base));
  if (get_api_base == NULL) {
    set_error(error, error_size, "ONNX Runtime 缺少 OrtGetApiBase 符号");
    (void)dlclose(handle->library);
    free(handle);
    return NULL;
  }
  handle->api = get_api_base()->GetApi(ORT_API_VERSION);
  if (handle->api == NULL) {
    set_error(error, error_size, "ONNX Runtime API 版本不兼容");
    (void)dlclose(handle->library);
    free(handle);
    return NULL;
  }

  const OrtApi *api = handle->api;
  OrtStatus *status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "trufor-go", &handle->env);
  if (status != NULL) {
    set_status_error(api, status, error, error_size);
    goto fail;
  }
  status = api->CreateSessionOptions(&handle->options);
  if (status != NULL) {
    set_status_error(api, status, error, error_size);
    goto fail;
  }
  status = api->SetIntraOpNumThreads(handle->options, threads);
  if (status != NULL) {
    set_status_error(api, status, error, error_size);
    goto fail;
  }
  status = api->SetInterOpNumThreads(handle->options, 1);
  if (status != NULL) {
    set_status_error(api, status, error, error_size);
    goto fail;
  }
  status = api->SetSessionGraphOptimizationLevel(handle->options, ORT_ENABLE_ALL);
  if (status != NULL) {
    set_status_error(api, status, error, error_size);
    goto fail;
  }
  status = api->CreateSession(handle->env, model_path, handle->options, &handle->session);
  if (status != NULL) {
    set_status_error(api, status, error, error_size);
    goto fail;
  }
  size_t input_count = 0;
  size_t output_count = 0;
  status = api->SessionGetInputCount(handle->session, &input_count);
  if (status == NULL) {
    status = api->SessionGetOutputCount(handle->session, &output_count);
  }
  if (status != NULL) {
    set_status_error(api, status, error, error_size);
    goto fail;
  }
  if (input_count != 1 || output_count != 3) {
    set_error(error, error_size, "TruFor ONNX 需要 1 个输入和 3 个输出");
    goto fail;
  }
  status = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &handle->memory_info);
  if (status != NULL) {
    set_status_error(api, status, error, error_size);
    goto fail;
  }

  api->ReleaseSessionOptions(handle->options);
  handle->options = NULL;
  return handle;

fail:
  if (handle->memory_info != NULL) {
    api->ReleaseMemoryInfo(handle->memory_info);
  }
  if (handle->session != NULL) {
    api->ReleaseSession(handle->session);
  }
  if (handle->options != NULL) {
    api->ReleaseSessionOptions(handle->options);
  }
  if (handle->env != NULL) {
    api->ReleaseEnv(handle->env);
  }
  (void)dlclose(handle->library);
  free(handle);
  return NULL;
}

static int copy_output(const OrtApi *api, OrtValue *value, tr_ort_buffer *output,
                       char *error, size_t error_size) {
  OrtTensorTypeAndShapeInfo *shape_info = NULL;
  void *data = NULL;
  OrtStatus *status = api->GetTensorTypeAndShape(value, &shape_info);
  if (status != NULL) {
    set_status_error(api, status, error, error_size);
    return -1;
  }
  size_t rank = 0;
  status = api->GetDimensionsCount(shape_info, &rank);
  if (status != NULL) {
    set_status_error(api, status, error, error_size);
    api->ReleaseTensorTypeAndShapeInfo(shape_info);
    return -1;
  }
  if (rank > 4) {
    set_error(error, error_size, "TruFor ONNX 输出维度超过 4");
    api->ReleaseTensorTypeAndShapeInfo(shape_info);
    return -1;
  }
  int64_t dims[4] = {0, 0, 0, 0};
  status = api->GetDimensions(shape_info, dims, rank);
  if (status != NULL) {
    set_status_error(api, status, error, error_size);
    api->ReleaseTensorTypeAndShapeInfo(shape_info);
    return -1;
  }
  size_t elements = 1;
  for (size_t i = 0; i < rank; ++i) {
    if (dims[i] <= 0 || (uintmax_t)elements > SIZE_MAX / (size_t)dims[i]) {
      set_error(error, error_size, "TruFor ONNX 输出形状非法");
      api->ReleaseTensorTypeAndShapeInfo(shape_info);
      return -1;
    }
    elements *= (size_t)dims[i];
  }
  ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  status = api->GetTensorElementType(shape_info, &element_type);
  api->ReleaseTensorTypeAndShapeInfo(shape_info);
  if (status != NULL) {
    set_status_error(api, status, error, error_size);
    return -1;
  }
  if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    set_error(error, error_size, "TruFor ONNX 输出必须是 float32");
    return -1;
  }
  status = api->GetTensorMutableData(value, &data);
  if (status != NULL) {
    set_status_error(api, status, error, error_size);
    return -1;
  }
  if (output->data == NULL || output->capacity < elements) {
    set_error(error, error_size, "TruFor ONNX 输出缓冲容量不足");
    return -1;
  }
  memcpy(output->data, data, elements * sizeof(*output->data));
  output->rank = rank;
  for (size_t i = 0; i < rank; ++i) {
    output->shape[i] = dims[i];
  }
  return 0;
}

int tr_ort_run(void *opaque, const float *input, const int64_t *input_shape,
               tr_ort_buffer *outputs, char *error, size_t error_size) {
  if (opaque == NULL || input == NULL || input_shape == NULL || outputs == NULL) {
    set_error(error, error_size, "tr_ort_run 参数非法");
    return -1;
  }
  tr_ort_session *handle = opaque;
  const OrtApi *api = handle->api;
  OrtValue *input_value = NULL;
  OrtValue *output_values[3] = {NULL, NULL, NULL};
  size_t input_elements = 1;
  for (size_t i = 0; i < 4; ++i) {
    if (input_shape[i] <= 0 ||
        (uintmax_t)input_elements > SIZE_MAX / (size_t)input_shape[i]) {
      set_error(error, error_size, "TruFor ONNX 输入形状非法");
      return -1;
    }
    input_elements *= (size_t)input_shape[i];
  }

  OrtStatus *status = api->CreateTensorWithDataAsOrtValue(
      handle->memory_info, (void *)input, input_elements * sizeof(*input),
      input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_value);
  if (status != NULL) {
    set_status_error(api, status, error, error_size);
    return -1;
  }
  const OrtValue *inputs[1] = {input_value};
  const char *input_names[1] = {"rgb"};
  const char *output_names[3] = {"out", "conf", "det"};
  status = api->Run(handle->session, NULL, input_names, inputs, 1, output_names, 3, output_values);
  if (status != NULL) {
    set_status_error(api, status, error, error_size);
    api->ReleaseValue(input_value);
    return -1;
  }
  int result = 0;
  for (size_t i = 0; i < 3; ++i) {
    if (copy_output(api, output_values[i], &outputs[i], error, error_size) != 0) {
      result = -1;
      break;
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    if (output_values[i] != NULL) {
      api->ReleaseValue(output_values[i]);
    }
  }
  api->ReleaseValue(input_value);
  return result;
}

void tr_ort_close(void *session) {
  tr_ort_session *handle = session;
  if (handle == NULL) {
    return;
  }
  const OrtApi *api = handle->api;
  if (handle->memory_info != NULL) {
    api->ReleaseMemoryInfo(handle->memory_info);
  }
  if (handle->session != NULL) {
    api->ReleaseSession(handle->session);
  }
  if (handle->options != NULL) {
    api->ReleaseSessionOptions(handle->options);
  }
  if (handle->env != NULL) {
    api->ReleaseEnv(handle->env);
  }
  (void)dlclose(handle->library);
  free(handle);
}

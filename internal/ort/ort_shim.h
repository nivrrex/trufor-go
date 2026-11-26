#ifndef TRUFOR_ORT_SHIM_H
#define TRUFOR_ORT_SHIM_H

#include <stddef.h>
#include <stdint.h>

/* 每个输出缓冲由 Go 分配，C 只借用；shape/rank 回填实际推理形状。 */
typedef struct tr_ort_buffer {
  float *data;
  size_t capacity;
  size_t rank;
  int64_t shape[4];
} tr_ort_buffer;

/* 返回不透明 session；失败时 error 写入调用方缓冲。线程安全：同一 session 可并发 Run。 */
void *tr_ort_open(const char *model_path, int threads, char *error, size_t error_size);
int tr_ort_run(void *session, const float *input, const int64_t *input_shape,
               tr_ort_buffer *outputs, char *error, size_t error_size);
void tr_ort_close(void *session);

#endif

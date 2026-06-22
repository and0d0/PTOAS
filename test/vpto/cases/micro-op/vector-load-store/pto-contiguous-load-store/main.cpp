// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#define ACL_CHECK(expr) do { const aclError _ret = (expr); if (_ret != ACL_SUCCESS) { std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, __LINE__); rc = 1; goto cleanup; } } while (0)

void LaunchPto_contiguous_load_store_kernel(float *src_f32, float *dst_f32,
                                            int *src_i32, int *dst_i32,
                                            void *stream);

int main() {
  size_t elemCount = 64;
  size_t f32FileSize = elemCount * sizeof(float);
  size_t i32FileSize = elemCount * sizeof(int);
  float *srcF32Host = nullptr;
  float *dstF32Host = nullptr;
  int *srcI32Host = nullptr;
  int *dstI32Host = nullptr;
  float *srcF32Device = nullptr;
  float *dstF32Device = nullptr;
  int *srcI32Device = nullptr;
  int *dstI32Device = nullptr;
  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMallocHost((void **)(&srcF32Host), f32FileSize));
  ACL_CHECK(aclrtMallocHost((void **)(&dstF32Host), f32FileSize));
  ACL_CHECK(aclrtMallocHost((void **)(&srcI32Host), i32FileSize));
  ACL_CHECK(aclrtMallocHost((void **)(&dstI32Host), i32FileSize));
  ACL_CHECK(aclrtMalloc((void **)&srcF32Device, f32FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dstF32Device, f32FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&srcI32Device, i32FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dstI32Device, i32FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./src_f32.bin", f32FileSize, srcF32Host, f32FileSize);
  ReadFile("./dst_f32.bin", f32FileSize, dstF32Host, f32FileSize);
  ReadFile("./src_i32.bin", i32FileSize, srcI32Host, i32FileSize);
  ReadFile("./dst_i32.bin", i32FileSize, dstI32Host, i32FileSize);
  ACL_CHECK(aclrtMemcpy(srcF32Device, f32FileSize, srcF32Host, f32FileSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dstF32Device, f32FileSize, dstF32Host, f32FileSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(srcI32Device, i32FileSize, srcI32Host, i32FileSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dstI32Device, i32FileSize, dstI32Host, i32FileSize, ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchPto_contiguous_load_store_kernel(
      srcF32Device, dstF32Device, srcI32Device, dstI32Device, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(dstF32Host, f32FileSize, dstF32Device, f32FileSize, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(dstI32Host, i32FileSize, dstI32Device, i32FileSize, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./dst_f32.bin", dstF32Host, f32FileSize);
  WriteFile("./dst_i32.bin", dstI32Host, i32FileSize);

cleanup:
  aclrtFree(srcF32Device);
  aclrtFree(dstF32Device);
  aclrtFree(srcI32Device);
  aclrtFree(dstI32Device);
  aclrtFreeHost(srcF32Host);
  aclrtFreeHost(dstF32Host);
  aclrtFreeHost(srcI32Host);
  aclrtFreeHost(dstI32Host);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}

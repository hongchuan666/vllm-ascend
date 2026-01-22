/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_v2_host_utils.h
 * \brief
 */

#ifndef OP_HOST_GROUPED_MATMUL_SWIGLU_QUANT_V2_HOST_UTILS_H
#define OP_HOST_GROUPED_MATMUL_SWIGLU_QUANT_V2_HOST_UTILS_H

#include <map>

namespace GroupedMatmulSwigluQuantParamsV2 {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t PER_TOKEN_SCALE_INDEX = 1;
constexpr uint32_t GROUPLIST_INDEX = 2;
constexpr uint32_t WEIGHT_INDEX = 3;
constexpr uint32_t SCALE_INDEX = 4;
constexpr uint32_t Y_DATA_INDEX = 0;
constexpr uint32_t Y_SCALE_INDEX = 1;
constexpr uint64_t TILING_KEY = 0;
constexpr uint64_t ATTR_INDEX_DEQUANT_MODE = 0;
constexpr uint32_t ATTR_INDEX_DEQUANT_DTYPE = 1;
constexpr uint64_t ATTR_INDEX_QUANT_MODE = 2;
constexpr uint32_t ATTR_INDEX_QUANT_DTYPE = 3;
constexpr uint64_t ATTR_INDEX_TRANS_W = 4;
constexpr uint32_t ATTR_INDEX_GROUP_LIST_TYPE = 5;
constexpr size_t MX_WEIGHT_SCALE_DIM =4;
constexpr size_t MX_X_SCALE_DIM =3;
constexpr uint64_t B4_DATACOPY_MIN_NUM = 2;
} // namespace GroupedMatmulSwigluQuantParamsV2
#endif
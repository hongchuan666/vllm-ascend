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
 * \file grouped_matmul_swiglu_quant_v2_apt.cpp
 * \brief
 */

#include "kernel_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "arch35/grouped_matmul_swiglu_quant_v2_mxquant.h"
#include "arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h"
using namespace AscendC;
using namespace matmul;

template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS, int8_t KERNEL_TYPE>
__global__ __aicore__ void grouped_matmul_swiglu_quant_v2(GM_ADDR x, GM_ADDR xScale, GM_ADDR groupList,
                                                                     GM_ADDR weight, GM_ADDR weightScale,
                                                                     GM_ADDR weightAssistanceMatrix, GM_ADDR bias,
                                                                     GM_ADDR smoothScale, GM_ADDR y, GM_ADDR yScale,
                                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);

    if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_NO_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS
        && KERNEL_TYPE == GMM_SWIGLU_QUANT_DEQUANT_FIXP) { // transX = false, transW = false
        GmmSwigluAswt<Act::Gemm::layout::RowMajor, Act::Gemm::layout::RowMajor>(x, weight, weightScale,
                                                                                xScale, weightAssistanceMatrix,
                                                                                smoothScale, groupList, y,
                                                                                yScale, workspace, tiling);
    } else if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS
        && KERNEL_TYPE == GMM_SWIGLU_QUANT_DEQUANT_FIXP) { // transX = false, transW = true
        GmmSwigluAswt<Act::Gemm::layout::RowMajor, Act::Gemm::layout::ColumnMajor>(x, weight, weightScale,
                                                                                   xScale, weightAssistanceMatrix,
                                                                                   smoothScale, groupList, y,
                                                                                   yScale, workspace, tiling);
    }
}

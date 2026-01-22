# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME GroupedMatmulSwigluQuantV2
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnnExc PRIVATE
        grouped_matmul_swiglu_quant_v2_def.cpp
)

target_sources(opapi PRIVATE
        op_api/grouped_matmul_swiglu_quant_v2.cpp
        op_api/aclnn_grouped_matmul_swiglu_quant_v2.cpp
)

target_sources(aclnn_ops_train PRIVATE
    op_api/grouped_matmul_swiglu_quant_v2.cpp
    op_api/aclnn_grouped_matmul_swiglu_quant_v2.cpp
)

target_sources(aclnn_ops_infer PRIVATE
    op_api/grouped_matmul_swiglu_quant_v2.cpp
    op_api/aclnn_grouped_matmul_swiglu_quant_v2.cpp
)

target_sources(optiling PRIVATE
        op_tiling/grouped_matmul_swiglu_quant_v2_tiling.cpp
)

target_sources(opsproto PRIVATE
        grouped_matmul_swiglu_quant_v2_infershape.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

file(GLOB _GMM_V2_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_grouped_matmul_swiglu_quant_v2.h")

install(FILES ${_GMM_V2_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)

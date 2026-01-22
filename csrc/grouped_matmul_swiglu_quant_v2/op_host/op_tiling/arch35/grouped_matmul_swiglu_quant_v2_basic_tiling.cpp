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
 * \file grouped_matmul_swiglu_quant_v2_basic_tiling.cpp
 * \brief
 */

#include <alog_pub.h>
#include <climits>
#include "log/log.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_type.h"
#include "register/op_impl_registry.h"
#include "grouped_matmul_swiglu_quant_v2_basic_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h"
using namespace Ops::Transformer::OpTiling;
using namespace GroupedMatmulSwigluQuantParamsV2;
using namespace optiling::GmmConstant;
namespace optiling{

void GroupedMatmulSwigluQuantDavidV2Tiling::Reset()
{
    tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
    return;
}

bool GroupedMatmulSwigluQuantDavidV2Tiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE); // 通路保证非负数
        inputParams_.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : inputParams_.groupListType;
        OP_CHECK_IF(!(inputParams_.groupListType == 0 || inputParams_.groupListType == 1),
                    OP_LOGE(context_->GetNodeName(), "GroupListType must be 0 or 1, but actual value is %d.",
                            inputParams_.groupListType),
                    return false);
    }
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is nullptr."), return false);
    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
    inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    return true;
}

bool GroupedMatmulSwigluQuantDavidV2Tiling::AnalyzeDtype()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "xDesc is nullptr."), return false);
    inputParams_.aDtype = xDesc->GetDataType();
    auto wDesc = context_->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context_->GetNodeName(), "wDesc is nullptr."), return false);
    inputParams_.bDtype = wDesc->GetDataType();
    auto scaleDesc = context_->GetInputDesc(SCALE_INDEX);
    OP_CHECK_IF(scaleDesc == nullptr, OP_LOGE(context_->GetNodeName(), "scaleDesc is nullptr."), return false);
    inputParams_.scaleDtype = scaleDesc->GetDataType();
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
    inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;
    auto outDesc = context_->GetOutputDesc(Y_DATA_INDEX);
    OP_CHECK_IF(outDesc == nullptr, OP_LOGE(context_->GetNodeName(), "OutDesc is nullptr."), return false);
    inputParams_.outDataDtype = outDesc->GetDataType();
    auto outScaleDesc = context_->GetOutputDesc(Y_SCALE_INDEX);
    OP_CHECK_IF(outScaleDesc == nullptr, OP_LOGE(context_->GetNodeName(), "OutScaleDesc is nullptr."), return false);
    inputParams_.outScaleDtype = outScaleDesc->GetDataType();
    return CheckDtype();
}

bool GroupedMatmulSwigluQuantDavidV2Tiling::IsFp4(ge::DataType dtype) {
    return dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_FLOAT4_E2M1;
}

bool GroupedMatmulSwigluQuantDavidV2Tiling::IsFp8(ge::DataType dtype) {
    return dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2;
}

bool GroupedMatmulSwigluQuantDavidV2Tiling::IsFp4Input() {
    return IsFp4(inputParams_.aDtype) && IsFp4(inputParams_.bDtype);
}

bool GroupedMatmulSwigluQuantDavidV2Tiling::IsFp8Input() {
    return IsFp8(inputParams_.aDtype) && IsFp8(inputParams_.bDtype);
}

bool GroupedMatmulSwigluQuantDavidV2Tiling::CheckDtype()
{
    OP_CHECK_IF(!(IsFp4Input() || IsFp8Input()),
                OP_LOGE(inputParams_.opName,
                        "Only FLOAT8 or FLOAT4 inputs are supported, but x dtype is %s, weight dtype is %s.",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
                return false);
    OP_CHECK_IF(
        inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0 || inputParams_.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
        OP_LOGE(inputParams_.opName, "Xscale and weightScale dtype must be DT_FLOAT8_E8M0, but actual dtype is %s, %s.",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str()),
        return false);
    OP_CHECK_IF(!(IsFp4(inputParams_.outDataDtype) || IsFp8(inputParams_.outDataDtype)),
                OP_LOGE(inputParams_.opName, "Only FLOAT8 or FLOAT4 output are supported, but out dtype is %s.",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.outDataDtype).c_str()),
                return false);
    OP_CHECK_IF(inputParams_.outScaleDtype != ge::DT_FLOAT8_E8M0,
                OP_LOGE(inputParams_.opName, "OutScale dtype must be DT_FLOAT8_E8M0, but actual dtype is %s.",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.outScaleDtype).c_str()),
                return false);

    OP_CHECK_IF(
        IsFp8Input() && !IsFp8(inputParams_.outDataDtype),
        OP_LOGE(inputParams_.opName, "When inputs are FLOAT8, outData dtype must be FLOAT8, but out dtype is %s.",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.outDataDtype).c_str()),
        return false);
    return true;
}

bool GroupedMatmulSwigluQuantDavidV2Tiling::SetQuantModeForGMMSwigluQuant()
{
    if (IsMicroScaling()) {
        inputParams_.bQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        inputParams_.aQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        return true;
    }
    return false;
}

bool GroupedMatmulSwigluQuantDavidV2Tiling::AnalyzeInputs()
{
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_IF(xStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "xStorageShape is nullptr."), return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();
    auto wStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_IF(wStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "wStorageShape is nullptr."), return false);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();
    auto scaleStorageShape = context_->GetDynamicInputShape(SCALE_INDEX, 0);
    OP_CHECK_IF(scaleStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "scaleStorageShape is nullptr."), return false);
    const gert::Shape &wScaleShape = scaleStorageShape->GetOriginShape();
    auto scaleDimNum = wScaleShape.GetDimNum();
    OP_CHECK_IF(
        scaleDimNum != MX_WEIGHT_SCALE_DIM,
        OP_LOGE(inputParams_.opName, "The dimension of weight_scale should be equal to 4, actual is %zu", scaleDimNum),
        return false);
    auto x1ScaleStorageShape = context_->GetInputShape(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(x1ScaleStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "XScaleStorageShape is nullptr."), return false);
    const gert::Shape &xScaleShape = x1ScaleStorageShape->GetOriginShape();
    auto xScaleDimNum = xScaleShape.GetDimNum();
    OP_CHECK_IF(
        xScaleDimNum != MX_X_SCALE_DIM,
        OP_LOGE(inputParams_.opName, "The dimension of x_scale should be equal to 3, actual is %zu", xScaleDimNum),
        return false);
    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(inputParams_.opName, "SetGroupNum failed."),
               return false);
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(inputParams_.opName, "SetMKN failed."), return false);
    auto aInnerSize = inputParams_.transA ? inputParams_.mSize : inputParams_.kSize;
    auto bInnerSize = inputParams_.transB ? inputParams_.kSize : inputParams_.nSize;
    OP_CHECK_IF(
        IsFp4Input() && (aInnerSize % B4_DATACOPY_MIN_NUM != 0 || bInnerSize % B4_DATACOPY_MIN_NUM != 0),
        OP_LOGE(inputParams_.opName, "When inputs are FLOAT4, x and weight inner axis element number shoud be even."),
        return false);
    OP_CHECK_IF(inputParams_.nSize % GmmConstant::BASIC_BLOCK_SIZE_128 != 0,
                OP_LOGE(inputParams_.opName, "Weight n axis element number shoud be an integer multiple of 128."),
                return false);

    OP_CHECK_IF(!SetQuantModeForGMMSwigluQuant(),
               OP_LOGE(inputParams_.opName, "SetQuantModeForGMMSwigluQuant failed."), return false);
    return true;
}

ge::graphStatus GroupedMatmulSwigluQuantDavidV2Tiling::DoOpTiling()
{
    tilingData_.gmmSwigluQuantParams.set_groupNum(inputParams_.groupNum);
    tilingData_.gmmSwigluQuantParams.set_groupListType(static_cast<uint8_t>(inputParams_.groupListType));
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *dequantDtypeTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_DTYPE);
        uint64_t dequantDtype = dequantDtypeTypePtr != nullptr ? static_cast<uint64_t>(*dequantDtypeTypePtr) : 0;
        const int64_t *quantDtypeTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_DTYPE);
        uint64_t quantDtype = quantDtypeTypePtr != nullptr ? static_cast<uint64_t>(*quantDtypeTypePtr) : 0;
        tilingData_.gmmSwigluQuantParams.set_quantDtype(static_cast<uint8_t>(quantDtype));
    }
    PrintQuantParams();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantDavidV2Tiling::DoLibApiTiling()
{
    CalBasicBlock();
    auto baseM_modified = std::min(basicTiling_.baseM, static_cast<uint64_t>(128));
    basicTiling_.baseM = GroupedMatmul::CeilAlign(baseM_modified, GmmConstant::CUBE_BLOCK);
    OP_CHECK_IF(CalL1Tiling() != ge::GRAPH_SUCCESS,
               OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed"), return ge::GRAPH_FAILED);
    tilingData_.mmTilingData.set_M(inputParams_.mSize);
    tilingData_.mmTilingData.set_N(inputParams_.nSize);
    tilingData_.mmTilingData.set_Ka(inputParams_.kSize);
    tilingData_.mmTilingData.set_Kb(inputParams_.kSize);
    tilingData_.mmTilingData.set_usedCoreNum(aicoreParams_.aicNum);
    tilingData_.mmTilingData.set_baseM(basicTiling_.baseM);
    tilingData_.mmTilingData.set_baseN(basicTiling_.baseN);
    tilingData_.mmTilingData.set_baseK(basicTiling_.baseK);
    tilingData_.mmTilingData.set_singleCoreM(basicTiling_.baseM);
    tilingData_.mmTilingData.set_singleCoreN(basicTiling_.singleCoreN);
    tilingData_.mmTilingData.set_singleCoreK(basicTiling_.singleCoreK);
    tilingData_.mmTilingData.set_depthA1(basicTiling_.depthA1);
    tilingData_.mmTilingData.set_depthB1(basicTiling_.depthB1);
    tilingData_.mmTilingData.set_stepM(basicTiling_.stepM);
    tilingData_.mmTilingData.set_stepN(basicTiling_.stepN);
    tilingData_.mmTilingData.set_stepKa(basicTiling_.stepKa);
    tilingData_.mmTilingData.set_stepKb(basicTiling_.stepKb);
    tilingData_.mmTilingData.set_isBias(inputParams_.hasBias ? 1 : 0);
    tilingData_.mmTilingData.set_iterateOrder(basicTiling_.iterateOrder);
    tilingData_.mmTilingData.set_dbL0A(2); // db switch, 1: off, 2: on
    tilingData_.mmTilingData.set_dbL0B(2); // db switch, 1: off, 2: on
    tilingData_.mmTilingData.set_dbL0C(basicTiling_.dbL0c);
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        if (basicTiling_.scaleFactorA >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorA <= SCALER_FACTOR_MAX &&
            basicTiling_.scaleFactorB >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorB <= SCALER_FACTOR_MAX) {
            tilingData_.mmTilingData.set_mxTypePara(
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (basicTiling_.scaleFactorB << SCALER_FACTOR_B_BIT) + basicTiling_.scaleFactorA);
        } else {
            tilingData_.mmTilingData.set_mxTypePara(
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) + SCALER_FACTOR_DEFAULT);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantDavidV2Tiling::PostTiling()
{
    context_->SetBlockDim(aicoreParams_.aicNum);
    OP_CHECK_IF(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
               OP_LOGE(context_->GetNodeName(), "Tiling data size[%zu] is not aligned to 8",
                                         tilingData_.GetDataSize()),
               return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void GroupedMatmulSwigluQuantDavidV2Tiling::PrintQuantParams()
{
    int32_t enable = AlogCheckDebugLevel(static_cast<int32_t>(OP), DLOG_DEBUG);
    if (enable != 1) {
        return;
    }
    optiling::GMMSwigluQuantParams &params = tilingData_.gmmSwigluQuantParams;
    std::ostringstream oss;
    oss << "GMMQuantParams: groupNum = " << params.get_groupNum()
        << ", groupListType = " << static_cast<uint32_t>(params.get_groupListType())
        << ", quant_dtype = " << static_cast<int32_t>(params.get_quantDtype());
    OP_LOGD(inputParams_.opName, "%s", oss.str().c_str());
}
uint64_t GroupedMatmulSwigluQuantDavidV2Tiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transB), static_cast<uint64_t>(inputParams_.transA),
        static_cast<uint64_t>(inputParams_.kernelType));
}

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulSwigluQuantV2, GroupedMatmulSwigluQuantDavidV2Tiling, 2);
} // namespace optiling
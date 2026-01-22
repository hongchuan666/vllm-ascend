/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <thread>
#include <gmock/gmock.h>
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"
#include "../../../../op_host/op_api/aclnn_grouped_matmul_swiglu_quant_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_GroupedMatmulSwigluQuantV2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_GroupedMatmulSwigluQuantV2_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_GroupedMatmulSwigluQuantV2_test TearDown" << endl;
    }
};

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w8a8_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_ND, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({bs, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend91095_test_opapi_normal_case)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m, k / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m, k /64 / 2 , 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend91095_test_opapi_mxfp4_normal_case)
{
    int64_t m = 2048;
    int64_t k = 8192;
    int64_t n = 7168;
    int64_t e = 4;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT4_E1M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT4_E1M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m, k / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m, k /64 / 2 , 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend91095_test_opapi_illegal_case)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m, k / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m, k /64 / 2 , 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, weight_desc, nullptr, xScale_desc,
                              nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}
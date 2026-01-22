# moe_distribute_dispatch_v2 INT4 量化输出支持总结

## 概述

本次修改为 `moe_distribute_dispatch_v2` 算子添加了 INT4 量化输出支持，参考了 `dynamic_quant` 算子的实现方式。

**重要说明**：为了支持 torch 调用（torch 不支持 INT4），还添加了 INT32 输出类型支持。INT32 是 INT4 的伪装，实际是 8 个 INT4 打包成 1 个 INT32。

---

## Host 侧支持

### 1. InferDataType（数据类型推断）

**文件**: `op_host/moe_distribute_dispatch_v2_infershape.cpp` (第253-284行)

**功能**: 推断输出数据类型

**关键修改**:
```cpp
// 检查输出数据类型是否已设置（用于支持INT4和INT32）
// INT32用于torch调用，实际是INT4打包成INT32（8个INT4打包成1个INT32）
const auto expandXOutDtype = context->GetOutputDataType(DISPATCH_OUTPUT_EXPAND_X_INDEX);
if (expandXOutDtype == ge::DT_INT4 || expandXOutDtype == ge::DT_INT32) {
    // 如果输出类型已设置为INT4或INT32，则保持原样
    // INT32是INT4的伪装，实际需要按照INT4计算
    context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_X_INDEX, expandXOutDtype);
} else if (quantFlag || (*quantMode != 0)) {
    context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_X_INDEX, ge::DT_INT8);
} else {
    context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_X_INDEX, xDtype);
}
```

**说明**:
- 支持 `DT_INT4` 和 `DT_INT32` 作为输出类型
- `DT_INT32` 是 `DT_INT4` 的伪装，用于 torch 调用
- 如果用户已设置输出类型为 INT4 或 INT32，则保持原样

### 2. InferShape（形状推断）

**文件**: `op_host/moe_distribute_dispatch_v2_infershape.cpp` (第188-201行)

**功能**: 推断输出形状

**关键修改**:
```cpp
// 如果输出是INT32（实际是INT4），最后一个维度需要除以8（8个INT4打包成1个INT32）
auto expandXOutDtype = context->GetOutputDataType(DISPATCH_OUTPUT_EXPAND_X_INDEX);
if (expandXOutDtype == ge::DT_INT32) {
    // INT32是INT4的伪装，8个INT4打包成1个INT32
    OP_CHECK_IF((h % 8) != 0,
        OP_LOGE(context->GetNodeName(), "If expandX dataType is int32 (int4 packed), the last dim of x must be divisible by 8, but the last dim is (%ld).", h),
        return ge::GRAPH_FAILED);
    expandXShape->SetDim(1U, h / 8);
} else {
    expandXShape->SetDim(1U, h);
}
```

**说明**:
- **INT32 输出**: 最后一个维度是输入的 1/8（8个INT4打包成1个INT32）
- **INT4 输出**: 最后一个维度与输入相同（但实际存储大小是输入的一半）
- 对于 INT32，需要检查输入的最后一个维度是否能被 8 整除

### 3. CheckTensorDataType（数据类型检查）

**文件**: `op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp` (第323-350行)

**功能**: 检查输入输出数据类型是否匹配

**关键修改**:
```cpp
if (quantMode != NO_SCALES) {
    // 支持INT8、INT4和INT32输出类型（INT32是INT4的伪装，用于torch调用）
    OP_TILING_CHECK((expandXDesc->GetDataType() != ge::DT_INT8) && 
        (expandXDesc->GetDataType() != ge::DT_INT4) && (expandXDesc->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be int8, int4, or int32, but is %s.",
        Ops::Base::ToString(expandXDesc->GetDataType()).c_str()), return false);
    
    // 如果输出类型是INT32（实际是INT4），检查输入的最后一个维度是输出的8倍
    if (expandXDesc->GetDataType() == ge::DT_INT32) {
        OP_TILING_CHECK(xLastDim != (expandXLastDim * 8),
            OP_LOGE(nodeName, "If expandX dataType is int32 (int4 packed), the last dim of x must be 8 times the last dim of expandX, "
            "but x last dim is (%ld) and expandX last dim is (%ld).", xLastDim, expandXLastDim), return false);
    } else if (expandXDesc->GetDataType() == ge::DT_INT4) {
        // 如果输出类型是INT4，检查最后一维是否能被2整除（因为2个int4打包成1个字节）
        OP_TILING_CHECK((xLastDim % 2) != 0,
            OP_LOGE(nodeName, "If expandX dataType is int4, the last dim of x must be divisible by 2, but the last dim is (%ld).",
            xLastDim), return false);
    }
}
```

**约束条件**:
- **INT4 输出**: 输入的最后一个维度必须能被 2 整除（2个INT4打包成1个字节）
- **INT32 输出**: 输入的最后一个维度必须是输出的 8 倍（8个INT4打包成1个INT32）

### 4. CheckWinSize（Window 内存大小检查）

**文件**: `op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp` (第983-994行)

**功能**: 检查 Window 内存大小是否足够

**关键修改**:
```cpp
// 根据输出数据类型调整 MAX_OUT_DTYPE_SIZE
uint64_t expandXOutDtypeSize = MAX_OUT_DTYPE_SIZE;
if (expandXDesc->GetDataType() == ge::DT_INT4 || expandXDesc->GetDataType() == ge::DT_INT32) {
    expandXOutDtypeSize = 1UL;  // INT4和INT32都使用1字节（INT32是INT4的伪装）
} else if (expandXDesc->GetDataType() == ge::DT_INT8) {
    expandXOutDtypeSize = MAX_OUT_DTYPE_SIZE;  // INT8使用2字节
}

uint64_t tokenNeedSizeCombine = ((h * expandXOutDtypeSize + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
uint64_t tokenActualLen = ((h * expandXOutDtypeSize + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN + SCALE_EXPAND_IDX_BUFFER;
```

**说明**:
- INT4 和 INT32 都使用 `1UL` 作为数据类型大小（因为都是按字节计算）
- 使用 `expandXOutDtypeSize` 替代固定的 `MAX_OUT_DTYPE_SIZE` 来计算内存大小

---

## Kernel 侧支持

### 1. 宏定义（统一处理 sizeof 计算）

**文件**: `op_kernel/moe_distribute_dispatch_v2.h` (第65-75行)
**文件**: `op_kernel/moe_distribute_dispatch_v2_full_mesh.h` (第60-70行)

**问题背景**: 由于 `sizeof(int4b_t)` 在逻辑上是 0.5 字节，但 C++ `sizeof` 只能返回整数（返回 0 或 1），导致计算有问题。

**解决方案**: 定义宏统一处理大小计算

```cpp
// 宏：计算 x * sizeof(TYPE)（字节数），其中sizeof为逻辑字节大小
// 对于 int4b_t: x * 0.5 = x >> 1
// 对于其他类型: x * sizeof(TYPE)
#define MUL_SIZEOF(x, TYPE) \
    (IsSameType<TYPE, int4b_t>::value ? ((x) >> 1) : ((x) * sizeof(TYPE)))

// 宏：计算 x / sizeof(TYPE)（元素数），其中sizeof为逻辑字节大小
// 对于 int4b_t: x / 0.5 = x << 1
// 对于其他类型: x / sizeof(TYPE)
#define DIV_SIZEOF(x, TYPE) \
    (IsSameType<TYPE, int4b_t>::value ? ((x) << 1) : ((x) / sizeof(TYPE)))
```

**优势**:
- 统一处理：所有 `sizeof(ExpandXOutType)` 的计算都通过宏统一处理
- 避免错误：不再需要手动写 `if constexpr` 判断 `int4b_t`
- 代码简洁：减少了重复的条件判断代码
- 易于维护：如果需要修改逻辑，只需修改宏定义

### 2. 输出大小计算（hOutSize_）

**文件**: `op_kernel/moe_distribute_dispatch_v2.h` (第395行)
**文件**: `op_kernel/moe_distribute_dispatch_v2_full_mesh.h` (第280行)

**修改**:
```cpp
hOutSize_ = MUL_SIZEOF(axisH_, ExpandXOutType);
```

**说明**:
- 对于 INT4：`hOutSize_ = axisH_ >> 1`（2个INT4打包成1个字节）
- 对于其他类型：`hOutSize_ = axisH_ * sizeof(ExpandXOutType)`

### 3. Window 对齐元素数计算（hAlignWinCnt_）

**文件**: `op_kernel/moe_distribute_dispatch_v2.h` (第402行)

**修改**:
```cpp
hAlignWinCnt_ = DIV_SIZEOF(hAlignWinSize_, ExpandXOutType);
```

**说明**:
- 对于 INT4：`hAlignWinCnt_ = hAlignWinSize_ << 1`（按元素数计算）
- 对于其他类型：`hAlignWinCnt_ = hAlignWinSize_ / sizeof(ExpandXOutType)`

### 4. 通信元素数计算（axisHCommu）

**文件**: `op_kernel/moe_distribute_dispatch_v2.h` (第532行)
**文件**: `op_kernel/moe_distribute_dispatch_v2_full_mesh.h` (第286行)

**修改**:
```cpp
uint32_t axisHCommu = DIV_SIZEOF(hScaleIdxSize, ExpandXOutType);
```

**说明**:
- 对于 INT4：`axisHCommu = hScaleIdxSize << 1`（按元素数计算）
- 对于其他类型：`axisHCommu = hScaleIdxSize / sizeof(ExpandXOutType)`

### 5. DataCopy 参数设置

**文件**: `op_kernel/moe_distribute_dispatch_v2.h` (第535-536行)
**文件**: `op_kernel/moe_distribute_dispatch_v2_full_mesh.h` (第357行)

**修改**:
```cpp
hCommuCopyOutParams_ = {1U, static_cast<uint32_t>(MUL_SIZEOF(axisHCommu, ExpandXOutType)), 0U, 0U, 0U};
expandXCopyParams_ = {1U, static_cast<uint32_t>(MUL_SIZEOF(axisH_, ExpandXOutType)), 0U, 0U, 0U};
```

**说明**:
- 使用 `MUL_SIZEOF` 宏统一计算 `blockLen`（字节数）
- 对于 INT4：自动处理为 `axisH_ >> 1` 或 `axisHCommu >> 1`
- 对于其他类型：正常计算 `axisH_ * sizeof(ExpandXOutType)`

### 6. DataCopy 地址偏移计算

**文件**: `op_kernel/moe_distribute_dispatch_v2_full_mesh.h` (第470, 489, 493行)

**修改**:
```cpp
DataCopy(dstWinGMTensor, outTensor_[DIV_SIZEOF(flagPadOffset_, ExpandXOutType)], axisHCommu_);
Copy(outTensor_[DIV_SIZEOF(flagPadOffset_, ExpandXOutType)], xInTensor, copyLen1, ...);
```

**说明**:
- 使用 `DIV_SIZEOF` 宏统一计算地址偏移（元素数）
- 对于 INT4：自动处理为 `flagPadOffset_ << 1`
- 对于其他类型：正常计算 `flagPadOffset_ / sizeof(ExpandXOutType)`

### 7. 量化过程（QuantProcess）

**文件**: `op_kernel/moe_distribute_dispatch_v2.h` (第1035-1071行)
**文件**: `op_kernel/moe_distribute_dispatch_v2_full_mesh.h` (第410-466行)

**关键修改**:

#### 7.1 INT4 最大值限制
```cpp
// INT4 量化需要限制最大值为2.56
if constexpr (IsSameType<ExpandXOutType, int4b_t>::value) {
    PipeBarrier<PIPE_V>();
    Min(floatLocalAbsTemp, floatLocalAbsTemp, float(2.56), axisH_);
    PipeBarrier<PIPE_V>();
}
```

#### 7.2 Scale 计算
```cpp
// 根据输出类型选择最大值：INT4使用7.0，INT8使用127.0
if constexpr (IsSameType<ExpandXOutType, int4b_t>::value) {
    dynamicScale = float(INT4_MAX_VALUE) / floatLocalAbsTemp.GetValue(0);  // INT4_MAX_VALUE = 7.0f
} else {
    dynamicScale = float(INT8_MAX_VALUE) / floatLocalAbsTemp.GetValue(0);  // INT8_MAX_VALUE = 127.0f
}
```

#### 7.3 类型转换
```cpp
// Cast 操作会自动处理 INT4 的打包（如果目标类型是 int4b_t）
LocalTensor<half> halfLocalTemp = floatLocalTemp.ReinterpretCast<half>();
LocalTensor<int32_t> int32LocalTemp = floatLocalTemp.ReinterpretCast<int32_t>();
Cast(int32LocalTemp, floatLocalTemp, RoundMode::CAST_RINT, axisH_);
PipeBarrier<PIPE_V>();
SetDeqScale((half)1.000000e+00f);
PipeBarrier<PIPE_V>();
Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);
PipeBarrier<PIPE_V>();
Cast(xOutTensor_, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);
```

**说明**:
- 直接使用 `Cast` 从 `half` 转换到 `int4b_t`，`Cast` 操作会自动处理 INT4 的打包
- 参考了 `dynamic_quant.h` 的实现方式

### 8. 常量定义

**文件**: `op_kernel/moe_distribute_dispatch_v2.h` (第37-38行)

```cpp
constexpr float INT4_MAX_VALUE = 7.0f;      // INT4 对称量化最大值
constexpr float INT8_MAX_VALUE = 127.0f;   // INT8 对称量化最大值
```

---

## 关键设计决策

### 1. INT32 作为 INT4 的伪装

**原因**: torch 不支持 INT4 数据类型，因此使用 INT32 作为 INT4 的伪装。

**实现**:
- Host 侧：INT32 输出时，形状的最后一个维度是输入的 1/8
- Kernel 侧：INT32 和 INT4 使用相同的内存大小计算（都按字节计算）

### 2. 宏定义统一处理 sizeof

**原因**: `sizeof(int4b_t)` 在逻辑上是 0.5 字节，但 C++ `sizeof` 只能返回整数。

**解决方案**: 
- 定义 `MUL_SIZEOF(x, TYPE)` 和 `DIV_SIZEOF(x, TYPE)` 宏
- 对于 `int4b_t`，使用位运算（`>> 1` 和 `<< 1`）代替乘除法
- 统一替换所有 `sizeof(ExpandXOutType)` 的使用

### 3. Cast 自动处理打包

**原因**: AscendC 的 `Cast` 操作在目标类型为 `int4b_t` 时会自动处理打包。

**实现**: 直接使用 `Cast` 从 `half` 转换到 `int4b_t`，无需手动打包。

---

## 文件修改清单

### Host 侧
1. `op_host/moe_distribute_dispatch_v2_infershape.cpp`
   - `InferDataTypeMoeDistributeDispatchV2`: 支持 INT4 和 INT32 输出类型
   - `InferShapeMoeDistributeDispatchV2`: INT32 输出时调整形状

2. `op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp`
   - `CheckTensorDataType`: 添加 INT4 和 INT32 数据类型检查及维度约束
   - `CheckWinSize`: 调整内存大小计算，支持 INT4 和 INT32

### Kernel 侧
1. `op_kernel/moe_distribute_dispatch_v2.h`
   - 添加 `MUL_SIZEOF` 和 `DIV_SIZEOF` 宏定义
   - 修改 `hOutSize_`、`hAlignWinCnt_`、`axisHCommu` 计算
   - 修改 `hCommuCopyOutParams_`、`expandXCopyParams_` 设置
   - 修改 `QuantProcess` 函数，支持 INT4 量化

2. `op_kernel/moe_distribute_dispatch_v2_full_mesh.h`
   - 添加 `MUL_SIZEOF` 和 `DIV_SIZEOF` 宏定义
   - 修改相关大小计算和 DataCopy 操作
   - 修改 `QuantProcess` 函数，支持 INT4 量化

---

## 测试建议

1. **数据类型测试**:
   - 测试 INT4 输出（`DT_INT4`）
   - 测试 INT32 输出（`DT_INT32`，实际是 INT4）
   - 测试 INT8 输出（`DT_INT8`，确保不影响原有功能）

2. **维度约束测试**:
   - INT4：输入的最后一个维度能被 2 整除
   - INT32：输入的最后一个维度能被 8 整除，且是输出的 8 倍

3. **量化精度测试**:
   - 验证 INT4 量化的精度是否符合预期
   - 验证 INT4 最大值限制（2.56）是否正确

4. **内存布局测试**:
   - 验证 Window 内存布局是否正确
   - 验证数据打包是否正确（2个INT4打包成1个字节）

---

## 注意事项

1. **INT4 打包**: 2个INT4打包成1个字节（`uint8_t`），8个INT4打包成1个INT32
2. **内存对齐**: INT4 输出时，对齐要求与 INT8 相同（32字节对齐）
3. **最大值限制**: INT4 量化需要限制最大值为 2.56（原因需要确认）
4. **Cast 自动打包**: `Cast` 操作会自动处理 INT4 的打包，无需手动处理

---

## 参考实现

- `ops-nn/quant/dynamic_quant/op_kernel/dynamic_quant.h`: INT4 量化实现参考
- `ops-nn/quant/dynamic_quant/op_host/dynamic_quant_infershape.cpp`: INT4/INT32 输出支持参考

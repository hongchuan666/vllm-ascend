# moe_distribute_dispatch_v2 内存布局视图

## 1. 常量定义

```cpp
constexpr uint32_t UB_ALIGN = 32U;           // UB缓冲区按32字节对齐
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;  // Window内存按512字节对齐
constexpr uint32_t EXPAND_IDX_INFO = 3U;    // 三元组：3个int32_t (12字节)
constexpr uint32_t BUFFER_NUM = 2;           // 多缓冲区数量
constexpr uint64_t SPLIT_BLOCK_SIZE = 512UL; // 分块大小（Window中）
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U; // 分块数据大小（480B + 32B对齐）
constexpr uint64_t WIN_STATE_OFFSET = 500UL * 1024UL;  // Window状态区偏移
constexpr uint64_t STATE_WIN_OFFSET = 950UL * 1024UL;   // 状态Window偏移
```

## 2. Window内存布局（通信窗口）

### 2.1 Window整体结构

Window内存分为两个通信域：
- **EP (Expert Parallel)**: `COMM_EP_IDX = 0`
- **TP (Tensor Parallel)**: `COMM_TP_IDX = 1`

每个Window分为前后两半区，用于连续两次dispatch的切换：
```
┌─────────────────────────────────────────────────────────┐
│                    Window内存区域                        │
├─────────────────────────────────────────────────────────┤
│  前半区 (dataState_ = 0)                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Combine数据区                                     │   │
│  │ axisMaxBS_ * (axisK_ + sharedExpertNum_) *      │   │
│  │ hSizeAlignCombine (对齐512B)                      │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Dispatch数据区                                    │   │
│  │ axisMaxBS_ * moeExpertNum_ * hAlignWinSize_      │   │
│  └──────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  后半区 (dataState_ = 1)                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Combine数据区                                     │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Dispatch数据区                                    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 2.2 单个Token在Window中的布局

每个token在Window中的存储结构（对齐到512B）：

```
┌─────────────────────────────────────────────────────────┐
│  Token数据 (hOutSizeAlign_)                             │
│  - 对齐到32B                                             │
│  - INT4: axisH_ / 2 字节                                │
│  - INT8/FLOAT16: axisH_ * sizeof(ExpandXOutType) 字节   │
├─────────────────────────────────────────────────────────┤
│  Scale数据 (32B)                                        │
│  - 1个float (4B)                                         │
│  - 对齐到32B                                             │
├─────────────────────────────────────────────────────────┤
│  三元组 (12B)                                           │
│  - rank_id (4B int32_t)                                 │
│  - token_id (4B int32_t)                                │
│  - topk_id (4B int32_t)                                 │
├─────────────────────────────────────────────────────────┤
│  填充到512B对齐                                          │
└─────────────────────────────────────────────────────────┘
```

**计算过程：**
```cpp
hOutSize_ = (INT4) ? (axisH_ >> 1) : (axisH_ * sizeof(ExpandXOutType))
hOutSizeAlign_ = Ceil(hOutSize_, UB_ALIGN) * UB_ALIGN;  // 对齐到32B
hScaleSizeAlign = hOutSizeAlign_ + UB_ALIGN;             // +32B (scale)
hScaleIdxSize = hScaleSizeAlign + EXPAND_IDX_INFO * sizeof(int32_t);  // +12B (三元组)
hAlignWinSize_ = Ceil(hScaleIdxSize, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;  // 对齐到512B
```

### 2.3 Window中的分块存储（Full Mesh模式）

**重要说明**: `SPLIT_BLOCK_SIZE` 和 `SPLIT_BLOCK_DATA_SIZE` **仅在 Full Mesh 模式下使用**。

在Full Mesh模式下，每个token按512B分块存储，每块包含：
- **数据部分**: 480B (SPLIT_BLOCK_DATA_SIZE)
- **标志部分**: 32B (SPLIT_BLOCK_SIZE - SPLIT_BLOCK_DATA_SIZE) - 用于异步通信状态检查

**为什么只在Full Mesh模式下使用？**

1. **异步通信需求**: Full Mesh模式使用异步通信，需要检查每个数据块是否已到达。每块的32B标志区域用于存储通信状态（如float值1.0表示数据已到达）。

2. **普通模式**: 普通模式使用同步通信（AlltoAll），不需要状态检查，直接对齐到512B即可，无需分块。

3. **分块优势**: 
   - 支持细粒度的数据到达检查
   - 可以部分读取已到达的数据块
   - 提高通信效率，减少等待时间

```
┌─────────────────────────────────────────────────────────┐
│  Block 0 (512B)                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Token数据 (480B)                                  │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ 对齐/标志 (32B)                                    │   │
│  └──────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  Block 1 (512B)                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Token数据 (480B)                                  │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ 对齐/标志 (32B)                                    │   │
│  └──────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  ...                                                    │
└─────────────────────────────────────────────────────────┘
```

**计算：**
```cpp
hOutSizeAlign_ = axisHExpandXAlignSize_ + UB_ALIGN * BUFFER_NUM;  // +64B
blockCntPerToken_ = Ceil(hOutSizeAlign_, SPLIT_BLOCK_DATA_SIZE);   // 向上取整到480B的倍数
hCommuSize_ = blockCntPerToken_ * SPLIT_BLOCK_SIZE;                // 总大小 = 块数 * 512B
expertPerSizeOnWin_ = axisMaxBS_ * hCommuSize_;                    // 每个expert在Window中的大小
```

### 2.4 Window状态区布局

```
┌─────────────────────────────────────────────────────────┐
│  Window状态区 (STATE_SIZE = 1MB)                        │
├─────────────────────────────────────────────────────────┤
│  偏移 0: 状态数据                                        │
│  - 每个expert的状态信息                                  │
│  - 对齐到32B                                            │
├─────────────────────────────────────────────────────────┤
│  偏移 WIN_STATE_OFFSET (500KB): 状态Window               │
│  - 用于通信的状态信息                                    │
├─────────────────────────────────────────────────────────┤
│  偏移 STATE_WIN_OFFSET (950KB): 状态Window               │
│  - 额外的状态信息                                        │
└─────────────────────────────────────────────────────────┘
```

## 3. UB (Unified Buffer) 临时缓冲区布局

### 3.1 Token数据缓冲区 (xQueue_ / xOutQueue_)

**大小**: `hOutAlignUbSize_ = Ceil(hScaleIdxSize, UB_ALIGN) * UB_ALIGN`

**布局：**
```
┌─────────────────────────────────────────────────────────┐
│  Token量化数据 (hOutSizeAlign_)                         │
│  - 对齐到32B                                            │
│  - INT4: axisH_ / 2 字节                                │
│  - INT8/FLOAT16: axisH_ * sizeof(ExpandXOutType) 字节  │
├─────────────────────────────────────────────────────────┤
│  Scale数据 (32B)                                        │
│  - 1个float (4B)                                        │
│  - 对齐到32B                                            │
├─────────────────────────────────────────────────────────┤
│  三元组 (12B)                                           │
│  - rank_id (4B)                                         │
│  - token_id (4B)                                        │
│  - topk_id (4B)                                         │
├─────────────────────────────────────────────────────────┤
│  填充到32B对齐                                          │
└─────────────────────────────────────────────────────────┘
```

**注释说明：**
- `xQueue_`: 非量化场景使用，注释为 `// 7k*2 + 32 + 12`
- `xOutQueue_`: 量化场景输出队列，注释为 `// 7K * 2 + 32 + 6`（可能是旧注释）

### 3.2 输入数据缓冲区 (xInQueue_)

**大小**: `hAlignSize = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN`

**布局：**
```
┌─────────────────────────────────────────────────────────┐
│  输入Token数据 (对齐到32B)                              │
│  - axisH_ * sizeof(XType) 字节                          │
│  - 对齐到32B                                            │
└─────────────────────────────────────────────────────────┘
```

### 3.3 其他UB缓冲区

#### 3.3.1 状态缓冲区 (statusBuf_)
- **大小**: `statusBufCntAlign * UB_ALIGN`
- **用途**: 保存发送数据量及flag，用于计算Window中的偏移
- **对齐**: 32B

#### 3.3.2 Expert IDs缓冲区 (expertIdsBuf_)
- **大小**: `expertIdsBufSize = max(expertIdsSize, bsAlign256)`
- **用途**: 存储expert IDs
- **对齐**: 32B

#### 3.3.3 量化相关缓冲区
- **receiveDataCastFloatBuf_**: 接收数据转换为float
- **smoothScalesBuf_**: 平滑后的scales
- **rowMaxBuf_**: 行最大值（动态量化用，32B）

#### 3.3.4 其他临时缓冲区
- **gatherMaskTBuf_**: 收集掩码
- **waitStatusBuf_**: 等待状态缓冲区
- **sumCoreBuf_**: 核间求和缓冲区 (aivNum_ * 32B)
- **sumLocalBuf_**: 本地求和缓冲区 (aivNum_ * 32B)
- **scalarBuf_**: 标量缓冲区 (96B = 32B * 3)

## 4. 内存对齐规则总结

### 4.1 UB缓冲区对齐
- **对齐单位**: 32B (UB_ALIGN)
- **所有UB缓冲区**: 必须对齐到32B
- **Token数据**: 对齐到32B
- **Scale数据**: 对齐到32B

### 4.2 Window内存对齐
- **对齐单位**: 512B (WIN_ADDR_ALIGN)
- **Token起始地址**: 对齐到512B
- **Combine数据**: 对齐到512B

### 4.3 数据类型大小
- **INT4**: 逻辑上0.5字节，实际打包为2个INT4 = 1个uint8_t
- **INT8**: 1字节
- **FLOAT16**: 2字节
- **BFLOAT16**: 2字节
- **INT32**: 4字节
- **FLOAT32**: 4字节

## 5. 内存布局示例

### 5.1 示例：FLOAT16输入，INT4输出

假设 `axisH_ = 7168` (7K):

**输入 (xInQueue_):**
- 大小: `7168 * 2 = 14336` 字节
- 对齐后: `Ceil(14336, 32) = 14336` 字节（已对齐）

**输出 (xOutQueue_):**
- Token数据: `7168 / 2 = 3584` 字节
- 对齐后: `Ceil(3584, 32) = 3584` 字节（已对齐）
- Scale: 32字节
- 三元组: 12字节
- 总大小: `3584 + 32 + 12 = 3628` 字节
- 对齐后: `Ceil(3628, 32) = 3648` 字节

**Window中:**
- 有效数据: 3628字节
- 对齐后: `Ceil(3628, 512) = 4096` 字节

### 5.2 示例：Full Mesh模式分块

假设 `hOutSizeAlign_ = 7000` 字节:

**分块计算:**
- `blockCntPerToken_ = Ceil(7000, 480) = 15` 块
- `hCommuSize_ = 15 * 512 = 7680` 字节
- 每块: 480B数据 + 32B对齐/标志

## 6. 关键变量说明

| 变量名 | 说明 | 计算方式 |
|--------|------|----------|
| `hOutSize_` | Token数据原始大小 | INT4: `axisH_ >> 1`<br>其他: `axisH_ * sizeof(ExpandXOutType)` |
| `hOutSizeAlign_` | Token数据对齐后大小 | `Ceil(hOutSize_, UB_ALIGN) * UB_ALIGN` |
| `hScaleSizeAlign` | Scale起始偏移 | `hOutSizeAlign_ + UB_ALIGN` |
| `hScaleIdxSize` | 包含三元组的总大小 | `hScaleSizeAlign + 12` |
| `hAlignWinSize_` | Window中对齐后大小 | `Ceil(hScaleIdxSize, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN` |
| `hOutAlignUbSize_` | UB缓冲区大小 | `Ceil(hScaleIdxSize, UB_ALIGN) * UB_ALIGN` |
| `blockCntPerToken_` | 分块数量 | `Ceil(hOutSizeAlign_, SPLIT_BLOCK_DATA_SIZE)` |
| `hCommuSize_` | 通信大小 | `blockCntPerToken_ * SPLIT_BLOCK_SIZE` |
| `expertPerSizeOnWin_` | 每个expert在Window中的大小 | `axisMaxBS_ * hCommuSize_` |

## 7. 内存复用策略

1. **量化场景**: `dstExpBuf_` 和 `subExpBuf_` 复用 `receiveDataCastFloatBuf_` 和 `smoothScalesBuf_`
2. **多缓冲区**: 使用 `BUFFER_NUM = 2` 实现流水线处理
3. **动态调整**: 根据UB使用情况，动态选择单缓冲区或多缓冲区模式

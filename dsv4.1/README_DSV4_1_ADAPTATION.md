# DeepSeek V4.1 支持与适配工作说明

## 工作概述

本次工作围绕 DeepSeek V4.1 的缓存结构变化、新版 vLLM KV Cache Layout，以及外部缓存命中后的恢复正确性，对 UCM 的 HMA/FAWA 路径进行了适配。主要覆盖模型路由、跨层共享与压缩缓存、量化数据存取、临时状态处理、续算与 replay、加载同步和缓存版本隔离。

本文合并介绍 `dsv4.1` 分支的两次改动，不按开发阶段拆分：

- `9da6f1d0`：`[Feat] Add DS V4.1 KV cache compatibility`。
- `7a750180`：`support vllm:v0.30.0`，补充新版统一布局及 replay 适配。

涉及 `deepseekv41-flash-0909` 临时镜像路径和后续统一缓存布局路径。下面的函数、字段及行为以这两个提交合并后的代码为准；版本兼容范围取决于实际缓存接口与布局，不以镜像版本号单独判断。

| dsv4.1变化             | 需要处理什么                                                |
| ---------------------- | ----------------------------------------------------------- |
| 压缩比例变化           | 重新计算逻辑 token 对应多少物理状态，以及地址偏移和传输长度 |
| KV / Indexer 跨层复用  | 按实际注册的缓存条目统计，替代固定层数公式                  |
| 临时状态与恢复要求变化 | 判断哪些组需要保存，哪些靠续算或 replay 重建                |
| 新版 vLLM 缓存布局变化 | 适配统一视图、value/scale 排列、整页 padding 和 placement   |

## 1. 跨层共享与缓存分组适配

**按照实际缓存组和物理状态计算存取量。**

在原有register_kv_caches中，补充了对 V4.1 缓存布局的解析：结合实际张量和逐层规格，正确描述压缩状态、量化 value/scale 及完整页的地址和传输大小。

- 基于 vLLM 注册的 `group.layer_names` 和 KV 张量建立存取描述，适应 KV 与 Indexer 的跨层复用；UCM 不按固定的模型层数或“每若干层复制一次”组织传输。
- 兼容 `compress_ratio` 和 `tokens_per_state` 两类描述，区分一个 block 覆盖的逻辑 token 数与实际保存的物理状态数。
- 从实际 spec/page 大小动态推导 FA、WA 的存储量及 4 KB 对齐后的文件大小，并检查 scheduler 推导结果与 worker 实际布局是否一致。

**修改原因：** 沿用 V4 的固定压缩比例、层数和文件大小，会造成逻辑 token、物理状态与实际传输字节数不匹配。

**主要位置：** `KVCacheGroupLayout`、`_group_tokens_per_state_by_layer`、`_init_group_metas`、`_derive_v41_file_sizes`、`register_kv_caches`。

## 2.新旧 KV Cache Layout 兼容

**概括：同时支持原有缓存视图和新版统一布局，确保量化数据、scale 与物理页内容完整对应。在原有布局解析框架上，补充主 KV 和 Indexer 的 packed 格式识别，为 value、scale 分别生成传输描述，复用已有存取流程。**

- 增加统一缓存视图的解析，明确 `[B, H, N, C]` 中的 `N` 表示物理状态数，不能直接当作逻辑 token block 大小，也不能误判成 K/V 维度。
- 利用 vLLM 提供的 `tokens_per_state`，按每层压缩比例解析实际缓存，适配新版 layout 的状态数和传输范围。
- 对 packed KV 建立 value、scale 两部分的地址和长度描述。补充主 KV 的 584/528/288 字节记录格式，以及 Indexer 的 68/132 字节格式，按实际 spec 选择对应布局。
- 在分段传输路径中分别计算 value、scale 的起始位置、状态步长及复制长度，避免仅根据张量表面 shape 推算连续内存范围。
- [value0 value1 value2 value3 | scale0 scale1 scale2 scale3] ,取部分状态时

**主要位置：** `KVCacheGroupLayout._build_layout`、`extract_addrs_with_offsets`、`segment_tensor_size_list`、`_group_physical_layout_by_layer`、`_v41_whole_page_sizes`、Store 创建逻辑。

## 3. 持久化范围调整

**概括：由 vLLM 的缓存组属性决定哪些状态需要存取，哪些状态应在续算时重建。**

- 根据 `prefix_cacheable=False` 或 `enable_kv_transfer=False` 识别无需持久化的组，跳过这些组的缓存注册、地址提取和 Store 数据传输。
- 对被排除的组保留原始 group ID 和占位元数据，维持 scheduler、worker 的 block 列表索引一致，避免过滤后发生组错位。
- 对仍可持久化的 SWA/window 组继续保存恢复所需的窗口尾部，不能仅因它是滑窗缓存就直接排除。
- 支持新版配置中没有持久化窗口组的情况：不创建 WA Store，lookup、prefetch、load、dump 均允许只处理 FA Store。

**修改原因：** V4.1 的临时 compressor/ring 状态不能一概视为可复用的持久化缓存；同时，旧版流程默认 FA、WA Store 必须同时存在，无法覆盖新版由 replay 重建窗口状态的配置。

**主要位置：** `_init_group_metas`、`_create_wa_store`、`_lookup_external_hit_blocks`、`_select_group_blocks`，以及 load/dump 流程。

## 4. 外部命中后的续算与 replay 适配

**概括：让缓存命中边界、模型恢复位置和实际计算进度保持一致。replay 期间不能把“命中到哪里”当成“实际算到哪里”，只有算过命中边界并产生新的完整块，才计划保存。**

- 根据各组压缩粒度计算恢复对齐要求，并校验 hash block 能否覆盖完整的压缩组。
- 当 prompt 长度恰好对齐完整 hash block 时，在 lookup 阶段保留最后一个完整块用于计算，避免加载块末状态后又退回该块内部续算。
- 读取新版 `prefix_replay_tokens`，配合调度器通过重算恢复临时状态；如果命中长度完全落在 replay 范围内，则不将其作为有效复用前缀。
- 构建 dump 计划时，引入 scheduler 提供的实际 `num_computed_tokens`，避免把“外部命中长度”误当成“本步已经完成的计算位置”，在 replay 期间提前保存未生成的数据。
- 补充短窗口快照的边界处理，避免负索引取到错误块；对超过已分配 block 范围的窗口切片明确报错。

**修改原因：** 持久化缓存恢复后，临时状态可能仍需补算。只有续算边界和调度进度正确，外部命中才能等价于正常 prefill。

**主要位置：** `get_num_new_matched_tokens`、`_select_group_blocks`、`_generate_dispatch_meta`、`build_connector_meta`。

#### 示例：命中边界与 replay 计算进度

假设 prompt 至少有 320 个 token，外部记录的 hash block 大小为 64 token：

```text
外部缓存命中：前 256 token
需要 replay：命中前缀末尾的 128 token
实际重算起点：256 - 128 = 128
每步计算：64 token
```

此时需要区分两个位置：

- **命中边界为 256**：持久化的 FA 缓存已经覆盖 `[0, 256)`。
- **初始计算起点为 128**：模型需要重算 `[128, 256)`，重建未保存的状态，再继续计算后续 token。

UCM 根据 scheduler 提供的实际 `num_computed_tokens` 生成 dump 计划：

| 步骤   | 本步计算起点 | 本步计算范围   | 是否生成新 dump 计划                     |
| ------ | -----------: | -------------- | ---------------------------------------- |
| 第一步 |          128 | `[128, 192)` | 否，仍在已命中范围内重建状态             |
| 第二步 |          192 | `[192, 256)` | 否，刚恢复到命中边界                     |
| 第三步 |          256 | `[256, 320)` | 是，计划保存新完成的完整块`[256, 320)` |

如果错误地把命中边界 256 当成第一步的计算起点，就会按 `256 + 64 = 320` 计划保存 `[256, 320)`。但这一轮模型实际只计算到 192，待保存的数据还没有生成。

因此，**外部命中长度仍为 256；128 是 replay 的起点，不是修正后的命中长度。dump 计划必须跟随实际计算进度，只有越过命中边界并产生新的完整块后，才安排保存。**

## 5. 外部加载精度问题修复

**概括：修复模型执行流与外部加载流之间的竞争，避免加载的数据被后续设备写操作覆盖。**

- 在 V4.1 确实存在外部加载任务时，先执行设备同步，再提交 Store 的 H2D 数据复制。
- 确保之前排队的缓存页清零、计算或其他写操作结束后，外部数据才开始回写。
- 将同步限制在 V4.1 的实际外部加载路径；保留其他模型原有处理逻辑。

此前可能出现的时序：`清零/写操作尚未完成 → 外部 KV 回写 → 延迟写入覆盖 KV`。

修复后的时序：`已有设备写操作完成 → 外部 KV 回写 → 加载完成后续算`。

**修改原因：** 这种竞争不会必然触发报错，缓存命中率也可能正常，但恢复的数据已被破坏，最终表现为答案变化和评测分数下降。临时镜像上的外部命中精度问题已获得用户实测修复反馈。

**主要位置：** `UCMFAWAConnector.start_load_kv`。

## 代码范围与验证说明

| 文件                                        | 主要职责                                             |
| ------------------------------------------- | ---------------------------------------------------- |
| `ucm/integration/vllm/ucm_connector.py`   | Connector 选择顺序与 FAWA 路由                       |
| `ucm/integration/vllm/hma_connector.py`   | 分组、layout、地址与字节数、持久化、replay、加载同步 |
| `ucm/integration/vllm/request_hasher.py`  | 缓存格式隔离与两侧 key 一致性                        |
| `test/suites/Unit/test_ucm_v41_compat.py` | V4.1 适配及 V4 兼容性回归测试                        |

验证情况区分为两部分：临时镜像的外部命中精度问题已有用户实测修复反馈；新版 layout 的地址、字节数和状态流转有专门回归测试覆盖，其完整模型精度和性能仍以对应镜像上的实测结果为准。本文不将其他分支的测试数量或昇腾 HLA 适配结果计入 DS V4.1 的验证结论。

# DeepSeek V4.1 支持与适配工作说明

## 工作概述

本次工作围绕 DeepSeek V4.1 的缓存结构变化、新版 vLLM KV Cache Layout，以及外部缓存命中后的恢复正确性，对 UCM 的 HMA/FAWA 路径进行了适配。主要覆盖模型路由、跨层共享与压缩缓存、量化数据存取、临时状态处理、续算与 replay、加载同步和缓存版本隔离。

本文合并介绍 `dsv4.1` 分支的两次改动，不按开发阶段拆分：

- `9da6f1d0`：`[Feat] Add DS V4.1 KV cache compatibility`。
- `7a750180`：`support vllm:v0.30.0`，补充新版统一布局及 replay 适配。

涉及 `deepseekv41-flash-0909` 临时镜像路径和后续统一缓存布局路径。下面的函数、字段及行为以这两个提交合并后的代码为准；版本兼容范围取决于实际缓存接口与布局，不以镜像版本号单独判断。

## 1. 模型识别与 Connector 路由

**概括：识别 V4.1 的缓存特征，确保模型进入正确的 HMA/FAWA 存取路径。**

- 结合 `kv_source_layer_ids`、`index_source_layer_ids`、压缩配置和临时缓存组等信息识别 V4.1，避免只依赖某一种压缩比例判断模型类型。
- 在 `UCMConnector` 中先判断是否属于 FAWA，再对其他模型执行 HLA 布局探测。
- 避免 HLA 探测访问新版 `KVCacheTensor` 已不再提供的 `shared_by` 字段，也避免将 V4.1 的临时缓存组当成 HLA 的共享物理缓存区域。

**修改原因：** V4.1 的缓存组织和描述接口发生变化，Connector 的选择必须与实际缓存语义一致，否则还没进入存取流程就可能误判或报错。

**主要位置：** `ucm_connector.py` 中的 `UCMConnector.__init__`；`hma_connector.py` 中的 `UCMFAWAConnector._init_group_metas`。

## 2. 跨层共享与缓存分组适配

**概括：按照实际缓存组和物理状态计算存取量，适应跨层共享后“模型层数”和“需要保存的缓存数量”不再一一对应的情况。**

- 基于 vLLM 注册的 `group.layer_names` 和 KV 张量建立存取描述，适应 KV 与 Indexer 的跨层复用；UCM 不按固定的模型层数或“每若干层复制一次”组织传输。
- 兼容 `compress_ratio` 和 `tokens_per_state` 两类描述，区分一个 block 覆盖的逻辑 token 数与实际保存的物理状态数。
- 对统一布局按每层 spec 计算物理状态数，例如逻辑 block 为 64 token、`tokens_per_state=2` 时，对应 32 个物理状态；同一组中不同层分别计算自己的偏移和传输长度。
- 调用 vLLM 的 `resolve_kv_cache_block_sizes` 获取对齐信息：分段传输使用 hash 粒度，完整页传输使用 scheduler 对齐粒度。
- 从实际 spec/page 大小动态推导 FA、WA 的存储量及 4 KB 对齐后的文件大小，并检查 scheduler 推导结果与 worker 实际布局是否一致。

**修改原因：** 沿用 V4 的固定压缩比例、层数和文件大小，会造成逻辑 token、物理状态与实际传输字节数不匹配。

**主要位置：** `KVCacheGroupLayout`、`_group_tokens_per_state_by_layer`、`_init_group_metas`、`_derive_v41_file_sizes`、`register_kv_caches`。

## 3. 新旧 KV Cache Layout 兼容

**概括：同时支持原有缓存视图和新版统一布局，确保量化数据、scale 与物理页内容完整对应。**

- 增加统一缓存视图的解析，明确 `[B, H, N, C]` 中的 `N` 表示物理状态数，不能直接当作逻辑 token block 大小，也不能误判成 K/V 维度。
- 对 packed KV 建立 value、scale 两部分的地址和长度描述。补充主 KV 的 584/528/288 字节记录格式，以及 Indexer 的 68/132 字节格式，按实际 spec 选择对应布局。
- 在分段传输路径中分别计算 value、scale 的起始位置、状态步长及复制长度，避免仅根据张量表面 shape 推算连续内存范围。
- 对提供 placement 元数据的新版 `BLHNC/BLNHC` 布局，读取 `layers`、`layer_stride` 等信息；以张量实际地址和 block stride 定位，按完整 layer page 传输，包括页内 padding。
- 完整页路径禁止页内偏移加载和部分页传输，并检查状态数、页大小、stride、压缩比例的整除关系。当前统一视图要求连续的单头状态区域，不能把该实现理解为支持任意布局。
- 创建 Store 时移除会覆盖分段描述的旧 `tensor_size` 标量配置，确保实际使用 `tensor_size_list`。

**修改原因：** 逻辑张量形状不一定能完整描述底层量化存储。漏掉 scale、忽略 padding 或错误切分物理页，都可能在外部命中后恢复出错误缓存。

**主要位置：** `KVCacheGroupLayout._build_layout`、`extract_addrs_with_offsets`、`segment_tensor_size_list`、`_group_physical_layout_by_layer`、`_v41_whole_page_sizes`、Store 创建逻辑。

## 4. 持久化范围调整

**概括：由 vLLM 的缓存组属性决定哪些状态需要存取，哪些状态应在续算时重建。**

- 根据 `prefix_cacheable=False` 或 `enable_kv_transfer=False` 识别无需持久化的组，跳过这些组的缓存注册、地址提取和 Store 数据传输。
- 对被排除的组保留原始 group ID 和占位元数据，维持 scheduler、worker 的 block 列表索引一致，避免过滤后发生组错位。
- 对仍可持久化的 SWA/window 组继续保存恢复所需的窗口尾部，不能仅因它是滑窗缓存就直接排除。
- 支持新版配置中没有持久化窗口组的情况：不创建 WA Store，lookup、prefetch、load、dump 均允许只处理 FA Store。

**修改原因：** V4.1 的临时 compressor/ring 状态不能一概视为可复用的持久化缓存；同时，旧版流程默认 FA、WA Store 必须同时存在，无法覆盖新版由 replay 重建窗口状态的配置。

**主要位置：** `_init_group_metas`、`_create_wa_store`、`_lookup_external_hit_blocks`、`_select_group_blocks`，以及 load/dump 流程。

## 5. 外部命中后的续算与 replay 适配

**概括：让缓存命中边界、模型恢复位置和实际计算进度保持一致。**

- 根据各组压缩粒度计算恢复对齐要求，并校验 hash block 能否覆盖完整的压缩组。
- 当 prompt 长度恰好对齐完整 hash block 时，在 lookup 阶段保留最后一个完整块用于计算，避免加载块末状态后又退回该块内部续算。
- 读取新版 `prefix_replay_tokens`，配合调度器通过重算恢复临时状态；如果命中长度完全落在 replay 范围内，则不将其作为有效复用前缀。
- 构建 dump 计划时，引入 scheduler 提供的实际 `num_computed_tokens`，避免把“外部命中长度”误当成“本步已经完成的计算位置”，在 replay 期间提前保存未生成的数据。
- 补充短窗口快照的边界处理，避免负索引取到错误块；对超过已分配 block 范围的窗口切片明确报错。

**修改原因：** 持久化缓存恢复后，临时状态可能仍需补算。只有续算边界和调度进度正确，外部命中才能等价于正常 prefill。

**主要位置：** `get_num_new_matched_tokens`、`_select_group_blocks`、`_generate_dispatch_meta`、`build_connector_meta`。

## 6. 外部加载精度问题修复

**概括：修复模型执行流与外部加载流之间的竞争，避免加载的数据被后续设备写操作覆盖。**

- 在 V4.1 确实存在外部加载任务时，先执行设备同步，再提交 Store 的 H2D 数据复制。
- 确保之前排队的缓存页清零、计算或其他写操作结束后，外部数据才开始回写。
- 将同步限制在 V4.1 的实际外部加载路径；保留其他模型原有处理逻辑。

此前可能出现的时序：`清零/写操作尚未完成 → 外部 KV 回写 → 延迟写入覆盖 KV`。

修复后的时序：`已有设备写操作完成 → 外部 KV 回写 → 加载完成后续算`。

**修改原因：** 这种竞争不会必然触发报错，缓存命中率也可能正常，但恢复的数据已被破坏，最终表现为答案变化和评测分数下降。临时镜像上的外部命中精度问题已获得用户实测修复反馈。

**主要位置：** `UCMFAWAConnector.start_load_kv`。

## 7. 缓存隔离与回归验证

**概括：防止不同布局误用同一份外部缓存，并为核心存取行为补充回归覆盖。**

- 扩展 `RequestHasher` 的缓存命名空间，纳入 V4.1 布局版本、cache dtype、block size 和压缩配置。
- 进一步纳入缓存组组成、传输开关、replay 参数，以及物理 placement 的层列表、layer stride、block stride、offset，区分不兼容的缓存格式。
- 新版有 placement 时，不依赖 scheduler/worker 可能不同的合并后 spec 细节生成 key，避免两侧针对相同缓存布局生成不同 key。
- 补充量化 value/scale 往返、压缩状态寻址、整页大小、临时组过滤、短窗口和全命中边界、FA-only 存取、replay dump 位置、加载同步及 V4 原有行为的测试。
- 对无法解析的 block size、布局或传输字节数不一致等情况明确报错，避免静默使用猜测值继续存取。

**修改原因：** 同一个模型名并不意味着缓存格式相同。格式发生变化后，需要自动隔离旧数据；同时必须保证 scheduler 和 worker 对同一格式使用一致的缓存 key。

**主要位置：** `request_hasher.py` 中的 `RequestHasher`；`test/suites/Unit/test_ucm_v41_compat.py`。

## 代码范围与验证说明

| 文件 | 主要职责 |
| --- | --- |
| `ucm/integration/vllm/ucm_connector.py` | Connector 选择顺序与 FAWA 路由 |
| `ucm/integration/vllm/hma_connector.py` | 分组、layout、地址与字节数、持久化、replay、加载同步 |
| `ucm/integration/vllm/request_hasher.py` | 缓存格式隔离与两侧 key 一致性 |
| `test/suites/Unit/test_ucm_v41_compat.py` | V4.1 适配及 V4 兼容性回归测试 |

验证情况区分为两部分：临时镜像的外部命中精度问题已有用户实测修复反馈；新版 layout 的地址、字节数和状态流转有专门回归测试覆盖，其完整模型精度和性能仍以对应镜像上的实测结果为准。本文不将其他分支的测试数量或昇腾 HLA 适配结果计入 DS V4.1 的验证结论。

# DeepSeek V4.1 支持与适配：代码阅读详细版

## 阅读范围与版本基准


## 总览：这次适配解决了什么

| 层次     | 回答的问题                            | 对应实现                            |
| -------- | ------------------------------------- | ----------------------------------- |
| 逻辑前缀 | 命中了多少 token，在哪个边界恢复？    | hash block、压缩对齐、replay        |
| 缓存组   | 哪些物理缓存需要保存？哪些需要重建？  | FA/WA/transient 分类、原始 group ID |
| 物理布局 | 每个状态的值、scale、padding 在哪里？ | 每层 spec、stride、分段/整页传输    |
| 执行时序 | 何时允许外部 KV 写入并被模型使用？    | load 前设备同步、load 完成等待      |

跨层共享主要改变缓存所有者和存储量；压缩主要改变逻辑 token 到物理状态的映射；新 layout 改变物理地址和传输粒度；replay 改变有效恢复位置。这些变化不能用一个固定层数或固定文件大小统一处理。

### 整体调用关系

以下是 Connector 层的主干调用关系，省略错误处理、TP 分工和监控；不是完整 vLLM 调度器调用栈。

```text
初始化
UCMConnector.__init__
  └─ 优先选择 UCMFAWAConnector
       ├─ _init_group_metas
       │    ├─ 持久化组 / 临时组分类
       │    ├─ resolve_kv_cache_block_sizes
       │    ├─ 压缩恢复对齐 / replay 参数
       │    └─ _derive_v41_file_sizes
       ├─ RequestHasher(..., kv_cache_config=...)
       └─ worker: register_kv_caches
            ├─ KVCacheGroupLayout._build_layout
            ├─ _store_tensor_size_list
            └─ 创建 FA Store，以及按需创建 WA Store

一次命中后的执行
scheduler: get_num_new_matched_tokens
  ├─ 请求 token → canonical hashes
  ├─ _lookup_external_hit_blocks
  └─ 记录命中边界与请求元数据
scheduler: build_connector_meta
  └─ _generate_dispatch_meta
       └─ _slice_group_block_ids → 各组 load/dump block 列表
worker: start_load_kv
  ├─ V4.1 有 load 时：device.synchronize()
  ├─ _extract_fa_ptr → _submit_load_task(FA)
  ├─ 有持久化窗口组时：_extract_wa_ptr → _submit_load_task(WA)
  └─ _wait_all_load_task → 等待加载完成
模型按调度计划 replay / 续算
worker: wait_for_save
  ├─ _extract_fa_ptr → _submit_dump_task(FA)
  └─ 有持久化窗口组时：_extract_wa_ptr → _submit_dump_task(WA)
```

FA 表示保存连续前缀的全局/压缩注意力缓存；WA 表示恢复窗口边界所需的尾部快照。FAWA 是具体 Connector 实现，HMA 是这里承接混合缓存组管理的路径，不能与 HLA Connector 混为一谈。

## 1. 模型识别与 Connector 路由

### 1.1 先判断 FAWA，再执行 HLA 探测

**位置：UCM 第 3360—3379 行，`UCMConnector.__init__`。**

```python
is_fawa = UCMFAWAConnector.can_handle_kv_cache_config(kv_cache_config)
use_hybrid_linear_attention = False
if not is_fawa:
    use_hybrid_linear_attention = (
        UCMHybridLinearAttentionConnector.supports_kv_cache_layout(
            kv_cache_config
        )
    )
```

原有流程先调用 HLA 布局探测，再决定是否走 FAWA。这意味着即使最后会进入 FAWA，也可能已经在 HLA 探测阶段访问不兼容字段。

改动后，识别为 FAWA 的配置直接绕过 HLA 探测。原因有两个：新版物理张量描述使用 `layers`，不能假定还存在旧的 `shared_by`；V4.1 的临时组也不能直接按 HLA 的共享物理区域语义解释。

这里改变的是公共入口的判断顺序，**这两个提交没有修改 `hla_connector.py`**。原有 FAWA 类型判断见 HMA 第 632 行的 `can_handle_kv_cache_config`，GPU 路径检查 `SlidingWindowMLASpec`；它与下面的 V4.1 特征识别不是同一个判断。

### 1.2 在 FAWA 内识别 V4.1 特征

**位置：HMA 第 727—758 行，`_init_group_metas`。**

```python
self.is_v41_layout = bool(
    self.transient_group_ids
    or getattr(hf_config, "kv_source_layer_ids", None)
    or getattr(hf_config, "index_source_layer_ids", None)
)
```

随后还根据 `compress_ratios` 是否属于 `{0, 1, 2}` 且包含 `1` 或 `2` 补充识别。不能只检查是否出现比例 `2`，否则只通过 KV/Indexer 共享来源字段表达特征的配置会漏判。

这些判断决定是否启用动态字节数、V4.1 命中边界、整页布局及加载同步等行为，并不是在 UCM 内重新实现模型的跨层共享。

**测试入口：** TEST 第 524 行 `test_route_checks_fawa_before_hla`；第 841 行 `test_v41_model_features_select_resolved_hash_without_ratio_two`。

## 2. 跨层共享与缓存分组：按实际物理状态计算

### 2.1 不再用 V4 固定层数估算 V4.1 存储量

**位置：HMA 第 1188—1237 行 `register_kv_caches`；第 932 行 `_specs_for_group`；第 944 行 `_derive_v41_file_sizes`。**

注册时仍以 vLLM 提供的 `group.layer_names` 找到真实缓存张量；新增的布局描述附着在这些张量上。scheduler 的大小计算也逐个读取组内层的 spec；若存在 `kv_cache_specs`，则按层名取各自的 spec。

因此“多个模型层复用一个 KV”的处理边界是：vLLM 提供缓存组、缓存所有者和张量视图，UCM 根据注册结果构造传输描述。**代码没有硬编码每 10 层保存一次，也没有新增按 `data_ptr` 自动去重的机制。**

V4 的固定层数与文件大小公式仍保留在 HMA 第 886—929 行；V4.1 在第 874—884 行走动态计算并提前返回，避免套用旧公式。

### 2.2 区分逻辑 token 数与物理状态数

**位置：HMA 第 417 行 `_group_tokens_per_state_by_layer`；第 207—231 行 `_build_layout`；第 803—810 行 `_init_group_metas`。**

统一视图为 `[B, H, N, C]`，当前支持的传输视图要求 `H=1`。其中 `N` 是物理状态数，不一定等于逻辑 block 覆盖的 token 数。

```python
expected_states = self.expected_block_size // tokens_per_state
if physical_states != expected_states:
    raise ValueError(
        f"Standardized KV cache state count mismatch for "
        f"{layer_name}: N={physical_states}, expected "
        f"{expected_states} from block_size="
        f"{self.expected_block_size} and "
        f"tokens_per_state={tokens_per_state}."
    )
```

例如一个组的逻辑 block 为 64 token：

| 视图 | tokens_per_state | 物理状态 N | 逻辑偏移 32 对应的物理偏移 |
| ---- | ---------------- | ---------- | -------------------------- |
| 层 A | 1                | 64         | 32                         |
| 层 B | 2                | 32         | 16                         |

同组不同层可以具有不同的物理状态数，因此不能给整个组套同一个物理偏移。初始化兼容旧 `compress_ratio` 和新 `tokens_per_state`，统一视图的每层寻址使用各层 `tokens_per_state`。

### 2.3 根据传输方式选择 block 对齐粒度

**位置：HMA 第 759—789 行，`_init_group_metas`。**

通过 `resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)` 获取 `(scheduler_block_size, hash_block_size)`，校验结果是两个正整数；没有 helper 或返回值不合法时明确报错，不继续沿用固定值。

```python
self.hash_block_size = int(
    resolved[0] if self.whole_page_sizes else resolved[1]
)
```

- 分段路径使用 vLLM 给出的 hash 粒度，可以将一个物理 block 中的合法分段作为外部缓存记录。
- 完整页路径使用 scheduler 对齐粒度，不能把带 padding 的整页再按更细 hash 拆开。
- 完整页 FA 组还要求 `token_block_size == self.hash_block_size`，校验位置为 HMA 第 812—817 行。

这里的 `self.hash_block_size` 是 **UCM 外部记录的逻辑 token 粒度**。整页路径中它被设为 scheduler 粒度，不应仅凭变量名理解为始终等于 vLLM 的细粒度 hash 大小。

### 2.4 scheduler 与 worker 各算一次，并检查一致性

**位置：HMA 第 944—1003 行 `_derive_v41_file_sizes`；第 1250 行 `_store_tensor_size_list`；第 1137—1155 行 `_create_store`。**

无整页 placement 时，scheduler 按以下优先级取 spec 的页字节数：`unpadded_page_size_bytes`、`real_page_size_bytes`、`page_size_bytes`。

公式示意：

```text
每层分段字节数 = spec 页字节数 × segment_tokens / spec.block_size
每组贡献       = 所有层分段字节数之和 × tail_blocks
每条外部记录   = 对该 Store 的组贡献求和，再向上对齐到 4096 字节
```

有整页 placement 时，直接累加每层 `layer_stride`，包含物理页 padding，不再通过 scheduler 合并后的代表 spec 猜大小。

worker 从真实张量的布局描述生成 `tensor_size_list`，再做 4 KB 对齐。若与 scheduler 的 `file_size[label]` 不一致，V4.1 路径直接报错。**物理页内 padding 与最终文件的 4 KB 对齐是两件不同的事，不能相互替代。**

**测试入口：** TEST 第 584 行压缩分段；第 631 行同组混合压缩率；第 823 行动态大小；第 884 行合并 spec 后整页大小一致性；第 1065 行九组拓扑。最后一项是构造的拓扑单测，不是运行真实模型得出的端到端结果。

## 3. 新旧 Layout：分段寻址与完整页传输

### 3.1 如何选择新布局

**位置：HMA 第 1197—1235 行，`register_kv_caches`。**

从 `kv_cache_tensors[*].layers` 收集统一视图层名，判断当前组是否需要 `standardized_layout`，再传入三个关键映射：

- `expected_tokens_per_state_by_layer`：每层的逻辑 token / 物理状态比例。
- `physical_layout_by_layer`：每层的 value 和 scale 字节数。
- `whole_page_sizes_by_layer`：有新版 placement 时，每层包含 padding 的整页大小。

旧布局的 3D、4D K/V 分离、5D 等分支继续保留。不是把所有张量统一 reshape 后复制，而是根据注册接口和布局分别解析。

### 3.2 packed KV 的 value 和 scale 必须分别定位

**位置：HMA 第 460—485 行 `_group_physical_layout_by_layer`；第 248—278 行 `_build_layout`。**

支持的 packed 字节描述如下，单位均为每个物理状态的字节数：

| 类型    | 总字节数 | value 字节数 | scale 字节数 |
| ------- | -------- | ------------ | ------------ |
| 主 KV   | 584      | 576          | 8            |
| 主 KV   | 528      | 512          | 16           |
| 主 KV   | 288      | 256          | 32           |
| Indexer | 132      | 128          | 4            |
| Indexer | 68       | 64           | 4            |

这些数值描述的是本实现识别的格式，不代表每种 dtype、硬件组合都已完成模型验证。主 KV 根据 spec 的 `cache_dtype_str` 和 `state_content_size_bytes` 选择；Indexer 根据层名及记录宽度识别。

关键不是“把 C 个字节当成一个 token 连续复制”。在这里适配的 packed 分段布局中，页内 value 区与 scale 区分别排列，代码为它们生成两个描述：

```python
descriptors = (
    (0, value_bytes, value_bytes),
    (physical_states * value_bytes, scale_bytes, scale_bytes),
)
```

三元组分别是 `(页内起始偏移, 每状态地址步长, 每状态传输字节数)`。

例如 `N=32`、value 为 576 字节、scale 为 8 字节，取从第 16 个物理状态开始的 16 个状态：

```text
value 起点 = 页首 + 16 × 576，长度 = 16 × 576
scale 起点 = 页首 + 32 × 576 + 16 × 8，长度 = 16 × 8
```

若直接从 `页首 + 16 × 584` 复制 `16 × 584`，就不能正确表达这两个区域。shape 看起来合理、复制也没有越界，并不代表数据恢复正确。

### 3.3 按每个视图换算地址和传输长度

**位置：HMA 第 361—405 行，`extract_addrs_with_offsets`、`segment_tensor_size_list`。**

```python
physical_token_offsets = (
    offsets[:, None]
    * self.tensor_block_sizes[None, :]
    // group_token_block_size
)
```

随后计算：

```python
return (
    block_ids[:, None] * self.block_strides[None, :]
    + physical_token_offsets * self.tensor_token_strides[None, :]
    + self.base_ptrs[None, :]
).astype(np.uint64, copy=False)
```

传输长度也按相同比例换算：

```python
tensor_tokens = (
    self.tensor_block_sizes * logical_tokens // group_token_block_size
)
return (self.tensor_sizes_per_token * tensor_tokens).tolist()
```

这里沿用了部分 `token` 命名，但在统一布局中，`tensor_block_sizes` 和相应的每 token 步长实际描述的是物理状态。地址和长度必须采用一致的换算，否则可能出现“读对起点、复制错长度”。

### 3.4 有 placement 的新版路径按完整 layer page 传输

**位置：HMA 第 444—457 行 `_v41_whole_page_sizes`；第 233—239、279—294 行 `_build_layout`。**

仅在具备 layout 和物理张量描述时启用整页大小解析，当前接受 `BLHNC`、`BLNHC`：

```python
sizes = {name: tensor.layer_stride for tensor in tensors for name in tensor.layers}
```

在布局构建时，整页描述优先于 value/scale 分段描述：

```python
if whole_page_size is not None:
    if whole_page_size % physical_states:
        raise ValueError("Whole KV page size must divide by N.")
    descriptors = ((0, 0, whole_page_size // physical_states),)
```

这里不是按“每个状态零步长”单独搬运，而是借用现有大小向量计算，最终一次搬运完整页。页首取 `tensor[0].data_ptr()`，跨 block 距离取 `tensor.stride(0) * tensor.element_size()`，整页长度取 `layer_stride`。

因此每层传输区域可以包含页内 value、scale 和 padding。物理 placement 的层偏移已体现在注册张量的地址中，代码没有再次手工叠加 `tensor.offset`；placement 元数据还用于缓存命名空间隔离，见第 7 节。

限制与保护：

- 视图必须是受支持的连续单头 `[B, 1, N, C]`，见 HMA 第 193—231 行。
- block 步长必须足以容纳页 payload，见第 279—287 行。
- 不允许页内非零起始偏移，见第 369—370 行。
- 不允许只传逻辑 block 的一部分，见第 400—401 行和第 964—966 行。
- 分段路径对 ROCm tiled Indexer 显式拒绝，见第 255—267 行；不能将本实现视为任意硬件布局的通用适配。

### 3.5 Store 必须使用分段大小向量

**位置：HMA 第 1141—1145 行，`_create_store`。**

```python
if self.is_v41_layout:
    # CacheStore prefers a scalar tensor_size over tensor_size_list.
    # Packed V4.1 pages need their per-segment sizes preserved.
    config.pop("tensor_size", None)
config["tensor_size_list"] = tensor_size_list
```

旧配置中的单个 `tensor_size` 会优先于大小列表。即使前面正确计算了不同层、value/scale 的传输长度，若不移除该标量，Store 仍可能按错误的统一长度搬运。

**测试入口：** TEST 第 164 行 packed 格式选择；第 214 行 value/scale 字节往返；第 499 行旧标量配置覆盖问题；第 536 行统一视图状态维度；第 884 行 padding 大小推导。

## 4. 持久化范围：哪些存，哪些靠恢复计算重建

### 4.1 以缓存属性为准，不按组数量或名字猜测

**位置：HMA 第 727—733、795—801 行，`_init_group_metas`。**

```python
self.transient_group_ids = {
    group_id
    for group_id, group in enumerate(groups)
    if not getattr(group, "enable_kv_transfer", True)
    or not getattr(group.kv_cache_spec, "prefix_cacheable", True)
}
```

这里 `transient_group_ids` 实际表示本实现排除传输的组集合，既可能是临时 ring/compressor 状态，也可能是新版明确不允许外部传输、交给 replay 重建的窗口状态。

对这些组保留 `KVCacheGroupMeta(group_id, block_size, 0, 0)`，不把它们从原始组序列中删除。worker 注册布局时跳过；生成 block 切片时返回空列表。

例如原始组为 `[FA, transient, WA]`，传输元数据仍应为 `[FA blocks, [], WA blocks]`，不能压成 `[FA blocks, WA blocks]`，否则 scheduler 与 worker 按 group ID 访问时会错位。

**相关位置：** HMA 第 1202—1204 行跳过注册；第 1463—1465 行 `_slice_group_block_ids` 返回空切片。概览中的 `_select_group_blocks` 在此提交的真实名称是 `_slice_group_block_ids`。

### 4.2 持久化窗口组仍保存尾部，不能一律跳过 SWA

**位置：HMA 第 802—841 行 `_init_group_metas`；第 1771 行 `_extract_wa_ptr`。**

未被排除的窗口组仍需保存恢复边界：`swa_cache` 保存滑窗尾部；其他窗口状态按既有 compressor 尾部规则计算 `tail_tokens`，再按 `tail_blocks` 构建 Store 行。

外部 load 对 FA 恢复所有命中的连续前缀，对 WA 只恢复最终命中边界的快照。这是复用的 FAWA 机制，不是新增“每个模型层分别 load”的协议。

### 4.3 没有持久化窗口组时，允许 FA-only

**位置：HMA 第 1027 行 `_create_wa_store`；第 1284 行 `_lookup_external_hit_blocks`；第 1308 行 `_prefetch_hit_key_hotness`；第 1822 行 `start_load_kv`；第 1894 行 `wait_for_save`。**

联动修改如下：

| 环节     | FA-only 行为                                      | 精确位置       |
| -------- | ------------------------------------------------- | -------------- |
| 创建     | 没有 window group 则返回`None`，不建 WA Store   | HMA 1034       |
| lookup   | FA 连续命中直接作为候选结果，不再要求 WA 边界文件 | HMA 1296       |
| prefetch | 跳过不存在的 Store                                | HMA 1333       |
| load     | 提交 FA 后跳过 WA                                 | HMA 1861       |
| dump     | TP 分工和单 rank 分支均跳过 WA                    | HMA 1948、2002 |

不能只把 WA Store 的创建关掉：其余环节若仍强制要求 WA，就会报错或将本应可用的 FA 前缀判成未命中。

**测试入口：** TEST 第 693 行保留组索引；第 865 行 replay 排除窗口快照；第 913 行 FA-only lookup/dump；第 118 行同时覆盖 FA-only load。

## 5. 外部命中后的续算与 replay

### 5.1 恢复位置必须落在完整压缩组边界

**位置：HMA 第 853—873 行，`_init_group_metas`。**

收集各组每层的 `tokens_per_state`，没有该字段时读取 `compress_ratio`，计算最小公倍数作为 `resume_token_alignment`。要求 `hash_block_size` 能被这个值整除。

原因是部分临时状态不保存。如果从压缩组中间恢复，后续计算可能需要缺失的前驱状态；仅有持久化 KV 并不足以恢复正确执行。

### 5.2 完整 prompt 命中时，预留最后一个完整 hash block

**位置：HMA 第 1376—1387、1415—1420 行，`get_num_new_matched_tokens`。**

```python
eligible_block_count = (
    (request.num_tokens - 1) // self.hash_block_size
    if self.is_v41_layout
    else len(canonical_hashes)
)
external_keys = canonical_hashes[wa_hbm_hit_block_num:eligible_block_count]
```

以 hash block 为 64 为例，在其他命中条件都满足时：

- prompt 为 256 token：最多选择前 3 个完整块，剩余最后 64 token 计算。
- prompt 为 257 token：前 4 个完整块都可参与 lookup，剩余第 257 个 token 计算。

旧的“先命中整个 prompt，再把命中 token 数减 1”可能造成加载的是完整块结束时的 WA 快照，却从该块内部重新执行。V4.1 在 lookup 候选阶段就排除最后一块，避免缓存状态边界与执行位置不一致。

这些例子只说明候选边界；实际是否命中仍受文件存在性、窗口快照、阈值和 replay 等条件影响。

### 5.3 replay 范围内的命中不能算有效复用

**位置：HMA 第 734—740 行读取 `prefix_replay_tokens`；第 1403—1410 行处理无效命中。**

```python
if (
    getattr(self, "prefix_replay_tokens", 0)
    and total_hit_block_num * self.hash_block_size <= self.prefix_replay_tokens
):
    # The scheduler discards hits that would be replayed in full.
    wa_hbm_hit_block_num = external_hit_blocks = total_hit_block_num = 0
    wa_computed_tokens = 0
```

UCM 读取 replay 要求并与调度结果配合；并没有在 Connector 里实现模型的 replay kernel。如果所有命中 token 最终都要重算，该候选不能作为有效前缀复用。

### 5.4 dump 使用实际计算进度，不能拿命中长度代替

**位置：HMA 第 1512 行 `_generate_dispatch_meta`，重点第 1564—1575 行；第 1605 行 `build_connector_meta`，重点第 1618—1622、1649—1653 行。**

新增可选 `computed_tokens` 参数：replay 开启时，传入 scheduler 对新请求或已缓存请求提供的 `num_computed_tokens`；非 replay 路径保留原有行为。

```python
computed_start_token = (
    req_meta.token_processed if computed_tokens is None else computed_tokens
)
computed_end_token = min(
    req_meta.num_token_ids,
    computed_start_token + new_tokens,
)
dump_start = max(
    req_meta.total_hit_block_num,
    computed_start_token // self.hash_block_size,
)
dump_end = computed_end_token // self.hash_block_size
```

例子与回归测试一致：外部命中 256 token、replay 128 token、每步调度 64 token。

| scheduler 的本步起点 | 本步计算后到达 | 是否产生新 dump                |
| -------------------- | -------------- | ------------------------------ |
| 128                  | 192            | 否，仍在已命中范围内 replay    |
| 192                  | 256            | 否，刚恢复到命中边界           |
| 256                  | 320            | 是，保存新完成的`[256, 320)` |

若错误地以命中长度 256 为第一步起点，就会在实际只计算到 192 时，提前计划保存 `[256, 320)` 的未完成数据。

### 5.5 短窗口快照与越界保护

**位置：HMA 第 1454—1510 行，`_slice_group_block_ids`。**

- 对比窗口长度更短的前缀，窗口索引下界裁到 0，重复首块填满固定形状的快照，避免 NumPy 负索引绕到尾部。
- 对超出请求已分配 block 范围的索引显式报错，避免构造无效 WA 地址。
- block-wise 和 chunk-wise 两种 WA 快照分支都处理了边界。

**测试入口：** TEST 第 716 行完整 prompt 命中；第 796 行短窗口；第 965 行 replay 期间 dump 进度；第 1028 行非法 resolver 返回值。

## 6. 精度修复：在外部 KV 回写前建立设备执行顺序

### 6.1 改动位置与实际调用顺序

**位置：HMA 第 1822—1884 行，`start_load_kv`，新增同步在第 1827—1832 行。**

```python
if self.is_v41_layout and any(
    request.load_keys for request in metadata.request_meta.values()
):
    # Finish pending KV page writes before the store starts its H2D copies.
    self.device.synchronize()
    logger.info_once("V4.1 compute stream synchronized before external KV load")
```

后续调用仍沿用 Store 任务提交与等待：

```text
start_load_kv
  → device.synchronize()                  新增：完成此前设备工作
  → _extract_fa_ptr / _extract_wa_ptr      计算目标 KV 地址
  → _submit_load_task                    HMA 1662
      → _rank_consistency.submit_load    向 Store 提交加载
  → _wait_all_load_task                  HMA 1886
      → _wait_load_task                  HMA 1696
          → _rank_consistency.wait_load  确认加载完成
  → 返回模型执行路径
```

### 6.2 为什么只有 load 完成等待还不够

外部 H2D 与先前模型执行流上的页初始化、清零或其他写操作，如果没有正确建立跨流先后关系，就可能写入同一块缓存。

错误时序示意：

```text
原执行流：   已提交页写操作 ────────────── 实际写入
外部加载流：          H2D 回写 KV → 完成
结果：                                   已加载 KV 又被覆盖
```

加载后等待，只能保证外部任务完成，不能补上“此前设备写操作必须先结束”的约束。此次修复在提交 H2D 前执行设备同步，建立以下顺序：

```text
此前设备写操作全部结束 → 外部 KV 回写 → 等待加载完成 → 模型使用 KV
```

这种数据竞争可能不越界、不抛异常，lookup 也照常命中，所以“没有 error、命中率很高、输出没有乱码”并不能排除 KV 被破坏。

此前用户反馈临时镜像上添加同步后精度问题得到解决；这支持该加载时序修复对已复现场景有效。**本文没有新增设备 trace，不将具体某个清零 kernel 的执行顺序写成重新实测的结论，也不把所有 layout 错误归结为同一个原因。**

### 6.3 影响范围与代价

- 仅 `is_v41_layout=True` 且本批次存在非空 `load_keys` 时执行新增同步；没有外部 load 时不执行。
- 非 V4.1 模型保持此前路径。这表示本次未改变它们，不代表从理论上证明其他模型永远不存在同类竞争。
- `device.synchronize()` 是较强的正确性保护，会等待设备已有工作，可能限制计算与传输重叠；这两个提交没有将其优化成精细的 event/stream 依赖。
- 本文目标版本的 FAWA `start_load_kv` 仍在返回前等待 load 完成，不应与另一分支的 request-level step-wise 异步返回机制混淆。

**测试入口：** TEST 第 118 行 `test_v41_external_load_waits_for_compute_before_submitting_copies`，通过事件列表检查同步先于 FA/WA 提交，并检查非 V4.1、无命中、FA-only 分支。该单测验证调用顺序，不等于真实 GPU 竞争复现测试。

## 7. 缓存隔离与回归验证

### 7.1 缓存 key 必须描述实际存储格式

**位置：HASH 第 65—145 行，`RequestHasher.__init__`。**

原有模型名、TP、dtype、rank 等命名空间不足以区分此次格式变化。新增 V4.1 布局信息：

```python
layout_info = (
    ":kv_layout=v41_packed_v3"
    f":cache_dtype={cache_config.cache_dtype}"
    f":block_size={cache_config.block_size}"
    f":compress_ratios={ratios}"
)
```

当传入 `kv_cache_config` 时，继续加入 `transfer_layout=v4` 描述：各组 block size、`prefix_cacheable`、`prefix_replay_tokens`、`enable_kv_transfer` 和层信息。

有 placement 时还加入：

```python
placement = tuple(
    (tuple(t.layers), t.layer_stride, t.block_stride, t.offset)
    for t in tensors
)
layout_info += f":placement={placement!r}"
```

这样相同 token 前缀在不兼容布局下生成不同的缓存命名空间，避免读入旧格式文件。这是通过 key 隔离，不是删除或迁移已有缓存文件。

### 7.2 scheduler 与 worker 必须生成一致的布局描述

**位置：HASH 第 99—136 行；HMA 第 580—589 行。**

无 placement 时，key 包含每层 `tokens_per_state`、状态宽度、dtype 和量化模式等细节。有 placement 时，组内只记录排序后的层名，再使用稳定的物理 placement 描述布局。

原因是 scheduler 可能把一个组内的多个 spec 合并成代表 spec，worker 却保留逐层 spec。若两侧直接对各自的 spec 序列化，即使实际存储格式一致，也可能出现“worker 保存了，scheduler 却始终找不到”的情况。

HMA 初始化完成 V4.1 分组后，重新用 `kv_cache_config` 构建 `RequestHasher`，scheduler 更新 `_seed`，随后重新绑定 request block hasher。否则仅修改 hasher 类，实际 Connector 仍可能使用初始化早期的旧命名空间。

HASH 的 V4.1 判断使用模型共享来源字段和压缩配置；HMA 的布局判断还包含临时组条件，两处判断并非完全相同。本文按现有实现描述，不将其扩大解释为任意混合模型的统一格式识别。

**测试入口：** TEST 第 406 行 `test_request_hasher_versions_v41_packed_layout_only`，覆盖命名空间变化及两侧布局描述一致性。

### 7.3 回归覆盖与验证边界

**测试文件：`test/suites/Unit/test_ucm_v41_compat.py`。**

| 行号          | 测试关注点                                |
| ------------- | ----------------------------------------- |
| 118           | load 前同步与提交先后顺序                 |
| 164、214      | packed 格式选择、value/scale 字节往返     |
| 406、499、524 | key 隔离、Store 配置优先级、FAWA 路由     |
| 536、584、631 | 统一视图 N 维、压缩状态数、逐层偏移       |
| 693、716、796 | 临时组索引、完整命中边界、短窗口          |
| 823、841      | 动态文件大小、特征识别                    |
| 865、884、913 | replay 排除组、padding 大小、FA-only 存取 |
| 965、1028     | replay dump 进度、非法 resolver 结果拒绝  |
| 1045、1065    | V4 原有 hash/文件大小、V4.1 九组拓扑      |

这些测试主要以 AST 提取目标方法、构造 spec/张量替身及内存字节复制来检查行为，不依赖完整模型推理。它们能发现地址、长度、边界和调用顺序问题，但不能替代真实镜像上的 ARC 精度、长上下文、TP 以及性能测试。

在已检出目标提交、且解释器具备测试依赖的环境中，可执行：

```bash
/root/vllm/.venv/bin/python \
  /root/unified-cache-management/test/suites/Unit/test_ucm_v41_compat.py -v
```

该命令供后续复核；本次工作是文档整理，没有切换工作分支，也没有重新运行模型或单元测试。新版 layout 的完整模型精度与性能仍需以对应运行环境实测为准，不能仅凭镜像版本号或单测存在就视为已经验证。

## 建议的源码阅读顺序

1. **UCM 3360**：先确认模型为什么进入 FAWA，而不是 HLA。
2. **HMA 715**：看组分类、hash 粒度、压缩对齐及动态文件大小的入口。
3. **HMA 1188 → 417 / 444 / 460 → 154**：看真实张量如何变成布局描述。
4. **HMA 361 / 393 → 1250 → 1122**：看地址、大小向量和 Store 配置如何对应。
5. **HMA 1344 → 1512 → 1605**：看命中边界、replay 与 dump 计划。
6. **HMA 1822 → 1662 / 1696 → 1894**：看实际加载、同步、等待及保存。
7. **HASH 65 → TEST 对应测试**：最后检查格式隔离和边界回归。

一句话总结：**这次改动将 V4.1 的外部缓存恢复落实到实际组、实际状态数、实际页布局和实际计算进度，并补上外部加载前的设备同步；不是仅修改 block size，也不是在 HLA 上重新实现 V4.1。**

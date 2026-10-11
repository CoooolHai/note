# Step-wise 相比 Layer-wise 的优势：结合当前分支分析

仓库：`/Users/chyanmio/unified-cache-management`。  
当前分支：`step_wise`。  
当前提交：`99601e34da57fb8f9bfa735b58488fc7c915b848`。  
整理日期：2026-10-02。

本次在源码中没有找到名为 `StepWiseConnector` 的类，但找到当前 `step_wise` 分支及其请求级异步加载实现。因此，本文将 step-wise 理解为：**按请求提交完整的外部 KV 加载，让该请求等待加载，而 vLLM 在后续推理 step 中继续执行其他就绪请求，加载结束后再把该请求加入计算。** 如果你指的是另一个方案，需要按它的实际搬运时机重新比较。

此前的 [Layer-wise 分析](./ucm_layerwise_vllm_analysis.md) 说明了逐层流程；本文重点回答当前 step-wise 在什么条件下会更有优势，以及如何证明优势。

## 1. 最有价值的优势是什么？

**你这套方案最值得强调的是多请求服务中的计算与加载解耦：一个新请求正在搬 KV，其他已经就绪的请求仍可以继续生成 token。**

其次是普通 Direct 路径可以减少逐层任务提交、事件和状态管理开销，并降低对模型逐层 hook 的依赖。当前分支把这种请求级生命周期扩展到 HLA 和 FA/WA，是实际的覆盖范围改进。

不能直接据此说 step-wise 在所有指标上都优于 layer-wise。它通常更适合比较在线服务的吞吐、已有请求的 TPOT 和尾延迟；单个新请求的 TTFT、低并发下的延迟，以及缓冲区需求，要另外比较。

| 维度 | Step-wise 的潜在优势 | 成立条件或边界 |
| --- | --- | --- |
| 已有请求的 decode | 等待新请求 KV 的工作可以移出已有请求的模型计算批次 | 有其他就绪请求；Scheduler 能继续推进；资源竞争可控 |
| Store 任务管理 | 普通 Direct 可一次提交请求所需的全部层数据 | 提交更粗不等于 DMA 次数或字节数同幅减少 |
| 模型适配 | 加载不必靠每层 Attention hook 才能推进 | 仍需正确的布局、状态拷贝顺序和完成通知 |
| 混合 KV 布局 | 各组可以独立加载，在请求边界汇合 | HLA、FA/WA 已做相应实现；不等于任意布局自动支持 |
| 策略扩展 | 请求是自然的限流、优先级和取消单位 | 当前代码有 pending 管理，专门的带宽/QoS 控制仍需实现 |
| 可观测性 | 可以按请求追踪提交、完成、失败和加载耗时 | 新的 `direct_*` 指标主要针对普通 Direct |

## 2. 先明确两种方案分别在重叠什么

### Layer-wise：同一个请求内部，各层加载与计算重叠

```text
请求 B 进入本轮模型计算
  → 等待 B 的第 0 层 KV
  → 加载 B 的第 1 层，同时计算第 0 层
  → 加载 B 的第 2 层，同时计算第 1 层
  → ……
```

如果 A 和 B 在同一计算批次中，当前层需要等待 B 的数据时，A 的这一层也无法独立越过这个 hook 继续完成该批次。Layer-wise 可以隐藏一部分搬运时间，但当前层加载不足时仍会成为批次的等待点。

### Step-wise：一个请求的加载，与其他请求的计算重叠

```text
step k：
  提交 B 的外部 KV 加载
  B 等待加载，不参与模型 forward
  A、C、D 继续计算

step k+1：
  B 还没加载完
  A、C、D 继续生成 token

step k+2：
  Worker 观察到 B 加载完成，返回完成通知

后续调度：
  B 变成就绪请求，加入某个计算批次
```

实际的完成通知时机、轮询和调度可能增加一轮或更多等待，上面的 step 编号只是示意。当前代码也不保证“一轮一定加载完”，更不是预测所有请求下一个 token 的 KV 再提前搬运。

**两者寻找的是不同的重叠窗口：layer-wise 用当前请求尚未完成的层计算；step-wise 用其他请求的计算。**

## 3. 当前实现的函数调用关系

普通模型通常需要配置：

```yaml
use_layerwise: false
use_request_async_load: true
```

对于 HLA，关闭 layer-wise 可以避免默认选到 `UCMHybridLinearAttentionLayerWiseConnector`；选择非逐层的 `UCMHybridLinearAttentionConnector` 后才能启用这里的请求级异步路径。FA/WA 有单独的模型布局选择分支，也实现了请求级异步加载。

`use_request_async_load` 默认是 `False`。只有代码明确允许的类才会开启：普通 `UCMDirectConnector`、非逐层 HLA connector、FA/WA connector。普通 Layer-wise 和 CP 没有因此自动变成请求级异步加载。

### 3.1 查询命中时，告诉 Scheduler 这个请求要等待加载

```text
get_num_new_matched_tokens(request, num_computed_tokens)
  → 查询外部缓存命中前缀
  → 返回 (external_hit_tokens, load_async=True)
  → vLLM 使用等待外部 KV 的请求状态管理这次加载
```

这里第二个返回值是真正的“请求级异步加载标志”。它允许请求的搬运计划与请求的模型计算分开调度。

源码：[普通前缀查询](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:1813)、[能力开关](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:1376)。

### 3.2 分配显存后，立即形成 load-only 计划

```text
update_state_after_alloc(request, blocks, num_external_tokens)
  → 根据命中前缀边界选出源缓存块和目标设备块
  → RequestDispatchMeta(
       load_block_ids=(源缓存 ID, 目标物理块编号),
       dump_block_ids=([], []),
       load_async=True,
     )
  → _pending_async_load_dispatches[request_id] = 搬运计划
  → _async_load_req_ids.add(request_id)
```

`build_connector_meta()` 会把这些 pending 计划加入输出，**即使对应请求不在本轮的 scheduled forward 请求列表中**。

这就是 step-wise 能跨请求重叠的关键。只把 `store.wait()` 删掉，并不足以实现这个效果；还需要 Scheduler 先分配目标块、传递独立加载计划，并让该请求等待正确的完成通知。

源码：[分配后生成计划](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:1978)、[合并独立加载计划](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2083)。

### 3.3 Worker 提交请求的全部层数据，不在本轮等待

普通 Direct 路径：

```text
start_load_kv()
  → extract_block_addrs(vllm_block_ids)
  → 把各层张量地址展平成每个块的一行
  → shard_indexs = [0] × 块数
  → _rank_consistency.submit_load(...)
  → _pending_load_tasks[request_id] = PendingLoadTask(...)
  → 本轮继续执行其他就绪请求的 forward
```

Direct 布局把一个 token 块的各层数据放到一个存储 shard 中，所以这里 `shard_index=0` 不是只读取模型第 0 层，而是读取该 token 块的完整 Direct shard。

源码：[普通加载提交](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2218)。

### 3.4 后续 step 轮询完成，通知 Scheduler 放行

```text
get_finished()
  → _poll_pending_load_tasks()
    → check_load(task)
    → 没完成：保留任务
    → 已完成：wait_load(task)，消费句柄并检查延迟错误
    → 清理 pending
    → 返回已结束的 request_id
  → finished_recving
  → Scheduler 收到完成通知
  → 请求具备重新进入模型调度的条件
```

失败也必须报告终止状态，并同时带回无效块和失败信息，否则请求可能一直停在等待外部 KV 的状态。当前普通路径的轮询实现已经考虑这一点。

多个 worker/rank 的完成通知需要按引擎的协议聚合，不能在某一个 rank 完成后就假定整个请求已经可计算。仓库的旧版本 aggregator 示例会累计所需 worker 的完成通知。

源码：[完成轮询](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2655)、[旧版本 Scheduler 等待恢复示例](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/patch/v0110/vllm/pc/v1/core/sched/scheduler.py:592)、[旧版本 worker 结果聚合示例](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/patch/v0110/vllm/pc/v1/distributed/kv_connector/utils.py:5)。这些旧版本文件用于说明协议，不能直接当作所有新版本的运行调用点。

## 4. 优势一：减轻新请求对已有请求 TPOT 的影响

TPOT 可以简单理解为“开始生成后，相邻输出 token 之间隔多久”。已有请求可能已经在稳定 decode，新请求突然带来大量外部 KV 加载，会影响整批计算的节奏。

Layer-wise 中，如果新请求进入同一个批次，每层都可能出现加载不足的等待。Step-wise 将新请求先放在等待状态，已有请求可以在加载期间继续调度。

示意：A 正在生成，B 的 KV 加载需要 24 ms，正常 decode step 需要 6 ms。

```text
在资源允许且引擎持续推进的理想情况下：

时间           0       6       12      18      24 ms
A 等请求计算   [step 0][step 1][step 2][step 3]
B 的 KV 加载   [-------------------------]
```

这个例子说明的是已有请求可以继续推进，不是保证加载请求 B 的 TTFT 减少了 24 ms。B 自己仍然等待完整的所需 KV，只是尽量不让 A 为它一起等待。

这类优势在以下场景更明显：持续有 decode 请求，新进入的请求前缀很长，缓存命中但数据搬运较慢，而且设备有足够空间让待加载请求与运行请求共存。

隔离也有边界：KV 搬运仍然会占用 PCIe/NVLink、设备内存带宽、CPU 或存储资源；Scheduler 的 prefix lookup 和计划生成也没有因此全部变成异步。它减少的是一类显式等待，不能保证已有请求完全不受影响。

**最适合验证的指标：已有 decode 请求的 p95/p99 TPOT，以及插入大前缀请求前后的 TPOT 波动。**

## 5. 优势二：减少普通路径的任务提交和控制开销

设需要加载的请求数是 R，本地 Attention 层数是 L：

| 操作 | 普通 Layer-wise | 普通 Direct 请求级异步 |
| --- | --- | --- |
| Store load 提交 | 通常约 R × L 个任务 | 通常约 R 个任务 |
| 加载推进 | 依赖每层的 wait/load 推进 | 依赖跨 step 的请求完成轮询 |
| 每批 Store dump 提交 | 有保存需求时通常每层一个，共约 L 个 | 本轮 forward 后通常合并成一个 |
| 保存事件 | 按成功提交的层记录 | 普通 Direct 的合并 dump 记录一个 |

例如 64 层、8 个需要加载的请求，普通 load 任务数可以从约 512 个降到约 8 个。减少的是 Python/Store 任务提交及其上下文数量，不是把所需 KV 字节数缩小为 1/64。

请求级完成轮询也有开销：一个很慢的加载可能在很多 step 中反复被 check。不能把整个生命周期的所有调用一概视为从 R×L 降到了 R。

控制开销减少在以下情况下更有意义：单层数据较少、模型层数较多、CPU 或任务队列成为瓶颈，以及后端存在明显的每次提交固定成本。

对于 `Cache|Posix`，普通 Direct 的大 shard 还可能减少分片文件 I/O 的固定开销。不过大 Store 任务内部仍可能遍历很多设备张量、执行多次拷贝；实际聚合效果取决于后端与拷贝方式，必须实测。

**最适合验证的指标：Store 提交次数、任务队列等待、CPU 使用量、H2D 有效带宽和端到端吞吐。**

源码：[Layer-wise 每请求每层提交](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2823)、[Direct 请求提交](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2218)、[Direct 合并保存](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2497)。

## 6. 优势三：更少依赖模型内部逐层 hook

普通 Layer-wise 的推进依赖：所有需要搬运的层在正确时机经过 Attention 的 wait/save hook。如果某个模型用了自定义 Attention 而漏掉 hook，就可能无法正确等待和推进下一层加载。

请求级异步加载把加载的关键逻辑放在 model runner 与 connector 生命周期边界：forward 前提交，后续轮询完成，Scheduler 放行后再计算这个请求。普通 Direct 的 `wait_for_layer_load()` 和 `save_kv_layer()` 是空操作。

因此可以减少：

- 为每个自定义 Attention 补加载推进 hook 的工作。
- 推测解码、回滚或重复访问某层时，围绕逐层任务句柄的管理。
- 图执行重放绕过 Python 逐层逻辑时的接入限制。

这不代表当前代码把 vLLM 中所有逐层 helper 调用都删除了，也不保证所有模型的 FULL CUDA Graph 都已经通过验证。优势是关键的加载状态机不再依赖每层 hook；布局、图执行、状态拷贝和模型适配仍需要实际检查。

普通 Direct 在全命中处理上通常保留 1 个重新计算的 token；普通 Layer-wise 使用至少 2 个，投机解码时还会增加。它是另一个具体差异，但不是主要吞吐收益来源；混合模型需要遵循自己的命中边界逻辑。

源码：[Direct 逐层空接口](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2395)、[全命中保留 token](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:1547)。

## 7. 优势四：当前提交扩大了混合模型的可用范围

分支相对本地 `develop` 的新增提交为 `99601e3`。提交描述和差异显示：已有普通 Direct 请求级异步加载来自较早的 `e80ff8b`（PR #1246）；当前新增工作主要将其扩展到 HLA、HMA/FAWA，并增加可观测性。需要区分原有能力与本次新增能力。

### HLA：把加载安排到请求边界，减少按层组就绪的耦合

当前新增路径主要是：

```text
UCMHybridLinearAttentionConnector.get_num_new_matched_tokens()
  → 返回请求异步标志
update_state_after_alloc()
  → 记录各 KV group 的真实物理块
  → _generate_hla_dispatch_meta(new_tokens=0, need_load=True)
  → 加入 load-only pending 计划
start_load_kv()
  → 按 MLA / 线性状态等的 rank scope 提交数据
  → 异步任务加入 _pending_load_tasks
后续使用父类的完成轮询
```

混合 Layer-wise 需要解释 row、在哪个 Full Attention hook 等下一 row、线性状态必须在何时就绪；请求级方案在本次加载所需数据都就绪后才让请求计算，可以减轻这部分模型执行顺序与加载状态机的耦合。

**仍有同步边界：HLA 的 `start_load_kv()` 在提交加载前执行 `device.synchronize()`，用于确保已有 Mamba 状态拷贝不会覆盖新加载数据。** 所以当前实现不是“所有同步全部消失”。

源码：[HLA 能力开关](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/hla_connector.py:759)、[HLA 分配后计划](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/hla_connector.py:1077)、[HLA 加载](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/hla_connector.py:1397)。

### FA/WA：不同任务在请求边界汇合

当前 FA/WA 实现会分别提交：

```text
FA 数据：加载所有外部命中的必要前缀块
WA 数据：加载最终命中边界的必要窗口尾部
  → 两类任务加入 FAWAPendingLoad
  → 轮询，全部任务达到终止状态后报告请求完成
```

这个结构自然适合不同数据组独立加载，在请求边界汇合；还处理了部分提交失败后已经发出的任务继续收尾，以及取消请求时不能过早释放还在被加载任务写入的显存。

FA 加载前缀、WA 只加载必要尾部的“少搬数据”收益属于 FA/WA 布局策略，本来就存在；当前请求级异步扩展的新增收益，是把这些加载与其他请求计算重叠。不能把两者算成同一个优化的收益。

此外，FA/WA 的保存收尾仍有自己的等待规则，例如为防止窗口块被复用而 drain 上一步的 dump。请求级异步加载不等于 FA/WA 整条执行链没有阻塞。

源码：[FA/WA 分配后计划](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/hma_connector.py:990)、[FA/WA 提交与轮询](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/hma_connector.py:1377)、[取消与完成收尾](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/hma_connector.py:1756)。

## 8. 优势五：请求更适合后续限流、优先级与观测

当前代码已经维护 pending 请求、目标块和加载任务。以后可以在此基础上做：

- 限制正在加载的请求数和总 KV 字节，给运行请求保留显存与带宽。
- 小请求优先，或给长时间等待的请求提升优先级。
- 在 decode 压力高时降低加载并发，优先保障 TPOT。
- 按请求记录“何时提交、何时加载完成、何时重新调度”。

这些是结构上的扩展机会，**不能说当前提交已经实现了专门的请求优先级或带宽自适应控制**。现有 vLLM 和后端仍有自己的队列及限制，但这次差异中没有新增一套完整的 pending 字节预算/QoS 算法。

当前分支增加的普通 Direct 指标包括：

| 指标 | 可以回答什么 |
| --- | --- |
| `direct_async_load_dispatched_total` | 提交了多少次请求异步加载尝试？ |
| `direct_async_load_completed_total` | 有多少任务成功结束？ |
| `direct_async_load_failed_total` | 有多少任务提交或完成时失败？ |
| `direct_async_load_pending` | 这个 worker 还有多少 pending 请求加载？ |
| `direct_async_load_duration_ms` | 从提交到被观察到完成/失败，经过多久？ |
| `direct_load_bytes`、`direct_load_tokens` | 每次提交的 payload 大小？ |
| `direct_step_interval_ms` | 调度侧相邻活动 metadata 构建之间隔多久？ |

`direct_step_interval_ms` **不是 GPU forward 时间**，可能含调度、通信甚至某些空闲区间。`direct_async_load_duration_ms` 包含轮询观察延迟，也不是纯传输时间。

`_direct_metrics_enabled` 只为具体的 `UCMDirectConnector` 开启，所以不能直接用这些指标代表 HLA 或 FA/WA 的全部新路径。普通 worker 计数还需要分清 rank，不能把多卡求和的任务数直接当用户请求数。

源码：[Direct 观测与 step interval](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2159)、[指标定义](/Users/chyanmio/unified-cache-management/examples/metrics/metrics_configs.yaml:199)。

## 9. 哪些场景不容易获得优势？

### 9.1 单请求或很低并发

如果只有请求 B，没有其他请求可计算，B 的加载没有跨请求重叠对象。它需要先等所有必要层数据就绪，然后才开始计算；Layer-wise 可以先加载首层，利用 B 自己的后续计算隐藏其他层加载。

因此，单请求 TTFT 常常是 layer-wise 更有潜力的指标。不过，如果逐层任务开销很大，整体加载更快，step-wise 仍可能胜出；必须用同样的模型、缓存命中、payload 和设备环境比较。

### 9.2 显存与中转缓冲区紧张

Step-wise 在加载前通过 `update_state_after_alloc()` 获得目标显存块。待加载请求尚未产出 token，已经占用一部分设备缓存容量。pending 多了，可能减少正在运行的请求数、造成抢占，或使加载与计算互相争资源。

普通 Direct 的存储 shard 包含一个 token 块的全部层数据，主机中转缓冲的一次分配粒度通常也更大。Layer-wise 按层 shard 处理，可能更容易利用较小缓冲做流水。不同后端会有差异，不能仅从异步标签推断内存更省。

已有 layer-wise 也需要 vLLM 分配目标 KV 块。真正的额外压力主要在于 step-wise 允许多个尚未计算的请求跨 step 持有这些块，而不是它首次引入了 KV 分配。

### 9.3 完成通知粒度较粗

传输在两个轮询点之间完成，Scheduler 要到 Worker 下一次上报、引擎聚合并处理后才知道。如果某轮计算很长，即使 KV 已经加载好，请求也可能暂时不能入批。

这个等待会影响新请求 TTFT。Layer-wise 的就绪检查就在使用当前层之前，不需要等整个请求跨轮完成通知才能启动首层。

### 9.4 没有外部加载，或者只有大规模重计算

纯 HBM 命中不需要 Store load；没有外部命中时，也没有可跨请求隐藏的加载时间。如果主要瓶颈是大 prefill 计算，step-wise 的异步加载无法单独解决已有 decode 被大计算批次拖慢的问题，还要看 chunked prefill 和调度策略。

### 9.5 PP / CP 等并行配置

当前 `UCMConnector` 明确拒绝 PP 启用且 `use_layerwise=False` 的组合；CP 也没有自动开启普通请求级异步能力。因此，不能把当前 step-wise 方案描述为已经覆盖所有并行部署。

## 10. 如何用实验验证你的优势？

建议至少比较这三组，才能区分“整体搬运粒度”与“请求级异步”的收益：

| 组别 | 配置 | 对比目的 |
| --- | --- | --- |
| A | `use_layerwise=true` | 逐层搬运基线 |
| B | `use_layerwise=false`、`use_request_async_load=false` | 整体加载但在当前执行中等待 |
| C | `use_layerwise=false`、`use_request_async_load=true` | 你的请求级 step-wise 路径 |

普通模型可以按以上配置直接区分；HLA 应核对实际选中的 connector，FA/WA 不能假定 A 会选到普通 `UCMLayerWiseConnector`。每轮实验记录日志中的实际类名和 effective load mode。

B 对 C 说明请求级异步是否有收益；A 对 C 才是完整策略的最终比较。模型、设备、命中前缀、到达率、批次预算、存储容量和后端等需要匹配，分别预热各组的缓存，避免把命中率差异算成搬运策略优势。

### 最值得做的四个负载

| 负载 | 要验证的判断 | 重点指标 |
| --- | --- | --- |
| 单个外部高命中请求 | 跨请求重叠缺失时的代价 | TTFT、总加载时间 |
| 稳定 decode + 插入长前缀命中请求 | 是否减少已有请求被加载牵连 | 已有请求 p95/p99 TPOT，新请求 TTFT |
| 持续多请求、不同并发 | 是否提高整体服务能力 | output tokens/s、requests/s、p99 TTFT/TPOT、实际运行 batch 大小 |
| 多个大 payload 同时 pending | 是否因内存占用或带宽竞争抵消收益 | 显存占用、抢占数、pending 字节、缓存队列等待 |

稳态吞吐可以用一个粗略判断理解：设新请求加载耗时为 T_load，同期有 T_other 的独立计算窗口，理论上需要暴露出来的加载空闲约为：

```text
max(0, T_load - T_other)
```

这只是系统忙闲重叠的估计，假设资源允许独立执行，并且没有计入通知延迟、排队和提交成本；它不是新请求 TTFT 的计算公式。新请求自身仍要等待加载。

最好增加两个时间戳：设备拷贝真正完成时刻，以及 Scheduler 让请求首次进入计算批次的时刻。两者差值可以区分“传输慢”与“已经传完但放行慢”。

对正确性还需覆盖：load-only step 没有计算请求时能否持续推进、多个 rank 的就绪聚合、加载失败后退出等待、取消时异步写入目标块不能过早复用，以及 FA 成功/WA 失败的部分完成情况。这些属于请求级异步生命周期的重要边界，功能测试通过后再判断性能。

## 11. 对这套方案的定位建议

可以这样描述它的价值：

> 将外部 KV 加载从模型逐层计算路径移到请求生命周期中，让待加载请求与已就绪请求分开调度，并用跨 step 的加载/计算重叠改善在线服务效率。

更具体的目标是：**在有足够独立计算、缓存命中和内存容量的在线负载下，降低新请求加载对已有 decode 的影响，同时减少普通路径的逐层控制开销。** 当前提交对 HLA、FA/WA 的扩展，使这个机制能覆盖更多实际模型。

现阶段最有力的证据会是：相同到达率和命中条件下，已有请求的 p99 TPOT 更稳；相同延迟目标下，可承受的请求到达率更高。单独证明 Store 带宽更高，还不足以证明在线服务更好。

## 12. 本次分析的证据范围

已核对当前分支相对本地 `develop` 的代码差异，以及普通、HLA、FA/WA 的请求异步提交和完成路径。尚未运行真实推理或性能实验，本文的性能优势是结合机制推导出的、带适用条件的判断。

仓库已有测试覆盖普通路径的 load-only dispatch、重新调度不重复加载、仅报告完成任务，以及失败后解除等待并标记无效块，见 [已有请求异步测试](/Users/chyanmio/unified-cache-management/test/test_ucm_connector_metrics.py:1588)。尝试执行该组测试时，当前 Python 环境缺少 `pytest`，因此本次没有取得测试通过结果；这些测试也不能替代 HLA/FAWA 的真实引擎与多卡验证。

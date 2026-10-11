# UCM / vLLM：scheduler 与 worker 之间的 metadata 如何传递

分析日期：2026-10-07。依据本机源码：

- UCM：`/Users/chyanmio/unified-cache-management`，`step_wise`，commit `99601e34da57fb8f9bfa735b58488fc7c915b848`。
- vLLM：`/Users/chyanmio/vllm`，`main`，commit `fa212b8dc5de5f0cd7b3171e7c7f584bb5a69c53`。

本文重点是当前克隆版本的 V1 执行流程，以及 UCM 的接入方式。没有展开兼容旧版 vLLM 的分支，也没有进行 GPU 实机运行验证。

## 1. 先直接回答你的问题

**是的，主要在每次调度 step 中传递。scheduler 把本次调度结果和 UCM 搬运计划交给 executor；executor 交给 worker。worker 执行后，把生成结果、KV 搬运完成情况和错误信息交回 scheduler。**

可以把它理解为：scheduler 发“工作单”，worker 回“执行回执”。

一次逻辑 step 的主线是：

```text
EngineCore.step()
  │
  ├─ scheduler.schedule()
  │    ├─ 决定本轮哪些请求计算多少 token、使用哪些 GPU block
  │    └─ UCM.build_connector_meta() 生成本轮 load / dump 工作单
  │
  ├─ executor.execute_model(SchedulerOutput)
  │    └─ worker 收到 SchedulerOutput，包括 kv_connector_metadata
  │         ├─ 更新本地请求状态和 block table
  │         ├─ 绑定 UCM metadata
  │         ├─ 发起 KV load、执行模型、处理 KV dump
  │         └─ 收集搬运完成 / 错误信息
  │
  ├─ 如果 execute_model() 返回 None：executor.sample_tokens(...)
  │    └─ 得到带 KVConnectorOutput 的 ModelRunnerOutput
  │
  └─ scheduler.update_from_output(本轮 SchedulerOutput, ModelRunnerOutput)
       └─ 更新请求状态，决定后续 step 是否可以计算、重算或释放 block
```

这里有三个时间点需要分开：

1. **计划下发**：本轮 `schedule()` 结束以后，执行模型以前。
2. **任务执行**：同步 load 在需要它的 forward 前开始；纯异步 load 可以在本轮 forward launch 后开始。
3. **完成上报**：worker 在执行流程收尾处检查任务，把已经观察到的完成情况随本轮结果返回。异步任务可能在后面的 step 才完成。

所以，“本轮传 metadata”不等于“本轮所有加载和保存都完成”。

源码：[EngineCore.step](/Users/chyanmio/vllm/vllm/v1/engine/core.py:630)、[调度器挂载 connector metadata](/Users/chyanmio/vllm/vllm/v1/core/sched/scheduler.py:1506)、[worker 的 connector 执行上下文](/Users/chyanmio/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:69)。

## 2. 先区分四种容易混淆的 metadata / 状态

| 名称 | 位于哪里 | 是否传给另一端 | 通俗解释 |
|---|---|---|---|
| UCM `RequestMeta` | scheduler 侧 connector | 不直接传 | scheduler 对请求的“长期账本”：命中多少 block，已经处理到哪里 |
| `SchedulerOutput` | scheduler 构建，worker 接收 | 向 worker 传 | 本次调度的总工作单，包括模型计算计划和 UCM 计划 |
| UCM `UCMConnectorMetadata` | 放在 `SchedulerOutput.kv_connector_metadata` 中 | 向 worker 传 | 本次 UCM 搬运计划 |
| UCM `UCMWorkerMetadata` | worker 构建，嵌入执行结果 | 向 scheduler 传 | 本次收集到的 UCM 错误和缓存一致性事件 |

另外，worker 的 `PendingLoadTask`、`PendingDumpTask` 是**执行中的本地任务对象**。它们保存 Store task、GPU block ID、计时信息或事件句柄，通常跨 step 存续，但没有作为调度 metadata 回传。

两端的 connector 是不同角色的实例。在多进程 executor 下，它们也处于不同进程。scheduler 修改自己的 `requests_meta`，不会直接修改 worker 的任务字典。

源码：[RequestMeta / RequestDispatchMeta](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:341)、[下发、待完成任务及回传类型](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:1307)。

## 3. scheduler → worker：传了哪些内容

### 3.1 外层：vLLM 的 SchedulerOutput

UCM 的 metadata 是这个对象中的一个字段，并非单独建立一条 scheduler → worker 的搬运 RPC。

```text
SchedulerOutput
  ├─ scheduled_new_reqs             新请求的完整初始化信息
  ├─ scheduled_cached_reqs          worker 已知请求的增量更新
  ├─ num_scheduled_tokens           每个请求本轮计算多少 token
  ├─ finished_req_ids               请求结束通知
  ├─ scheduled_spec_decode_tokens   推测解码计划
  ├─ scheduled_encoder_inputs       多模态 encoder 计划
  ├─ new_block_ids_to_zero          本轮需要清零的缓存 block
  ├─ kv_cache_block_copies          缓存 block 复制计划
  ├─ has_sync_kv_loads              本轮是否有同步 KV load
  └─ kv_connector_metadata
       └─ UCMConnectorMetadata
            ├─ request_meta
            └─ preempted_req_ids
```

外层重要字段如下；不是完整字段穷举。

| 字段 | 内容 / 作用 |
|---|---|
| `scheduled_new_reqs` | 每个新请求的 ID、prompt token、采样 / pooling 参数、按 KV group 划分的 block IDs、已计算 token 数、LoRA、多模态相关信息等 |
| `scheduled_cached_reqs` | 已知请求的 ID、新增 block、已计算 token 数、输出 token 数、恢复请求标记等；不同执行模式还使用相应 token 更新字段 |
| `num_scheduled_tokens` | `request_id → 本轮 token 数`；表示参加本轮模型计算的请求 |
| `total_num_scheduled_tokens` | 本轮计算的 token 总数，决定是否真的运行模型 forward |
| `finished_req_ids` | 前后轮之间产生的请求结束通知；worker 用它清理请求状态，connector 也可据此处理待完成保存 |
| `num_common_prefix_blocks` | 本轮共享前缀 block 的相关信息，用于相应优化 |
| `scheduled_spec_decode_tokens` | 本轮推测解码候选 token |
| `scheduled_encoder_inputs`、`free_encoder_mm_hashes` | 多模态 encoder 调度和缓存释放信息 |
| `new_block_ids_to_zero`、`kv_cache_block_copies`、`block_table_updates` | 本轮缓存清零、复制或 block table 更新指令；是否使用取决于执行模式 |
| `kv_connector_metadata` | UCM 等 connector 的本轮执行计划 |
| `has_sync_kv_loads` | 是否必须在本轮 forward 前启动 load，是**整个 step 的标志** |

`kv_connector_block_state` 是一个特殊例子：它供 scheduler 内部的 connector 读取 block 状态，包含本地解析能力。当前 scheduler 在发给 worker 前将它清掉，不能把它当作发送给 worker 的远程状态对象。

源码：[NewRequestData](/Users/chyanmio/vllm/vllm/v1/core/sched/output.py:43)、[CachedRequestData](/Users/chyanmio/vllm/vllm/v1/core/sched/output.py:139)、[SchedulerOutput](/Users/chyanmio/vllm/vllm/v1/core/sched/output.py:232)、[清理 scheduler-local 状态](/Users/chyanmio/vllm/vllm/v1/core/sched/scheduler.py:1527)。

### 3.2 内层：UCMConnectorMetadata

普通 Direct / LayerWise 路径使用：

```python
UCMConnectorMetadata(
    request_meta={
        request_id: RequestDispatchMeta(
            load_block_ids=(ucm_keys, gpu_destination_block_ids),
            dump_block_ids=(ucm_keys, gpu_source_block_ids),
            load_async=False,
        )
    },
    preempted_req_ids=set(),
)
```

| 字段 | 类型 | 作用 |
|---|---|---|
| `request_meta` | `dict[str, RequestDispatchMeta]` | 按请求 ID 查找本轮搬运计划 |
| `load_block_ids[0]` | `list[bytes]` | 要从 UCM 读取的缓存 key |
| `load_block_ids[1]` | `list[int]` | 数据应该写入的 GPU block ID |
| `dump_block_ids[0]` | `list[bytes]` | 保存到 UCM 时使用的缓存 key |
| `dump_block_ids[1]` | `list[int]` | 要从哪些 GPU block 读取数据 |
| `load_async` | `bool` | 该请求的 load 是否可以提交后跨 step 等待 |
| `preempted_req_ids` | `set[str]` | 被抢占的请求；worker 可在更新、复用 block 前处理这些请求的待完成 dump |

注意：`request_meta` 这个字段名容易造成误解。**字典的值是 `RequestDispatchMeta`，不是 scheduler 长期保存的 `RequestMeta`。**

下发的普通 UCM 计划不包含 KV tensor 的内容，不包含每个 block 的 CUDA 地址，也不包含 Store task 和 CUDA event 对象。worker 收到 block ID 后，使用自己的 `kv_cache_layout.extract_block_addrs()` 解析 GPU 地址，再调用 Store 完成实际搬运。

普通非 MLA 路径中，非零 TP rank 还会根据自己的 rank 对外部 key 做转换。因此，同一份逻辑计划可以广播给多个 worker，再由每个 worker 访问自己的缓存分片。

本文举例默认 `CP=1`，此时可以简单理解为 key 与 GPU block ID 对应。CP 扩展下一个 GPU block 可能对应多个外部 key；HLA / FAWA 也有额外布局规则，不能无条件对所有路径做一一 `zip`。

源码：[UCMConnectorMetadata](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:1307)、[worker 解析地址和提交加载](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2218)、[抢占处理](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2485)。

### 3.3 通俗例子：工作单和数据搬运是两件事

以下 key 名称是为了阅读方便，实际是 `bytes` 类型的缓存标识：

```python
request_meta = {
    "R": RequestDispatchMeta(
        load_block_ids=([b"h1", b"h2"], [3, 6]),
        dump_block_ids=([b"h3"], [8]),
        load_async=False,
    )
}
```

它表达的是：

```text
从 UCM 取 h1 → 放进 worker 自己 GPU 上的 block 3
从 UCM 取 h2 → 放进 worker 自己 GPU 上的 block 6
本轮计算使 block 8 对应的数据完整以后 → 使用 h3 作为 key 保存
```

metadata 本身只描述这件事。真正的 KV 内容沿 Store 的读写、CPU / GPU 搬运路径流动，不放进 scheduler 的工作单里。

同样，dump 计划可以在 forward 前制定：scheduler 根据“之前已计算多少 token + 本轮准备计算多少 token”，确定本轮结束后哪些完整 block 可以保存。worker 必须在相应计算和依赖完成后才能读这些 block，提前下发计划不代表提前读未完成的数据。

源码：[生成 load / dump 区间](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2027)、[Direct 保存路径](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2497)。

## 4. 这些 metadata 在 scheduler 的什么位置生成

函数关系：

```text
Scheduler.schedule()
  ├─ connector.get_num_new_matched_tokens(...)
  │    └─ UCM 查询外部命中，并记录 scheduler 本地 RequestMeta
  ├─ kv_cache_manager.allocate_slots(...)
  │    └─ 为请求分配 GPU block ID
  ├─ connector.update_state_after_alloc(...)
  │    └─ async load 路径先记录“key → 已分配 block”的待下发计划
  ├─ 创建 SchedulerOutput
  └─ _build_kv_connector_meta(connector, SchedulerOutput)
       └─ UCMConnector.build_connector_meta(...)
            └─ 具体 connector.build_connector_meta(...)
                 ├─ 为 scheduled_new_reqs 生成计划
                 ├─ 为 scheduled_cached_reqs 生成计划
                 ├─ 清理已结束请求的 scheduler 本地状态
                 ├─ 合并 _pending_async_load_dispatches
                 └─ 返回 UCMConnectorMetadata
```

最后 scheduler 执行：

```python
scheduler_output.kv_connector_metadata = meta
```

先分配 GPU block，再制定“往哪里写”，这是顺序上的关键。外部命中查询只回答“已有多少可复用数据”，不能在 GPU block 尚未分配时就完成目的地映射。

**你的 step-wise 路径还有一个关键设计：等待异步 load 的请求，不在本轮模型计算列表中，但它的搬运计划会通过 `_pending_async_load_dispatches` 合并进 metadata。**

因此，以下两组 ID 可以不同：

```text
num_scheduled_tokens.keys()          本轮参加模型计算的请求
kv_connector_metadata.request_meta   本轮有 UCM 搬运计划的请求
```

这使 worker 能一边计算 A、B，一边给暂时等待的 R 加载缓存。

源码：[Scheduler 分配后通知 connector](/Users/chyanmio/vllm/vllm/v1/core/sched/scheduler.py:1253)、[异步 load 分配后建计划](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:1978)、[build_connector_meta](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2083)、[wrapper 委托](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:3726)。

## 5. 实际如何跨进程传输

### 5.1 多进程 executor：Python 对象序列化 + 消息队列

当前 `MultiprocExecutor` 路径：

```text
EngineCore.step()
  └─ MultiprocExecutor.execute_model(scheduler_output)
       └─ collective_rpc("execute_model", args=(scheduler_output,))
            └─ rpc_broadcast_mq.enqueue(
                 ("execute_model", args, kwargs, output_rank)
               )
                 └─ MessageQueue.enqueue(): pickle 序列化
                      ├─ 本地普通消息：共享内存 ring buffer
                      ├─ 本地超大消息：共享内存通知 + socket 传数据
                      └─ 远端 reader：ZMQ socket

各 worker 进程：
WorkerProc.worker_busy_loop()
  └─ rpc_broadcast_mq.dequeue()
       └─ 反序列化出 RPC 方法名和参数
            └─ _execute_worker_rpc()
                 └─ worker.execute_model(scheduler_output)
                      └─ GPUModelRunner.execute_model(scheduler_output)
```

传输对象是包含 `SchedulerOutput` 的 RPC 参数包。这里不是每个请求单独调用一次 RPC，也不是每个层调用一次 RPC。

本地共享内存保存的是序列化消息；这不意味着 scheduler 和 worker 共同操作同一个 Python 字典。当前队列采用 `pickle`，并支持大 buffer 的额外传输。代码中的 `cloudpickle` 主要用于 RPC 方法本身是 callable 的情况；这里方法是字符串 `"execute_model"`。

以上描述的是**调度命令的传输**。模型的 TP 通信、KV Store 数据搬运使用各自机制，应与这条消息队列链路分开理解。整个 `SchedulerOutput` 在某些功能下可能带 prompt embedding 等 tensor；这里只能说普通 UCM 搬运计划不携带 KV 数据，不能泛化为整个调度对象绝不包含 tensor。

源码：[MultiprocExecutor.execute_model](/Users/chyanmio/vllm/vllm/v1/executor/multiproc_executor.py:340)、[collective_rpc](/Users/chyanmio/vllm/vllm/v1/executor/multiproc_executor.py:377)、[MessageQueue.enqueue / dequeue](/Users/chyanmio/vllm/vllm/distributed/device_communicators/shm_broadcast.py:834)、[worker RPC 循环](/Users/chyanmio/vllm/vllm/v1/executor/multiproc_executor.py:1032)、[GPU worker 转发给 model runner](/Users/chyanmio/vllm/vllm/v1/worker/gpu_worker.py:1318)。

### 5.2 单进程和 Ray：语义相同，传输实现不同

| executor | 传递方式 | 对 UCM 的影响 |
|---|---|---|
| `UniProcExecutor` | 直接调用同一进程中的 worker 方法 | 不需要上述 IPC 序列化，仍然通过 `SchedulerOutput` 传本轮计划 |
| `MultiprocExecutor` | RPC 参数包，经共享内存 / socket 队列广播 | 每个 worker 获得本轮计划并使用本地资源执行 |
| `RayDistributedExecutor` | 通过 Ray compiled DAG 提交 `(scheduler_output, grammar_output)` | 仍使用相同逻辑对象，但不能套用多进程 executor 的队列实现 |

当前 Ray 生成路径还可能先在 executor 的 `execute_model()` 中保存 `scheduler_output`，到 `sample_tokens()` 再提交 DAG。所以下发时点更准确地说是“schedule 后、该批次在 worker 上执行前”，不应把所有 executor 都描述成 `execute_model()` 一调用就已经通过同一种 IPC 发出。

源码：[UniProcExecutor.collective_rpc](/Users/chyanmio/vllm/vllm/v1/executor/uniproc_executor.py:90)、[Ray execute_model](/Users/chyanmio/vllm/vllm/v1/executor/ray_executor.py:405)、[Ray _execute_dag](/Users/chyanmio/vllm/vllm/v1/executor/ray_executor.py:450)。

## 6. worker 收到以后，在哪些时间点使用

### 6.1 先更新模型计算状态，再绑定 UCM 计划

当前 GPUModelRunner 的主要顺序是：

```text
GPUModelRunner.execute_model(SchedulerOutput)
  ├─ connector.handle_preemptions(kv_connector_metadata)
  │    └─ 在缓存 block 更新 / 复用前处理待完成保存
  ├─ _update_states(SchedulerOutput)
  │    ├─ 更新 worker 的请求缓存和 persistent input batch
  │    ├─ 更新 block table
  │    └─ 处理结束请求、缓存清零 / 复制等
  ├─ 准备 GPU 输入、构建 attention metadata / slot mapping
  └─ maybe_get_kv_connector_output(...)
       └─ _get_kv_connector_output(...)
            ├─ bind_connector_metadata(本轮 UCMConnectorMetadata)
            ├─ 根据 has_sync_kv_loads 决定 load 启动位置
            ├─ yield：执行模型 forward
            ├─ wait_for_save()，或延后 finalize
            ├─ get_transfer_results() / 错误 / 统计 / worker metadata
            └─ clear_connector_metadata()，或延后 finalize
```

`bind_connector_metadata()` 本质上将收到的对象保存到 worker connector 的 `_connector_metadata`，后续 load / save 方法从这里取本轮计划。`clear_connector_metadata()` 将该引用设为 `None`。

模型计算需要的 attention metadata 是 worker 根据本轮调度、本地请求状态和缓存布局构建的另一个对象，不等同于 scheduler 下发的 UCM metadata。

源码：[GPUModelRunner.execute_model](/Users/chyanmio/vllm/vllm/v1/worker/gpu_model_runner.py:4117)、[_update_states](/Users/chyanmio/vllm/vllm/v1/worker/gpu_model_runner.py:1187)、[_build_attention_metadata](/Users/chyanmio/vllm/vllm/v1/worker/gpu_model_runner.py:2252)、[绑定 / 清理 metadata](/Users/chyanmio/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:238)。

### 6.2 同步与异步：区别在消费时间，不在另发一份 metadata

当前 model runner 的判断是：

```python
start_after_forward = not scheduler_output.has_sync_kv_loads
```

| 本轮情况 | load 的启动位置 | 原因 |
|---|---|---|
| 有本轮 forward 需要的同步 load | forward 前调用 `start_load_kv()` | 当前计算必须先获得外部 KV |
| 没有同步 load，只有异步 load 或无 load | forward launch 后调用 `start_load_kv()` | 将加载提交的 CPU 开销移出当前 forward launch 的关键路径 |
| 本轮既有同步请求，也有异步请求 | 统一在 forward 前调用 `start_load_kv()`，内部按请求标志处理 | `has_sync_kv_loads` 是 step 级；`load_async` 是 request 级 |

“forward launch 后”并不保证 GPU 已经算完。是否与后续计算重叠，仍取决于 stream、事件、Store 和硬件资源。

**两种情况下，工作单都已经随本轮调度对象到达 worker。异步路径只是延后执行 `start_load_kv()`，没有在 forward 后让 scheduler 再发一次 metadata。**

LayerWise 的 load / save 则由层级 hook 使用同一份已绑定计划：到某一层时调用 `wait_for_layer_load(layer_name)`，该层产生 KV 后调用 `save_kv_layer(layer_name, ...)`。层名来自模型执行位置，不需要 scheduler 每层重新传一个 RPC。

源码：[启动时机判断](/Users/chyanmio/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:69)、[LayerWise 等待当前层](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2908)、[LayerWise 保存当前层](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2963)。

### 6.3 没有 token 计算的 step，也可以推进 KV 传输

如果 R 正在等外部 KV，并且这一轮没有其他请求可计算，`total_num_scheduled_tokens` 可以为 0。

当前 model runner 在启用 KV connector 时走 `kv_connector_no_forward()`，仍然进入 connector 执行上下文，提交需要的搬运、检查 pending task 并返回 KV 结果。这避免“所有请求都在等 load，因此不 forward；不 forward 又无人检查 load 完成”的循环。

这里说的是 engine 仍有请求 / 待处理工作时的零 token step。engine 完全没有请求时，会在 `step()` 入口直接返回，并不是无限发送空工作单。

源码：[零 token 分支](/Users/chyanmio/vllm/vllm/v1/worker/gpu_model_runner.py:4164)、[kv_connector_no_forward](/Users/chyanmio/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:24)。

## 7. worker → scheduler：回传哪些内容

### 7.1 回传对象的嵌套关系

```text
ModelRunnerOutput
  ├─ 请求 ID、采样 token、logprob 等模型结果
  └─ kv_connector_output: KVConnectorOutput
       ├─ finished_recving
       ├─ finished_sending
       ├─ invalid_block_ids
       ├─ failed_recving
       ├─ kv_connector_stats
       ├─ kv_cache_events
       ├─ expected_finished_count
       └─ kv_connector_worker_meta: UCMWorkerMetadata
            ├─ is_mla
            ├─ load_failed_reqs
            ├─ missing_reqs
            ├─ missing_blocks
            └─ dump_succeeded_blocks
```

注意 `finished_recving`、`finished_sending`、`invalid_block_ids` 等是 vLLM 的 `KVConnectorOutput` 字段，**不在 UCMWorkerMetadata 内部**。`recving` 是源码里的实际拼写。

| vLLM 回传字段 | 含义 |
|---|---|
| `finished_recving` | KV 接收 / 加载流程已结束的请求；还应结合错误信息判断是否成功 |
| `finished_sending` | 满足 connector 完成条件的发送 / 保存请求；scheduler 可以据此处理此前保留的 block |
| `invalid_block_ids` | load 失败影响的 GPU block ID，供 scheduler 调整重算位置 |
| `failed_recving` | 按请求 ID 表达的接收失败；新版增加了对混合、多池布局更明确的错误身份表达 |
| `kv_connector_worker_meta` | connector 自定义信息，这里是 UCMWorkerMetadata |
| `kv_connector_stats`、`kv_cache_events` | 统计和缓存事件；是否实际产生由 connector 决定 |
| `expected_finished_count` | 一条请求需要收齐多少份完成通知；默认 0 表示不更新当前聚合设置 |

| UCM 自定义字段 | 含义 |
|---|---|
| `is_mla` | 聚合策略所需标志 |
| `load_failed_reqs` | UCM load 失败的请求 ID |
| `missing_reqs` | 其中属于缓存缺失类错误的请求 |
| `missing_blocks` | 缺失的外部缓存 key，使用统一的逻辑 key 命名空间 |
| `dump_succeeded_blocks` | 本次收集到的 dump 成功外部 key，供 scheduler 侧一致性跟踪器更新 |

`build_connector_worker_meta()` 返回已累计的事件后，会换成一个新的 `UCMWorkerMetadata` 对象。它更像“自上次收集以来发生了什么”，而非每轮发送所有 worker 状态的完整快照。在不需要一致性事件、也没有失败事件时，还可能返回 `None`。

源码：[ModelRunnerOutput](/Users/chyanmio/vllm/vllm/v1/outputs.py:264)、[KVConnectorOutput](/Users/chyanmio/vllm/vllm/v1/outputs.py:215)、[UCMWorkerMetadata](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:1331)、[收集并重置 worker metadata](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2600)。

### 7.2 回传的函数调用链

```text
worker connector 收尾
  ├─ get_transfer_results(finished_req_ids)
  │    └─ UCM.get_finished(...)
  │         ├─ _poll_pending_load_tasks()
  │         └─ 对已结束请求处理 pending dump
  ├─ get_block_ids_with_load_errors()
  └─ build_connector_worker_meta()
       ↓
KVConnectorOutput
       ↓
GPUModelRunner 返回 ModelRunnerOutput
  或先保存结果，待 sample_tokens() 返回 ModelRunnerOutput
       ↓
WorkerProc.handle_output()
  └─ enqueue_output()
       └─ worker_response_mq.enqueue((SUCCESS, output))
       ↓
executor 从各 worker response queue 读取结果
  └─ KVOutputAggregator.aggregate(...)
       ↓
EngineCore 获取本轮 ModelRunnerOutput
  └─ scheduler.update_from_output(...)
       ├─ 处理 load 错误、调整重算边界
       └─ _update_from_kv_xfer_finished(...)
            ├─ UCM.update_connector_output(...)
            ├─ finished_recving：登记 KV 接收结束
            └─ finished_sending：处理延迟释放的 block
```

在启用 vLLM async scheduling 时，worker 还可能先把输出交给自己的异步输出线程，再放入响应队列；这不改变回传对象的含义。

当前常见生成路径中，`GPUModelRunner.execute_model()` 保存本轮执行状态和 `kv_connector_output` 后返回 `None`；`sample_tokens()` 生成最终输出时才把它装进 `ModelRunnerOutput`。所以画调用图时必须包含采样阶段。零 token、pooling 等分支可以直接产生输出，不能认为所有结果必经采样。

源码：[收集 connector 输出](/Users/chyanmio/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:69)、[execute_model 保存状态后返回 None](/Users/chyanmio/vllm/vllm/v1/worker/gpu_model_runner.py:4461)、[sample_tokens](/Users/chyanmio/vllm/vllm/v1/worker/gpu_model_runner.py:4498)、[构建最终输出](/Users/chyanmio/vllm/vllm/v1/worker/gpu_model_runner.py:4712)、[worker 响应队列](/Users/chyanmio/vllm/vllm/v1/executor/multiproc_executor.py:986)、[scheduler 处理回传](/Users/chyanmio/vllm/vllm/v1/core/sched/scheduler.py:1992)。

### 7.3 TP 多个 worker：完成事件要收齐，错误事件要合并

有 connector 聚合器时，多进程 executor 会收集各 worker 的返回，合并 KV 信息，再交给 scheduler。

`KVOutputAggregator` 的完成判断不是简单取各 rank 的完成集合并集。它有跨调用保留的计数器：每收到一份某请求的完成通知，剩余份数减一，收齐才向 scheduler 报告完成。默认期望数量来自 connector 的 `get_finished_count()`，未指定时使用 executor 的 `world_size`。

例如假设需要 2 个 worker 的完成通知：

| step 的输出 | rank 0 | rank 1 | 聚合后 scheduler 能看到的 `finished_recving` |
|---|---|---|---|
| S | R 完成 | R 尚未完成 | 不包含 R，剩余 1 份 |
| S+1 | 不重复报告 R | R 完成 | 包含 R，收齐两份 |

每个 worker 应在任务完成时报告一次，不能在后续每个 step 无限重复同一个完成事件，否则会破坏计数语义。

UCM 自定义 metadata 的聚合规则另外定义：

- load 失败请求、缺失请求、缺失 key：取并集，任意 rank 的错误都需要保留。
- `dump_succeeded_blocks`：当前 `is_mla=False` 时取交集，`is_mla=True` 时取并集。

两类聚合不能混淆：前者是“这条请求的传输完成通知收齐了吗”，后者是“本次各 rank 上报的哪些缓存 key 可以认定为成功 / 缺失”。UCM 的 dump 成功 key 聚合代码还保留了 PP-aware TODO，不能把它解释成已经完整覆盖所有 PP 拓扑的一致性证明。

源码：[KVOutputAggregator](/Users/chyanmio/vllm/vllm/distributed/kv_transfer/kv_connector/utils.py:62)、[聚合器初始化](/Users/chyanmio/vllm/vllm/v1/executor/abstract.py:296)、[UCM metadata aggregate](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:1348)。

## 8. 用你的 step-wise 路径举一个跨 step 的完整例子

假设：R 已命中外部缓存 h1、h2，对应目的地 GPU block 3、6；A、B 可以正常计算。为了清楚，先按单 worker 描述，实际多 worker 还需上述聚合。

### Step S：下发 R 的异步加载计划，同时计算 A、B

```text
scheduler：
  R 已分配 GPU block，但状态为 WAITING_FOR_REMOTE_KVS
  num_scheduled_tokens = {"A": 16, "B": 1}
  kv_connector_metadata.request_meta 包含：
    "R": load(h1,h2 → block 3,6), load_async=True

worker：
  收到整份 SchedulerOutput
  A、B 的 forward launch
  start_load_kv() 提交 R 的 Store task
  _pending_load_tasks["R"] = PendingLoadTask(...)
  收尾检查 R 的任务，还没完成
  回传结果里暂时没有 finished_recving={"R"}
```

### Step S+1：不重新下发同一加载计划，继续检查旧任务

```text
scheduler：
  R 仍然等待，不参加本轮 forward
  不必在新 metadata 中重复 R 的原加载指令

worker：
  绑定 S+1 的 metadata
  S 的 _pending_load_tasks["R"] 仍然存在
  本轮收尾时检查该任务
  如果未完成，继续保留它
```

这里的两个生命周期不同：S 的工作单已经用过、可以清理；S 创建的 Store task 仍在执行。

### Step S+2：worker 观察到加载结束，并回传

```text
worker：
  check_load(task) → 已完成
  wait_load(task) → 检查延迟错误、回收任务上下文
  从 pending 字典中移除 R
  KVConnectorOutput.finished_recving = {"R"}

scheduler：
  update_from_output() 接收聚合后的完成事件
  将 R 加入 finished_recving_kv_req_ids
```

这条通知通过 step 的结果路径回传。源码中没有让 Store task 一完成就直接远程修改 scheduler 请求状态的机制；实际通知时间取决于 worker 下一次检查任务并返回结果的时间。

### 后续调度：R 重新获得计算资格

后续 `schedule()` 检查等待请求时，发现 R 的接收完成通知，调用 `_update_waiting_for_remote_kv()`，登记有效 prefix，并处理尾 token / replay 等规则，允许请求离开加载等待状态，再依据 token budget、显存和其他限制决定是否调度 R。

因此，“S+2 看到完成，S+3 一定计算 R”不是保证；更准确的是，R 在后续调度中具备重新进入计算的条件。

源码：[pending load 检查](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2655)、[完成事件进入 scheduler](/Users/chyanmio/vllm/vllm/v1/core/sched/scheduler.py:3165)、[等待请求恢复处理](/Users/chyanmio/vllm/vllm/v1/core/sched/scheduler.py:3084)、[调度时检查等待状态](/Users/chyanmio/vllm/vllm/v1/core/sched/scheduler.py:3131)。

## 9. 如果 load 失败，metadata 如何让 scheduler 知道要重算

假设 R 加载 h2 时发现外部数据缺失，block 6 没有获得有效 KV。

worker 的相关输出可能同时表达：

```text
KVConnectorOutput.finished_recving = {"R"}   这个加载任务已结束
KVConnectorOutput.invalid_block_ids 包含 6  对应 GPU block 不可用
UCMWorkerMetadata.load_failed_reqs 包含 R   请求加载失败
UCMWorkerMetadata.missing_reqs 包含 R       原因属于缓存缺失
UCMWorkerMetadata.missing_blocks 包含 h2    外部缓存 key 缺失
```

它们分别回答“还等不等”“哪些 block 不可信”“是哪条请求失败”“外部缓存出了什么问题”。失败时也报告任务结束，是为了让请求能退出加载等待；不能仅凭 `finished_recving` 判断 KV 可用。

scheduler 分两部分处理：

1. vLLM 根据无效 block / 失败身份识别受影响请求。开启加载失败重算策略时，会回退有效计算前缀并安排重算；关闭该策略时，则按失败请求处理，不能保证继续生成。
2. UCM 根据自定义 metadata 更新缺失缓存跟踪器。对于缺失类错误，会保留请求账本，把命中 / 已处理位置退回本地 HBM 前缀边界，使后续重算和 dump 有机会修复外部缓存；其他 load 错误则清理该请求的 UCM 账本，跳过后续缓存写入。

**适配新版要特别检查 `failed_recving`。**当前 vLLM 已提供这个按请求身份表达失败的字段，但本仓库的 UCM 路径主要上报 `invalid_block_ids` 和 `UCMWorkerMetadata.load_failed_reqs`。UCM 没有自行覆盖 `get_transfer_results()` 来把后者自动写入前者；基类包装 `get_finished()` 时只构造完成集合。

所以，“新版字段存在”不代表“现有 UCM 已经填充这个字段”。这里还有明确的兼容边界：**当前 vLLM 在多个 KV cache group 的模型上收到非空 `invalid_block_ids`，会在 `_handle_invalid_blocks()` 直接抛出 RuntimeError，要求 connector 改用 `KVConnectorTransferResults.failed_recving`。**

例如两个 group 都可以有 block 6，一个属于 full attention，一个属于 window attention。回传 `{6}` 无法告诉 scheduler 失败的是哪个 group、哪条请求；回传失败请求 `{R}` 则身份明确。新版当前的按请求失败处理也更保守：在重算策略开启时，对等待加载的失败请求把 `num_computed_tokens` 设为 0，而不是按这个 block ID 精确保留部分外部前缀。

适配 FAWA / 多 KV pool 时，必须核对按请求报告失败的链路，且不能在填入新字段以后还继续向这个分支上报非空整数 block 错误集合，否则仍会触发上述异常。这是源码中的明确检查，不只是潜在歧义。

源码：[UCM load 检查及失败结束通知](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2655)、[UCM 根据错误更新 scheduler 账本](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:2612)、[vLLM 无效 block 处理](/Users/chyanmio/vllm/vllm/v1/core/sched/scheduler.py:3193)、[按请求处理失败](/Users/chyanmio/vllm/vllm/v1/core/sched/scheduler.py:3297)、[多 group 拒绝整数 block 错误](/Users/chyanmio/vllm/vllm/v1/core/sched/scheduler.py:3313)、[基类 get_transfer_results](/Users/chyanmio/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:396)、[一致性跟踪器应用回传](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/rank_consistency.py:288)。

## 10. 每轮传一次，为什么还能异步执行

| 状态 / 对象 | 生命周期 |
|---|---|
| scheduler 的 `RequestMeta` | 通常跟随请求，记录长期命中和 dump 进度 |
| 单个 `SchedulerOutput` | 一次逻辑调度；异步 engine 中可以在 batch queue 等待对应结果 |
| worker 绑定的 `_connector_metadata` | 当前执行批次；常规收尾时解除绑定，draft 等情况可能延后 finalize |
| `_pending_load_tasks` / `_pending_dump_tasks` | 随任务，从提交到完成；允许跨 step |
| 一次返回的 `UCMWorkerMetadata` | 一次事件收集；收集后创建新对象累计下一批事件 |

`clear_connector_metadata()` 只是清除当前计划引用，不是取消先前已经提交的 Store task。任务所需的上下文已经由 pending 对象或具体 connector 状态保留。

vLLM 自己的 async scheduling 还可能在前一批执行结果返回前调度后一批。`step_with_batch_queue()` 保存对应的 `(future, scheduler_output, ...)`，后续取回结果时用匹配的原始调度对象更新 scheduler。

所以这里的 step 是“一份调度计划及其对应输出”的逻辑单位，不是要求整个系统所有异步任务在同一时间刻度全部结束。

你的 request-async load 和 vLLM 的 async scheduling 也是两个开关：前者控制 KV 加载是否跨 step 等待，后者控制 engine 调度 / 执行流水线；不能把二者当成同一种异步机制。

源码：[清理绑定](/Users/chyanmio/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:251)、[worker pending task 类型](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/ucm_connector.py:1313)、[step_with_batch_queue](/Users/chyanmio/vllm/vllm/v1/engine/core.py:670)、[延后 finalize](/Users/chyanmio/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:52)。

## 11. HLA / FAWA 的 metadata 有什么不同

前面用普通 Direct 的 key / block 结构讲主线，但混合模型需要额外描述不同缓存组。

| 路径 | 下发的额外内容 | 为什么需要 |
|---|---|---|
| HLA | `load_full_attn_count`、`dump_full_attn_count` | 在继承普通计划的基础上区分 full attention 部分，处理 MLA 等 rank 范围 |
| FAWA | `load_keys`、`load_hash_start/end`、按 group 组织的 `load_vllm_block_ids`；dump 侧对应字段；`load_async` | Full Attention 与 Window Attention 的可复用区间、block 映射不同，不能只使用普通路径第一组 block IDs |

FAWA 使用 `UCMFAWAConnectorMetadata`，外层仍是 `request_meta` 和 `preempted_req_ids`。transport 和 step 生命周期没有因为字段变多而改变：仍随 `SchedulerOutput.kv_connector_metadata` 下发，再由 worker 用本地 group / layout 解释。

源码：[HLARequestDispatchMeta](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/hla_connector.py:68)、[FAWARequestDispatchMeta / UCMFAWAConnectorMetadata](/Users/chyanmio/unified-cache-management/ucm/integration/vllm/hma_connector.py:270)。

## 12. 继续适配新版时，建议沿这五处核对

1. **计划构建**：`build_connector_meta()` 是否正确读取新版 new / cached request 字段，能否把等待请求的 async 计划独立合并进去。
2. **启动时机**：`has_sync_kv_loads` 是否与实际需要同步加载的请求一致，避免当前 forward 在 KV 就绪前读取目的 block。
3. **地址解析**：metadata 传逻辑 block ID，worker 按新版实际 KV layout / group / stride 生成地址；不要把旧版物理布局假设塞进调度端。
4. **结果回传与聚合**：同时核对 `execute_model()`、`sample_tokens()`、空 forward 分支、rank 完成计数、失败请求身份，避免某个分支丢失 connector 输出。
5. **block 生命周期**：抢占、请求结束、加载失败时，完成通知和 pending task 必须协同处理，不能让 worker 仍在访问的 GPU block 被提前复用。

`finished_sending` 应理解为 connector 定义的“现在可以处理 block 释放”的完成条件。结合 UCM Store 的多级缓存实现，它不自动等同于所有后端介质都已经完成持久化；应沿具体 Store 的任务完成边界判断。

前缀缓存命中、分配、load、dump、释放的整体链路见：[UCM prefix cache 全流程](./ucm_prefix_cache_full_flow_analysis.md)。

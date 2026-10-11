# Step-wise 请求级异步加载工作说明

## 工作概述

Step-wise 将外部 KV 加载作为可以跨调度 step 存续的请求级任务：请求等待缓存时，调度器可以继续执行其他就绪请求；加载完成后，再恢复该请求的计算。目标是减少同步加载对其他请求的阻塞，并为加载与计算重叠创造条件。

本文以当前 `step_wise` 分支为准，合并介绍以下工作：

- 复用 `e80ff8bc`（PR #1246）已有的 Direct 请求级异步加载协议。
- `99601e34`：将协议扩展到 bulk HLA、HMA/FAWA，并补充 Direct 加载和调度指标。
- `9ee647e2`：补充昇腾 HLA 状态恢复、同步边界及测试适配。

以下按功能说明，不按开发阶段拆分。DS V4.1 新 layout 的独立适配见 [README_DSV4_1_ADAPTATION.md](README_DSV4_1_ADAPTATION.md)，不将其算入本分支的 step-wise 增量。

## 1. 请求级异步加载与能力开关

**概括：将已有 Direct 异步加载能力扩展为由各 Connector 明确选择支持的机制。**

- 通过 `use_request_async_load` 控制加载模式，默认关闭，保留同步加载行为。
- 将原先只允许 `UCMDirectConnector` 开启的判断，改为 `_supports_request_async_load()` 能力接口；bulk HLA 和 FAWA 分别声明支持。
- 对未适配的子类关闭请求级异步并打印提示，避免仅因为继承 Direct 就自动启用不同语义的加载路径。
- 在启动日志中打印具体 Connector、scheduler/worker 角色、配置开关及 `effective_mode`，便于确认实际生效模式。
- 复用请求元数据中的 `load_async` 标记，同步请求仍在当前加载调用中等待，异步请求则提交后由后续 step 检查完成情况。标记采用 keyword-only 参数，维持原有构造参数兼容性。

**作用：** 用统一开关控制三条已适配路径，同时明确支持边界，方便同步、step-wise、layer-wise 对照测试。

**主要位置：** `ucm_connector.py` 的 `UCMDirectConnector.__init__`、`_supports_request_async_load`、`RequestDispatchMeta`；HLA/FAWA 的同名能力接口。

## 2. 调度元数据与加载生命周期解耦

**概括：请求尚未进入模型 forward 时，也能下发加载任务；任务状态可以保留到后续 step。**

- 外部命中时返回异步加载标记，交由 vLLM 现有远端 KV 等待与恢复机制管理请求。
- 在 `update_state_after_alloc()` 中，使用已经分配的 HBM block 建立外部缓存 key 到设备 block 的映射，生成只加载、不计算新 token 的 load-only 元数据。
- 将待下发任务单独保存在 `_pending_async_load_dispatches`，在 `build_connector_meta()` 中合并下发；即使请求没有出现在本轮正常计算列表里，也不会漏掉加载。
- 使用 `_async_load_req_ids` 标记已派发的请求，恢复计算时继续生成正常 dump 计划，同时避免再次发起相同 load。
- 在重新查询、抢占后重调度及请求结束时清理相应 scheduler 状态，避免旧映射影响新一轮分配。

基本调用关系：

```text
外部命中 → 分配 HBM block → 下发 load-only 元数据 → worker 提交 load
                                                      ↓
其他就绪请求继续运行 ← 等待期间轮询 ← get_finished()
                                      ↓ 完成
                              finished_recving → 恢复计算
```

**作用：** 将“加载请求”与“本轮执行模型的请求”分开管理，使加载能够跨 step 存续，并复用 vLLM 的请求恢复协议。

**主要位置：** 三类 Connector 的 `get_num_new_matched_tokens`、`update_state_after_alloc`、`build_connector_meta`、`start_load_kv`、`get_finished`。

## 3. HLA 多组缓存的异步加载适配

**概括：将全注意力 KV 和 Mamba state 的多组恢复接入请求级异步流程。**

- 分配后记录全部 group 的 block 列表，并检查数量与 HLA group manager 一致，避免只处理单组缓存。
- 复用 `_generate_hla_dispatch_meta()`，以 `new_tokens=0` 生成初始异步加载计划，保留既有全注意力 KV 与 Mamba state 的恢复规则。
- 明确传入的是完整 block 表还是新增 block 列表，避免初始加载与恢复计算时重复追加 block。
- worker 完成 rank 范围处理后，将任务和实际目标 block 保存在 `PendingLoadTask`；异步分支不立即 `wait_load()`，由共享完成轮询流程接管。
- 保留现有 HLA 的 LCM 对齐、状态命中和窗口边界语义，改变的是加载生命周期，而不是放宽有效缓存的判定条件。

**作用：** 将单组 Direct 的异步协议扩展到 HLA，保证不同类型缓存同时参与恢复且设备地址映射正确。

**主要位置：** `hla_connector.py` 的 `update_state_after_alloc`、`_generate_hla_dispatch_meta`、`build_connector_meta`、`start_load_kv`。

## 4. HMA/FAWA 双 Store 任务聚合

**概括：将同一请求的 FA、WA 加载聚合为一个请求级任务，全部子任务结束后才回报完成。**

- 新增 `FAWAPendingLoad`，保存请求 ID、FA/WA 子任务、涉及的设备 block、失败状态和成功传输字节数。
- 在分配后保存完整的多组 block 映射，并通过空的增量 block 列表生成 load-only 元数据，避免重复累计映射。
- 保留 FA/WA 原有加载语义：FA 加载命中的前缀块，WA 加载最终命中边界对应的窗口状态。
- `start_load_kv()` 将异步任务存入 `_pending_fawa_loads`；同步任务仍走原来的等待流程。
- 新增 `_poll_pending_fawa_loads()`，逐个检查 FA、WA 子任务；尚未完成的继续保留，全部结束后才返回请求完成。请求整体成功时再记录相应加载字节数。

**作用：** 避免 FA 已完成而 WA 尚未就绪时提前恢复计算，也支持同一批元数据中的同步与异步任务分别处理。

**主要位置：** `hma_connector.py` 的 `FAWARequestDispatchMeta`、`FAWAPendingLoad`、`update_state_after_alloc`、`start_load_kv`、`_poll_pending_fawa_loads`。

## 5. 异常处理、重复提交与资源释放保护

**概括：让任务成功、失败及请求取消都能结束等待，同时避免在途传输写入已回收的缓存块。**

- scheduler 在请求恢复时抑制重复 load，worker 也检查 pending 表，忽略同一请求的重复异步加载元数据。
- 复用 `RankConsistencyManager` 的 `submit_load/check_load/wait_load`：先非阻塞检查，完成后再调用 `wait_load()` 暴露延迟错误并完成任务收尾。
- 出错时记录失败、上报失效 block，并通知加载进入终态，避免请求永久停留在远端 KV 等待状态。后续失败或重算由支持范围内的 vLLM 策略处理；混合多组路径不能据此视为支持自动重算。
- FAWA 某个子任务提交失败时，仍保留此前已经提交的任务，等待它们结束后再回报；两个子任务都失败时，分别记录任务错误，但不重复累计同一请求的失效处理。
- FAWA 在 `get_finished()` 中区分仍有加载在途的请求与可结束发送的请求，避免取消/结束路径过早触发 HBM block 释放，并防止同一请求同时进入两类完成集合。

**作用：** 防止重复 I/O、任务等待不退出、部分缓存尚未就绪就续算，以及请求结束后仍有 DMA 写入旧 block 等问题。

**主要位置：** `_poll_pending_load_tasks`、`_record_fawa_load_error`、`_poll_pending_fawa_loads`，以及 FAWA 的 `get_finished`。

## 6. 昇腾 HLA 状态恢复与同步边界

**概括：补齐 load-only step 后的 Mamba 状态恢复，并减少没有加载时的设备同步。**

- 扩展已有 Ascend Mamba 拷贝补丁：执行预处理后，清理本轮零 scheduled token 请求留下的状态索引，避免保留 `-1` 等旧值，使下一次实际计算能按恢复后的 computed-token 位置找到源状态。
- 使用函数签名绑定参数，兼容全位置参数、全关键字参数以及混用方式；保留原函数元信息和返回值。
- 继续复用已有“预处理阶段执行 state copy、随后清零 copy offset”的顺序保护，避免 runner 后续的延迟拷贝再次覆盖加载结果。
- 将 bulk HLA 的设备同步移到本批第一次实际加载提交之前；同批后续加载复用该同步结果，无加载或仅收到重复任务时跳过同步。

**作用：** 保持状态拷贝、缓存清零与外部 DMA 的正确顺序，同时避免每个空加载 step 都发生全设备同步。

**主要位置：** `patch/v0210/vllm_ascend/mamba_copy_order_patch.py`；`UCMHybridLinearAttentionConnector.start_load_kv`。

## 7. 加载与调度可观测性

**概括：补充加载规模、任务状态和调度间隔指标，帮助分析异步模式的开销与收益。**

- 为 `PendingLoadTask` 增加起始时间，记录 Direct 同步加载耗时、异步任务从派发到终态的耗时。
- 增加 Direct 异步提交、成功、失败计数及当前 pending 数，提交失败也纳入任务结果统计。
- 按成功提交的请求加载记录 token 数和字节数，避免将单个请求的加载规模混淆为每块或每 step 的规模。
- 在 Direct scheduler 侧记录相邻活跃 `build_connector_meta()` 调用间隔、本轮 scheduled token 数和请求数；由 scheduler 单独上报，避免 TP 重复计数。
- 同步更新内置 `ucm/default_metrics_config.py` 与示例 `examples/metrics/metrics_configs.yaml`，复用已有指标分发和展示链路。

**口径说明：** 当前新增的 `direct_*` 指标只由 `UCMDirectConnector` 上报，不能直接用来判断 HLA/FAWA 是否发生异步加载。`direct_step_interval_ms` 是调度可见间隔，不是设备 forward 执行时间；可能包含不可见的空闲时间。逐 rank 指标也不能直接求和当作唯一请求数。

**主要位置：** Direct 的 `_record_direct_*` 方法、`_poll_pending_load_tasks`，以及两份 metrics 配置文件。

## 8. 回归覆盖与兼容性

**概括：围绕异步任务的关键状态转换补充或复用测试，保留同步及既有 Connector 行为。**

- 复用 Direct 已有测试，检查没有 scheduled forward 时也能派发 load、恢复后不重复加载、只回报已完成任务，以及失败后上报失效块并结束等待。
- 保留 `load_async` 的 keyword-only 构造兼容性测试，避免影响 Connector 子类原有位置参数。
- 补充 HLA 同步/异步加载测试，检查设备同步先于提交、每批只同步一次、异步分支不立即等待、重复元数据不重提，以及空加载 step 不额外同步。
- 补齐测试所需的 vLLM/设备类型接口桩和 Lite 配置桩，使相关测试能在无实际加速卡的 CPU 环境运行。
- 保持配置默认关闭；layer-wise、CP 等没有声明支持请求级异步的路径继续使用各自原有行为。

**验证范围：** 上述单测验证元数据、调用顺序和任务状态，不替代设备端精度与并发性能测试。HMA 双任务异常/取消、不同设备与模型组合仍应在对应场景验证。当前性能实测中两种模式差异较小，尚不据此宣称确定的吞吐或 TTFT 提升比例。

**主要位置：** `test/test_ucm_connector_metrics.py`；各 Connector 的能力判断与同步回退逻辑。

## 使用与支持范围

启用请求级异步加载的 UCM 顶层配置：

```yaml
use_layerwise: false
use_request_async_load: true
```

昇腾 HLA 同时需要在启动前设置 `ENABLE_UCM_PATCH=1`，使 Mamba 兼容补丁生效。

| 路径 | 请求级 step-wise | Layer-wise |
| --- | --- | --- |
| 普通 Direct | 支持 | 由独立 `UCMLayerWiseConnector` 提供 |
| HLA bulk | 支持 | 由独立 HLA layerwise 类提供 |
| HMA/FAWA | 支持 | 当前没有对应实现 |
| 普通/HLA layerwise 类 | 当前不支持与请求级 step-wise 叠加 | 支持 |
| CP、Blend、Mock 等其他派生路径 | 当前未声明支持请求级 step-wise | 按各自实现判断 |

启动日志应显示具体 Connector 和 `effective_mode=request_async`。实际重叠收益取决于是否有其他就绪请求、剩余计算量、存储带宽、调度以及同步屏障；单请求或集中突发的高命中场景可能收益有限。HLA 保存阶段仍存在等待，step-wise 的完成也不代表整个存取过程都变成无阻塞。

## 主要代码范围

| 文件 | 主要职责 |
| --- | --- |
| `ucm/integration/vllm/ucm_connector.py` | 公共协议、Direct 任务管理、能力开关和 Direct 指标 |
| `ucm/integration/vllm/hla_connector.py` | HLA 多组 load-only 元数据、异步提交与同步边界 |
| `ucm/integration/vllm/hma_connector.py` | FA/WA 任务聚合、失败处理和完成回报 |
| `ucm/integration/vllm/patch/v0210/vllm_ascend/mamba_copy_order_patch.py` | 昇腾 Mamba 状态恢复与参数兼容 |
| `ucm/default_metrics_config.py` | 内置指标定义 |
| `examples/metrics/metrics_configs.yaml` | 对应示例指标配置 |
| `test/test_ucm_connector_metrics.py` | 加载生命周期、指标和 HLA 调用顺序回归 |

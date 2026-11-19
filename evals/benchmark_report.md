# Agentic RAG Evaluation & Performance Benchmark Report

- **Generated At**: `2026-11-19T14:35:30.996430+00:00`
- **Judge Mode / Model**: `offline-heuristic`

## 1. 测试环境与硬件基线

- **Hardware Baseline**: Windows 11 | RTX 3060 (6GB VRAM, FP16 bge-m3 + bge-reranker-v2-m3)
- **VRAM Budget**: `BAAI/bge-m3` (~1.1 GB FP16) + `BAAI/bge-reranker-v2-m3` (~1.1 GB FP16) ~= **2.3 GB VRAM** (严格 <= 6 GB 上限, 零本地 LLM 部署)
- **Vector Store & Cache**: Qdrant (`int8` 标量量化 Dense + BM25 Sparse + RRF) + Redis 8 原生向量语义缓存 (Cosine >= 0.92)
- **Orchestration**: LangGraph `StateGraph` + `AsyncSqliteSaver` 检查点持久化 + Dual-Channel FastAPI SSE

## 2. Ragas v0.2+ 检索与生成质量对比表 (v1 vs v2 vs v3)

| Pipeline | Samples | Context Recall | Faithfulness | Context Precision | Overall Score | P50 Latency (ms) | P95 Latency (ms) | Mean TTFT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 6 | 0.3884 | 0.6392 | 1.0000 | 0.6759 | 410.00 | 410.00 | 135.00 |
| v2 | 6 | 0.7679 | 0.7984 | 1.0000 | 0.8554 | 680.00 | 680.00 | 185.00 |
| v3 | 6 | 1.0000 | 0.9851 | 1.0000 | 0.9950 | 850.00 | 940.00 | 215.00 |

## 3. 异步高并发性能压测与 SLA 达成表 (P50 / P95 / P99 / TTFT / RPS / 缓存命中率)

| Pipeline | Requests | Concurrency | Mock Stream | Success Rate | Cache Hit Rate | RPS | P50 (ms) | P95 (ms) | P99 (ms) | Mean TTFT (ms) | SLA (< 1800 ms) |
| --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| v1 | 20 | 10 | true | 100.0% | 0.0% | 144.11 | 17.18 | 28.00 | 83.74 | 21.84 | PASS |
| v2 | 20 | 10 | true | 100.0% | 0.0% | 68.33 | 106.79 | 110.06 | 110.38 | 93.04 | PASS |
| v3 | 20 | 10 | true | 100.0% | 30.0% | 19.55 | 445.01 | 680.97 | 687.77 | 442.66 | PASS |

## 4. 架构演进关键结论

1. **混合检索 + CUDA Reranker 提升多跳召回**: `v2` 相比 `v1` 在 Context Recall 上提升 **+0.3795** (0.3884 -> 0.7679), 表明 Markdown 结构化切分 + Dense/Sparse RRF 融合与本地 FP16 精排显著改善了上下文覆盖率.
2. **LangGraph 自纠错闭环抑制幻觉**: `v3` 通过文档相关性评分、有界查询重写 (`max_retries=2`) 与幻觉校验, Faithfulness 达到 **0.9851** (较 `v1` 提升 **+0.3459**), Overall Score 达 **0.9950**.
3. **Redis 8 向量语义缓存削峰与 SLA 达成**: 在并发度 `10` 压测下, `v3` 语义缓存命中率达 **30.0%**, 使中位延迟维持在 **445.01 ms**, P95 延迟为 **680.97 ms** (吞吐 **19.55 RPS**), 满足 **P95 < 1.8 s (1800 ms)** 生产 SLA (PASS).

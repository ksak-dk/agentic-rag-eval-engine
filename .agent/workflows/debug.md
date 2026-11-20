---
description: 系统化根因排查工作流（/debug）：严格遵循 Reproduce -> Isolate -> Fix -> Verify 四步法，禁止盲目试错打补丁。
---

# `/debug` — 系统化根因排查工作流 (Systematic Root-Cause Debugging)

## 触发场景
- 用户输入 `/debug`。
- 遇到 `pytest` 单测失败、SSE 双通道流异常（断流/事件格式错误/前端渲染异常）、Redis/内存语义缓存命中异常、LangGraph 状态流转异常或引用角标 `.cite-badge` 联动失效时。

## 强制四步法：`Reproduce -> Isolate -> Fix -> Verify`

### 1. Reproduce（稳定复现）
- **严禁盲目猜测、严禁在未复现前试错改代码**。
- 先在终端运行精确的失败命令或编写最小复现测试断言：
  - 单测失败：`uv run pytest <test_file>::<test_name> -vv --tb=short`
  - SSE 流或缓存/图节点异常：在对应 `tests/` 下构造最小可复现单测（Mock 外部 LLM/GPU 边界，保持异步纯度），确认在当前代码上稳定失败并记录真实报错堆栈与退出码。

### 2. Isolate（定位根因）
- 顺着报错堆栈与数据流，定位到具体模块与精确行号（常见排查边界）：
  - **SSE / API 层**：`app/api/v1/chat.py`（流式分帧、事件类型、`PIPELINE_VERSION` 路由、语言对齐）。
  - **图编排层**：`app/graph/nodes.py`、`app/graph/workflow.py`（`StateGraph`、`AsyncSqliteSaver` 状态转移与异常边界）。
  - **检索 / 缓存 / Mock 适配器**：`app/cache/semantic_cache.py`（Cosine 阈值 `0.92`、分词/哈希）、`app/adapters/mock/router.py`（`ZH/EN/KO` 本地化合成与引用标签）。
  - **前端工作台**：`tests/sse_viewer.html`（SSE 解析、`renderSafeMarkdown`、`.cite-badge` 点击高亮、三语状态同步）。
- 明确向用户说明：**现象是什么、根因在哪一行/哪个状态转换、为什么会触发**。

### 3. Fix（最小化修复）
- 仅针对确认的根因做最小化、高内聚修复，遵守 `GEMINI.md` 第 4 节代码质量门禁（严格类型注解、Pydantic V2、异步纯度、显式异常边界与结构化日志）。
- 若修复需要修改超过 2 个文件，或涉及爆炸半径动作（修改公共 `/api/v1/` 契约、`pyproject.toml` 依赖、破坏现有 65 项 `pytest` 断言），必须先暂停并走 `/plan` 或请求人工确认。

### 4. Verify（回归闭环）
- 先运行复现用例确认由 FAIL 转为 PASS；
- 再运行 `/verify` 完整质量门禁（`uv run ruff check .`、`uv run ruff format --check .`、`uv run pytest -v`），并在回复中附上真实命令输出与退出码。

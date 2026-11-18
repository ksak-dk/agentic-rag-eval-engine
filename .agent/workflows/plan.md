---
description: 规约驱动的计划工作流（/plan）：生成原子级 Implementation Plan Artifact；当存在有效 UPDATE_SPEC.md 时自动放行（auto_approve: true）并直接编码与 /verify 验收。
auto_approve: true
---

# `/plan` — 规约驱动计划工作流 (Specification-Driven Planning)

## 触发场景
- 用户输入 `/plan`。
- 任何开发新特性、架构重构、新增模块，或预计修改超过 **2 个文件** 的任务。

## 强制执行规则

### Step 0: 内联规约自动落盘 (Inline Spec Auto-Save)
- 在开始任何分析或检索前，先检查本次用户输入的 `/plan` 消息中是否包含 `--- SPEC START ---` 分隔符：
  - **如果包含**：立即将 `--- SPEC START ---` 之后的全部内容原样提取并写入（或覆盖）项目根目录的 `UPDATE_SPEC.md`（已被 `.gitignore` 忽略，严禁提交到 Git），然后再基于该文件制定计划。
  - **如果不包含**：按原逻辑直接读取项目根目录已有的 `UPDATE_SPEC.md`；若该文件不存在，则直接根据用户简述制定计划。

### 后续规划与门禁规则

1. **全自动放行与规划门禁（Autonomous Execution Policy）**
   - 当根目录存在有效 `UPDATE_SPEC.md`（或本次 `/plan` 已提取 `--- SPEC START ---` 落盘为 `UPDATE_SPEC.md`）且未触发下方爆炸半径门禁时：**自动将生成的 `Implementation Plan` Artifact 视作已批准（`auto_approve: true`）**，生成计划后**无需暂停等待人工确认**，立即进入编码实现、`/verify` 门禁测试与 `HANDOVER.md` 更新。
   - 仅当不存在有效 `UPDATE_SPEC.md` 或触发爆炸半径拦截门禁时，才在生成 `Implementation Plan` 后暂停并请求人工确认，严禁绕过确认直接修改高风险配置文件。

2. **对照本地规约与硬件约束**
   - 完整读取根目录 `UPDATE_SPEC.md`（若存在）中的架构约束、模块边界、状态字段与接口契约。
   - 同时对照根目录 [GEMINI.md](GEMINI.md) 中的硬件约束（RTX 3060 6GB VRAM、≤ 2.3 GB 分配上限、禁止本地加载 LLM）与代码质量门禁。

3. **输出原子级 `Implementation Plan` Artifact**
   - 计划必须拆解为单个 **2～5 分钟可验证** 的原子步骤，每一步必须明确写出：
     1. **精确修改/新增的文件路径**（例如：`app/graph/nodes.py`、`app/adapters/mock/router.py`、`tests/sse_viewer.html`）。
     2. **函数签名与数据模型变更**（Pydantic V2 模型字段、公共函数类型注解、SSE 事件载荷变化等）。
     3. **对 `ZH / EN / KO` 三语兼容性的影响评估**（含 `detect_query_language`、多语言 Prompt、Demo 本地化证据及前端三语 UI/Citation 联动是否受影响）。
     4. **可执行的验收命令**（精确到 `uv run pytest ... -v` 目标用例与 `uv run ruff check .`）。

4. **爆炸半径检查 (Blast-Radius Check)**
   - 若计划涉及以下高风险变更，必须在计划中显式标注 `⚠️ 需要人工确认`：
     - 修改 `pyproject.toml` 新增/升级/移除第三方依赖；
     - 更改 `/api/v1/` 公共接口签名或响应契约；
     - 修改数据库 Schema、SQLite Checkpoint 结构或执行数据清理；
     - 调整或破坏原有 56 项 `pytest` 单测断言；
     - 修改 `.github/workflows/` CI/CD 或发版配置。

5. **自动放行执行与编码后 `/verify` 衔接**
   - 若存在有效 `UPDATE_SPEC.md` 且未命中爆炸半径门禁，生成 `Implementation Plan` Artifact（`RequestFeedback: false`）后立即按计划编码，随后自动执行 `/verify`（注意在允许 `127.0.0.1` 回环通信的前台非沙箱终端运行 `pytest`）并按 `GEMINI.md` Section 8 更新 `HANDOVER.md`。
   - 若无有效 `UPDATE_SPEC.md` 或命中爆炸半径门禁，则在生成 `Implementation Plan` Artifact 后停下并指出关键决策点与风险，等待 Approve 后再编码。

6. **全局语言规范 (Language Policy)**
   - 所有面向用户的自然语言输出（包括 `Implementation Plan`、`Task` 清单、`Walkthrough`、代码审查说明及日常对话）必须默认使用**简体中文**。
   - 仅代码标识符、文件路径、终端命令、技术专有名词以及 Git Commit Message（遵循 Conventional Commits）保持英文。

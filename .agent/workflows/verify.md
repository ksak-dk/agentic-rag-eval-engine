---
description: 完成前强制证据门禁与交接工作流（/verify）：依次实跑 Ruff 与 Pytest，按需浏览器巡检，通过后按 GEMINI.md Section 8 更新 HANDOVER.md。
---

# `/verify` — 完成前强制证据门禁 + 交接工作流

## 触发场景
- 用户输入 `/verify`。
- 任何代码、测试、前端工作台或工程配置修改完成，准备向用户汇报“已完成”之前。

## 强制执行流程

### 1. 终端质量门禁（严禁跳步或口头声称已通过）
必须在终端真实依次执行以下 3 条命令，并在最终回复中附上**真实执行摘要与退出码（exit code）**：

1. **Lint 检查**：
   ```powershell
   uv run ruff check .
   ```
   要求：退出码 `0`，`All checks passed!`。

2. **格式化检查**：
   ```powershell
   uv run ruff format --check .
   ```
   要求：退出码 `0`，全部文件已格式化。

3. **全量测试套件**：
   ```powershell
   uv run pytest -v
   ```
   - **执行环境约束**：执行 `pytest` 时**必须**申请前台非沙箱权限（`BypassSandbox: true`，允许 `127.0.0.1` 本地回环套接字通信），**严禁**在沙箱中运行或丢入沙箱后台任务运行（否则测试中的本地回环通信会阻塞卡死）。
   - 要求：退出码 `0`，全部用例通过（无未预期跳过或放宽断言；注意原有基线用例均需保持通过）。

> 若任一命令失败（退出码非 `0`），立即中止收尾并转入 `/debug` 排查修复，严禁绕过检查或篡改测试断言。

### 2. 前端三栏工作台按需巡检（仅当改动涉及 `tests/sse_viewer.html` 时）
- 若本次修改涉及 `tests/sse_viewer.html`（或相关 SSE 前端交互/样式/三语逻辑）：
  - 可按需启动本地测试服务并调用 Browser Subagent 访问对应本地端口（如 `http://127.0.0.1:<port>/viewer` 或 `/tests/sse_viewer.html`）；
  - 验证浏览器 Console **零报错**；
  - 验证顶部 `KO / ZH / EN` 三语切换正常运行；
  - 验证正文 `.cite-badge` 点击后可平滑滚动并触发右侧 Citations 卡片高亮联动。

### 3. 自动化交接更新（遵循 `GEMINI.md` Section 8）
- 仅在全部质量门禁通过且确认准备/完成 `git commit` 后执行：
  - **严禁**在本地手动创建任何 Git Tag（版本 Tag 与 GitHub Release 由 `.github/workflows/ci.yml` 在 `push origin main` 后全自动生成）。
  - 覆盖更新根目录的 `HANDOVER.md`（已被 `.gitignore` 忽略，严禁提交到 Git），确保完整包含以下 4 项交接要素：
    1. **最新 Git 状态**：`git log -n 3 --oneline` 与 `git status` 的实际输出。
    2. **质量门禁结果**：`uv run ruff check .`、`uv run ruff format --check .` 与 `uv run pytest -v` 的真实执行结果与通过统计。
    3. **新增/修改模块接口摘要**：关键 Class 与公开函数签名变化。
    4. **下一阶段衔接点**：待办事项、依赖契约与注意事项。

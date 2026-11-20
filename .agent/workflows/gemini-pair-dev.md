---
description: 启动 Antigravity（本地执行者）与网页版 Gemini（架构师）基于 bsk（BrowserSkill CLI + Skill）的协同开发、审查与测试闭环
---

# 双端协同开发协议（`bsk` 网页版 Gemini 架构师 + Antigravity 执行者）

> 触发条件：仅当用户显式调用 `/gemini-pair-dev` 或明确要求“双端/联合审查”时触发；常规对话按 `GEMINI.md` 对话自适应分流直接执行。

## 核心协同链路

```text
[双端协同] 仅当你点名 /gemini-pair-dev 或要求联合审查时触发
  ├─ 0. 前置检查：确认非 main 分支；生成/更新精简状态板 .pair-state.md（只记目标、约束、未决项）+ git --no-pager diff
  ├─ 1. 额度与动态选型（含兜底）：通过 bsk 连接网页版 Gemini 查额度；在排除黑名单（如 4 Carbon）与不可用项后动态锁定当前最高级可用推理/编程模型（仅当前/每周用量 > 85% 时降级轻量/自动模式）；若 bsk daemon 未启动、网页超时或额度耗尽，自动降级为本地单端自审
  ├─ 2. 审查闭环（Max 3 轮熔断）：仅向网页端发送 .pair-state.md + 精简 diff；网页端只输出架构与审查意见（禁止吐全量代码）─▶ 本地落盘修改并更新 .pair-state.md ─▶ 连续 2 次同类报错或达 3 轮上限即熔断汇报
  ├─ 3. 防退化迁移：若网页会话超长或出现退化/幻觉，自动触发 A+B 迁移（新开网页会话，直接注入 .pair-state.md + 最新 diff，不再搬运冗长历史对话）
  └─ 4. 收尾验收：审查通过后执行 bsk close，自动执行 /verify（跑通 uv run pytest -q 与 ruff）并清理 .pair-state.md 等临时文件
```

---

## `bsk` 网页交互硬性操作规范（防卡死 / 防抢占）
参考 [browser-skill/SKILL.md](file:///d:/Code/agentic-rag-eval-engine/.agent/skills/browser-skill/SKILL.md) 执行所有浏览器命令，并严格遵守：
1. **非沙箱前台执行**：`bsk` 需读取宿主机 `daemon.json` 与 IPC，调用时使用 `BypassSandbox: true` 且**前台同步等待完成**，严禁丢入后台后提前结束回合；Git 查看命令统一加 `--no-pager`。
2. **PowerShell 单引号与状态同步**：在 Windows PowerShell 中调用 `bsk` 时，`@eN` 元素编号必须加单引号（如 `bsk click '@e101'`）避免被解析为 splatting；每次页面跳转、弹窗或菜单展开后必须先执行 `bsk sync` 刷新 DOM 上下文。
3. **流式等待与精确提取**：发送消息后网页端需要 5~30 秒生成回复，**严禁立即抓取**；必须等待“停止生成（Stop）”消失且回复下方操作栏（`复制 / 重做`）出现后，使用 `bsk read --selector "model-response"` 精确提取最新一条回复文本及当前线程 `page_url`（不读侧边栏历史）。
4. **人工接管与让权保护**：若弹出登录验证、CAPTCHA 或安全提示，立即暂停提示用户手动处理后 `bsk sync` 继续；若遇 `User activity revoked this action`，说明用户正在操作该标签页，等待用户操作完毕后 `bsk sync` 再继续。
5. **断线重连**：首轮发送后立即记录线程完整 URL（`https://gemini.google.com/app/<thread_id>`）；后续轮次若 `bsk` 报错，直接用 `bsk open "<thread_url>"` 恢复会话，严禁新开空白页导致上下文中断。

---

## 0. 前置检查与状态板初始化
1. **分支隔离检查**：执行 `git branch --show-current` 与 `git status`，确保当前处于特性分支（**严禁直接在 `main` 分支上推进未验收改动**；若当前在 `main`，先切至独立分支如 `git checkout -b feat/pair-dev-...`）。
2. **生成/更新 `.pair-state.md`**：在项目根目录维护精简状态板 `.pair-state.md`（已被 `.gitignore` 忽略，严禁提交到 Git），仅记录：
   - **总目标**：本次任务范围与验收标准；
   - **硬性约束**：`GEMINI.md` 零成本锁（`USE_VERTEX_AI=false`、`offline_mode=True`）、6GB VRAM 上限、65 项单测基线与爆炸半径门禁；
   - **当前未决项与核心 Diff**：待审查问题及最新 `git --no-pager diff` 精简摘要（严禁粘贴整文件源码）。

---

## 1. 额度检查、动态最高级模型锁定与自动兜底
1. **连接与额度检测**：
   - 先运行 `bsk browsers` 检查 daemon 与 Chrome 扩展连接状态，并用 `bsk open "https://gemini.google.com/app" --browser <id>` 打开会话。
   - 点击左下角 `设置` → `用量限额`，读取当前方案（如 `PRO`）的 `当前用量（已使用 % / 重置时间）` 与 `每周限额（已使用 % / 重置时间）`，随后点击 `发起新对话` 返回主页（或重开已有线程 URL）。
2. **动态锁定最高级可用模型**（不写死模型名，不按任务难度降级）：
   - 通过 `bsk` 点击模式选择器打开模型下拉菜单并执行 `bsk sync`，实时读取菜单内所有模型选项的名称、版本号与描述文本。
   - **黑名单与不可用过滤**：排除黑名单模型（如 `4 Carbon`）及已灰显/禁用项；若选中某选项后弹出升级付费提示框，立即点击 `以后再说` 关闭并将其列入本次会话黑名单，重新选择次高可用项。
   - **能力优先选型**：当“当前用量”与“每周限额”均未超过 `85%` 时，从剩余可用选项中动态锁定当前能力最强、定位最高级的推理/编程模型（例如版本号最新，或描述中标注为更深入分析、最强推理、复杂任务专用的模型；若存在可用的 `思考等级` 子菜单，按需选择较高推理深度）。
3. **额度保护与自动降级兜底**：
   - **高水位额度降级**：仅当查看到的“当前用量”或“每周限额”**超过 `85%`** 时，才在可用选项中动态降级选择轻量/快速模型或“自动”模式以控制消耗。
   - **故障/耗尽单端兜底**：若出现 **`bsk daemon` 未启动/扩展未连接、网页端连续超时/不可达、或额度耗尽**，**不得卡死阻塞**——立即关闭或放弃浏览器会话并**自动降级为本地单端自审**（依托 `@skills/code-review-and-quality` 与 `@skills/code-simplification` 完成审查并推进 `/verify`），并在最终汇报中如实说明降级原因。

---

## 2. 审查闭环（Max 3 轮熔断）
在每轮交互中严格遵守以下规则：
1. **精简投喂（禁止长篇上下文）**：仅向网页版 Gemini 发送 `.pair-state.md` 摘要 + 精简 `git --no-pager diff`（及最近一次 `uv run pytest` 核心结果/报错前 25 行，单轮汇报严格控制精简）。
2. **架构师输出约束**：首轮 Prompt 明确要求网页端：
   - **只输出架构建议、风险点、Diff 审查意见与下一步最小可验证指令，禁止输出全量代码**；
   - 当所有目标完美达成且测试全绿时，在回复最后一行单独输出 `[ALL_TASKS_COMPLETED]`。
3. **本地落盘与状态刷新**：每轮按审查意见修改代码/配置，前台同步验证后更新 `.pair-state.md` 再发往下一轮。
4. **双条件熔断保护**：满足以下任一条件立即熔断停止循环，向用户汇报当前卡点、最新 Diff 与 `.pair-state.md` 状态：
   - 连续 **2 次**出现同类报错未能解决；
   - 达到最大轮次上限（**Max 3 轮**）仍未收到 `[ALL_TASKS_COMPLETED]`。

---

## 3. 防退化轻量迁移（A + B 迁移）
当同一网页线程交互过长导致页面明显变慢，或检测到模型出现**退化/幻觉**（遗忘 `.pair-state.md` 约束、重复已否决方案、输出格式漂移或上下文矛盾）时，自动触发防退化迁移：
1. **可选留档（Step A）**：如需保留旧线程原文，可通过 Gemini Voyager 两步导出（`导出对话记录` → 弹窗 `导出` → 底部消息栏 `导出` 执行 `bsk download`）归档到本地临时目录。
2. **开新线程直注状态板（Step B）**：点击 `发起新对话` 并按 **第 1 节** 选好模型，**不再向新会话搬运冗长历史对话**，直接通过 `bsk fill` 注入 **`.pair-state.md` + 最新 `git --no-pager diff`** 并要求确认同步；通过 `bsk read --selector "model-response"` 记录新线程 `page_url` 替换旧 URL 后继续闭环。

---

## 4. 收尾验收与临时文件清理
1. 当审查通过（收到 `[ALL_TASKS_COMPLETED]` 或完成降级自审）后，先执行 `bsk close` 释放浏览器会话（若使用了浏览器）。
2. 自动执行 `/verify` 门禁（前台非沙箱 `BypassSandbox: true` 下保持 `USE_VERTEX_AI=false` 零成本离线模式）：
   - `uv run ruff check .`
   - `uv run ruff format --check .`
   - `uv run pytest -q`（或 `-v`）
3. 门禁全绿后，**清理根目录临时状态板 `.pair-state.md`** 等临时文件，保持工作区干净，并向用户汇报改动清单、关键 Diff 与真实命令退出码（默认不自动 `git commit`，交由用户确认）。

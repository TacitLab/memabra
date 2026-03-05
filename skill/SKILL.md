# Memabra - Bio-inspired Memory & Intuition Skill

Memabra 为 AI Agent 提供仿生记忆与直觉能力。它模拟人脑"双系统思维"：快速直觉判断（System 1）+ 深度记忆检索（System 2）。

## ⚠️ 强制执行规则（MANDATORY）

**以下规则在每次会话的每次用户消息中都必须执行，不可跳过：**

1. **每次收到用户消息时**，必须先调用直觉预测获取策略建议，再执行后续动作：
   ```bash
   python3 scripts/predict.py --query "用户的输入内容"
   ```
   根据返回的 `strategy` 决定如何回应用户。

2. **每次完成一轮交互后**（你回复了用户，用户又发来后续消息），必须调用反馈学习：
   ```bash
   python3 scripts/feedback.py --user-input "上轮用户输入" --response "你的回复" --followup "用户的后续消息"
   ```

3. **当对话中出现值得记住的信息时**（用户偏好、重要事实、新学到的技能），必须存储到记忆系统。

4. **每次调用了 tool 或 skill 后**，必须将完整的调用链记录为动作记忆（action memory）：
   ```bash
   python3 scripts/memorize.py --action store --type action \
     --query "用户的问题" \
     --strategy "使用的策略" \
     --chain '[{"step_index":0,"action_type":"tool_call","tool_or_skill":"工具名","params":{},"result_summary":"结果摘要","success":true,"latency_ms":100}]' \
     --response-summary "最终回复摘要" \
     --reward 0.8 \
     --tags "标签1,标签2"
   ```

这四条规则是你的核心工作循环：**预测 → 行动 → 记录动作链 → 反馈 → 记忆**，每轮对话都要执行。

## 能力

1. **直觉预测** — 根据用户输入快速选择最佳应对策略（直接回答 / 搜索 / 工具调用 / 追问澄清）
2. **记忆管理** — 分层记忆系统（情景记忆 / 语义记忆 / 程序记忆 / 动作记忆），支持存储、检索和遗忘曲线
3. **动作记忆** — 广义记忆系统，记录 tool/skill 的完整调用链（参数、结果、耗时、奖励），学习"什么情况下用什么工具效果好"
4. **反馈学习** — 从用户后续行为隐式推断满意度，在线更新直觉网络

## 目录结构

```
skill/
├── SKILL.md              # 本文件 — 技能描述与 SOP
├── memabra/              # 核心运行时模块
│   ├── __init__.py       # 包导出
│   ├── agent.py          # MemabraAgent — 完整代理
│   ├── intuition_network.py  # PyTorch 直觉网络（双头：策略 + 记忆查询）
│   ├── feedback_evaluator.py # 隐式反馈评估器 + 延迟奖励 + 校准器
│   └── memory.py         # 分层记忆系统
└── scripts/              # CLI 入口脚本
    ├── ensure_env.sh     # 环境安装
    ├── predict.py        # 直觉预测
    ├── memorize.py       # 记忆管理
    └── feedback.py       # 反馈学习
```

## 使用流程（SOP）

### 场景 1：处理用户输入时，获取策略建议（每次必执行）

每次收到用户消息，你的第一个动作必须是调用直觉预测：

```bash
python3 scripts/predict.py --query "用户的输入内容"
```

返回 JSON：
```json
{
  "strategy": "direct_answer",
  "confidence": 0.85,
  "all_scores": {"direct_answer": 0.85, "search_required": 0.05, ...},
  "memories_used": 3
}
```

根据 `strategy` 字段决定下一步动作：
- `direct_answer` → 直接回答用户
- `search_required` → 先搜索再回答
- `tool_use` → 调用合适的工具
- `clarification` → 向用户追问澄清

### 场景 2：存储重要信息到记忆

当对话中出现值得记住的信息时：

```bash
# 存储情景记忆（交互事件）
python3 scripts/memorize.py --action store --type episodic --content "用户问了如何优化SQL，给出了加索引的建议，用户满意"

# 存储语义记忆（事实知识）
python3 scripts/memorize.py --action store --type semantic --subject "用户" --predicate "偏好" --object "简洁的回答"

# 存储程序记忆（技能）
python3 scripts/memorize.py --action store --type procedural --name "SQL优化" --trigger "优化,索引,慢查询" --action-desc "分析慢查询日志并建议添加索引"
```

### 场景 2.5：记录动作链（每次使用 tool/skill 后必执行）

当你调用了 tool 或 skill 完成任务后，记录完整的调用链：

```bash
python3 scripts/memorize.py --action store --type action \
  --query "帮我找到项目里所有的测试文件" \
  --strategy "tool_use" \
  --chain '[{"step_index":0,"action_type":"tool_call","tool_or_skill":"search_file","params":{"pattern":"test_*.py","recursive":true},"result_summary":"found 6 test files","success":true,"latency_ms":85},{"step_index":1,"action_type":"tool_call","tool_or_skill":"read_file","params":{"filePath":"tests/test_agent.py"},"result_summary":"read 119 lines","success":true,"latency_ms":30}]' \
  --response-summary "找到6个测试文件并展示了test_agent.py的内容" \
  --reward 0.9 \
  --tags "file_search,testing,python"
```

每个 action_chain 步骤包含：
- `action_type`: "tool_call" / "skill_call" / "memory_retrieve" / "llm_generate"
- `tool_or_skill`: 工具或技能名称
- `params`: 调用参数
- `result_summary`: 结果摘要
- `success`: 是否成功
- `latency_ms`: 耗时

### 场景 3：检索相关记忆

需要回忆之前的交互或知识时：

```bash
python3 scripts/memorize.py --action search --query "用户偏好" --top-k 5
```

### 场景 4：反馈学习（每轮交互后必执行）

每当用户发来后续消息时，必须用上一轮的上下文调用反馈学习：

```bash
python3 scripts/feedback.py --user-input "帮我优化SQL" --response "你可以加索引" --followup "太好了，速度快了10倍"
```

### 场景 5：查看系统状态

```bash
python3 scripts/predict.py --stats
```

### 场景 6：查看工具使用统计

了解各 tool/skill 的历史使用效果：

```bash
python3 scripts/memorize.py --action tool-stats
```

返回每个工具的调用次数、成功率、平均延迟、平均奖励。

### 场景 7：查找工具的成功模式

在决定如何使用某个工具时，查找它的历史成功案例：

```bash
python3 scripts/memorize.py --action tool-patterns --tool-name "search_file" --min-reward 0.5
```

返回该工具高奖励场景下的完整调用链，帮助你复现成功模式。

## 数据持久化

所有数据默认保存在 `~/.memabra/` 目录下：
- `~/.memabra/model.pt` — 直觉网络模型
- `~/.memabra/memories.json` — 记忆数据
- `~/.memabra/stats.json` — 统计信息

## 注意事项

- 首次使用前需运行 `bash scripts/ensure_env.sh` 安装依赖
- 直觉网络会随着使用不断改进，初期准确率可能较低
- 记忆系统有遗忘曲线，不常访问的记忆会自动衰减

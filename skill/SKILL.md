# Memabra - Bio-inspired Memory & Intuition System

Memabra 为 AI Agent 提供仿生记忆与直觉能力。它模拟人脑"双系统思维"：快速直觉判断（System 1）+ 深度记忆检索（System 2）。

## 能力

1. **直觉预测** - 根据用户输入快速选择最佳应对策略（直接回答/搜索/工具调用/追问澄清）
2. **记忆管理** - 分层记忆系统（情景记忆/语义记忆/程序记忆），支持存储、检索和遗忘曲线
3. **反馈学习** - 从用户后续行为隐式推断满意度，在线更新直觉网络

## 使用流程（SOP）

### 场景 1：处理用户输入时，获取策略建议

当你收到用户输入但不确定该如何处理时，调用直觉预测：

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

### 场景 3：检索相关记忆

需要回忆之前的交互或知识时：

```bash
python3 scripts/memorize.py --action search --query "用户偏好" --top-k 5
```

### 场景 4：反馈学习

完成一轮交互后，用用户的后续回复更新直觉网络：

```bash
python3 scripts/feedback.py --user-input "帮我优化SQL" --response "你可以加索引" --followup "太好了，速度快了10倍"
```

### 场景 5：查看系统状态

```bash
python3 scripts/predict.py --stats
```

## 数据持久化

所有数据默认保存在 `~/.memabra/` 目录下：
- `~/.memabra/model.pt` - 直觉网络模型
- `~/.memabra/memories.json` - 记忆数据
- `~/.memabra/stats.json` - 统计信息

## 注意事项

- 首次使用前需运行 `bash scripts/ensure_env.sh` 安装依赖
- 直觉网络会随着使用不断改进，初期准确率可能较低
- 记忆系统有遗忘曲线，不常访问的记忆会自动衰减

# Memabra

**Memabra** = **Mem**ory（记忆）+ Cere**bra**（大脑皮层）

一个受生物记忆机制启发的 AI Agent 记忆与直觉 **Skill**，专为 [OpenClaw](https://github.com/jasons20/openclaw) 设计。

🌏 [English Version (英文版本)](README.md)

---

## 它能做什么

人类解决问题时，首先依赖**直觉**快速判断该用什么方式处理，然后调用相关记忆，最后执行。如果结果有效，这个路径会被强化；如果失败，则触发深度搜索和模式修正。

Memabra 将这种双系统思维作为 OpenClaw Skill 引入 AI Agent：

1. **直觉预测** — 神经网络选择最佳策略（直接回答 / 搜索 / 工具调用 / 追问澄清）
2. **分层记忆** — 情景、语义、程序三层记忆，支持遗忘曲线
3. **反馈学习** — 从用户行为隐式推断满意度，在线更新网络

## 项目结构

```
memabra/
├── skill/                            # OpenClaw Skill（自包含）
│   ├── SKILL.md                      # 技能描述与 SOP
│   ├── memabra/                      # 核心运行时模块
│   │   ├── __init__.py               # 包导出
│   │   ├── agent.py                  # MemabraAgent — 完整代理
│   │   ├── intuition_network.py      # PyTorch 双头网络（策略 + 记忆查询）
│   │   ├── feedback_evaluator.py     # 隐式反馈评估器 + 延迟奖励 + 校准器
│   │   └── memory.py                 # 分层记忆系统
│   └── scripts/                      # CLI 入口脚本
│       ├── ensure_env.sh             # 环境安装
│       ├── predict.py                # 直觉预测
│       ├── memorize.py               # 记忆管理
│       └── feedback.py               # 反馈学习
├── tests/                            # 测试套件
│   ├── conftest.py                   # 路径配置
│   ├── test_intuition.py             # 直觉网络测试
│   ├── test_memory.py                # 记忆系统测试
│   ├── test_feedback.py              # 反馈系统测试
│   ├── test_agent.py                 # Agent 集成测试
│   └── test_skill_scripts.py         # Skill CLI 端到端测试
├── config/
│   └── default.yaml                  # 默认配置
├── docs/                             # 设计文档
└── pyproject.toml                    # 项目构建配置
```

## 快速开始（OpenClaw Skill）

### 安装

```bash
cd skill
bash scripts/ensure_env.sh
```

### 使用

```bash
# 直觉预测 — 获取策略建议
python3 scripts/predict.py --query "如何优化数据库性能"

# 存储情景记忆
python3 scripts/memorize.py --action store --type episodic --content "用户问了SQL优化的问题"

# 存储语义记忆（事实）
python3 scripts/memorize.py --action store --type semantic --subject "用户" --predicate "偏好" --object "简洁的回答"

# 存储程序记忆（技能）
python3 scripts/memorize.py --action store --type procedural --name "SQL优化" --trigger "优化,索引,慢查询" --action-desc "分析慢查询日志并建议添加索引"

# 检索记忆
python3 scripts/memorize.py --action search --query "用户偏好" --top-k 5

# 反馈学习
python3 scripts/feedback.py --user-input "帮我优化SQL" --response "你可以加索引" --followup "太好了，速度快了10倍"

# 查看系统状态
python3 scripts/predict.py --stats
```

详见 [`skill/SKILL.md`](skill/SKILL.md) 获取完整的 5 个使用场景 SOP。

## 核心模块

### 直觉网络 IntuitionNetwork (`intuition_network.py`)
基于 PyTorch 的神经策略网络，双头架构：
- **策略选择头**：通过温度缩放 softmax 选择最佳策略
- **记忆查询头**：生成 L2 归一化的查询向量用于记忆检索

通过 REINFORCE 策略梯度训练，支持熵正则化。

### 分层记忆 HierarchicalMemory (`memory.py`)
三层记忆系统：
- **情景记忆**：具体交互和结果的记录
- **语义记忆**：以"主语-谓词-宾语"三元组存储的事实知识
- **程序记忆**：带有触发模式和成功率追踪的技能

支持 Ebbinghaus 遗忘曲线和磁盘持久化。

### 隐式反馈评估器 ImplicitEvaluator (`feedback_evaluator.py`)
无需用户显式评分，通过检测以下信号推断满意度：
- 正向/负向关键词检测
- 放弃信号识别
- 语义重复分析
- 重述模式检测
- 上下文话题切换满意度推断

### 核心代理 MemabraAgent (`agent.py`)
完整循环：输入嵌入 → 策略预测 → 记忆检索 → 执行 → 收集反馈 → 更新网络。

## 测试

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## 数据持久化

所有数据存储在 `~/.memabra/`：
- `model.pt` — 直觉网络权重
- `memories.json` — 记忆数据
- `stats.json` — 统计信息

## 文档

- [架构设计](docs/architecture.md) — 系统整体架构
- [核心机制](docs/core-mechanisms.md) — 直觉网络、记忆检索、执行循环
- [隐式反馈系统](docs/feedback-system.md) — 如何从用户行为推断满意度
- [开发路线图](docs/roadmap.md) — 实现计划

## 环境要求

- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy >= 1.24.0

## 开源协议

MIT

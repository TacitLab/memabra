# Memabra

**Memabra** = **Mem**ory（记忆）+ Cere**bra**（大脑皮层）

一个受生物记忆机制启发的 AI Agent 记忆与直觉系统。

🌏 [English Version (英文版本)](README.md)

---

## 核心理念

人类解决问题时，首先依赖**直觉**快速判断该用什么方式处理，然后调用相关记忆，最后执行。如果结果有效，这个路径会被强化；如果失败，则触发深度搜索和模式修正。

Memabra 尝试将这种双系统思维（直觉 + 分析）引入 AI Agent 的记忆管理。

## 设计原则

1. **生物启发**：借鉴人脑的记忆形成、巩固、遗忘机制
2. **隐式反馈**：不依赖用户显式评价，从交互流中自然读取信号
3. **端到端学习**：直觉网络随经验自我优化
4. **渐进复杂**：从简单原型开始，逐步演化

## 项目结构

```
memabra/
├── src/memabra/
│   ├── __init__.py               # 包导出
│   ├── agent.py                  # MemabraAgent — 核心集成
│   ├── intuition_network.py      # PyTorch 直觉网络（双头：策略选择 + 记忆查询）
│   ├── intuition.py              # 轻量直觉网络（基于余弦相似度）
│   ├── feedback_evaluator.py     # 隐式反馈评估器 + 延迟奖励分配器 + 校准器
│   ├── feedback.py               # 简化版反馈系统
│   └── memory.py                 # 分层记忆系统（情景 / 语义 / 程序）
├── skill/                        # OpenClaw 技能集成
│   ├── SKILL.md                  # 技能描述与 SOP
│   └── scripts/
│       ├── ensure_env.sh         # 环境安装脚本
│       ├── predict.py            # 直觉预测 CLI
│       ├── memorize.py           # 记忆管理 CLI
│       └── feedback.py           # 反馈学习 CLI
├── tests/                        # 测试套件（60 个测试用例）
│   ├── test_intuition.py         # 直觉网络测试
│   ├── test_memory.py            # 记忆系统测试
│   ├── test_feedback.py          # 反馈系统测试
│   ├── test_agent.py             # Agent 集成测试
│   └── test_skill_scripts.py     # Skill CLI 端到端测试
├── config/
│   └── default.yaml              # 默认配置
├── docs/                         # 设计文档
└── pyproject.toml                # 项目构建配置
```

## 核心模块

### 直觉网络 IntuitionNetwork (`intuition_network.py`)
基于 PyTorch 的神经策略网络，双头架构：
- **策略选择头**：选择最佳行动策略（直接回答、搜索、调用工具、澄清）
- **记忆查询头**：生成用于记忆检索的查询向量

通过 REINFORCE 策略梯度训练，支持熵正则化和自适应温度参数。

### 分层记忆 HierarchicalMemory (`memory.py`)
模拟人类认知的三层记忆系统：
- **情景记忆**：具体交互和结果的记录
- **语义记忆**：以"主语-谓词-宾语"三元组存储的事实知识
- **程序记忆**：带有触发模式的技能和操作流程

支持 Ebbinghaus 遗忘曲线和磁盘持久化（保存/加载）。

### 隐式反馈评估器 ImplicitEvaluator (`feedback_evaluator.py`)
无需用户显式评分，通过检测以下信号推断满意度：
- 正向/负向关键词
- 放弃信号
- 语义重复（用户重复相同问题）
- 重述模式
- 话题切换及上下文满意度推断

### 核心代理 MemabraAgent (`agent.py`)
编排完整循环：输入嵌入 → 策略预测 → 记忆检索 → 执行 → 收集反馈 → 更新网络。

## 快速开始

```bash
# 以开发模式安装
pip install -e .

# 运行内置演示
python -m memabra.agent
```

### 使用示例

```python
from memabra import MemabraAgent

agent = MemabraAgent()

# 处理用户输入
result = agent.process("今天天气怎么样？")
print(result['response'])
print(f"策略: {result['strategy']}, 置信度: {result['confidence']:.2f}")

# 用户后续回复（反馈自动推断）
feedback = agent.on_user_followup("谢谢，我知道了")
print(f"反馈: {feedback['feedback_type']}, 奖励: {feedback['reward']:+.2f}")

# 保存模型状态
agent.save("model.pt")
```

## OpenClaw 技能集成

Memabra 可作为 [OpenClaw](https://github.com/jasons20/openclaw) Skill 使用，通过 CLI 脚本为 AI Agent 提供仿生记忆与直觉能力。

### CLI 脚本

```bash
# 直觉预测 — 获取用户输入的策略建议
python3 skill/scripts/predict.py --query "如何优化数据库性能"

# 记忆存储 — 保存情景/语义/程序记忆
python3 skill/scripts/memorize.py --action store --type episodic --content "用户问了SQL优化的问题"

# 记忆检索 — 检索相关记忆
python3 skill/scripts/memorize.py --action search --query "用户偏好" --top-k 5

# 反馈学习 — 从用户后续回复更新直觉网络
python3 skill/scripts/feedback.py --user-input "帮我优化SQL" --response "你可以加索引" --followup "太好了，速度快了10倍"

# 查看系统状态
python3 skill/scripts/predict.py --stats
```

详见 [`skill/SKILL.md`](skill/SKILL.md) 获取完整的 5 个使用场景 SOP。

## 测试

共 60 个测试用例，覆盖所有模块：

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行全部测试
python -m pytest tests/ -v
```

| 测试文件 | 覆盖范围 | 用例数 |
|---------|---------|-------|
| `test_intuition.py` | 直觉网络 | 12 |
| `test_memory.py` | 分层记忆 | 12 |
| `test_feedback.py` | 反馈评估器 | 12 |
| `test_agent.py` | Agent 集成 | 10 |
| `test_skill_scripts.py` | Skill CLI 端到端 | 14 |

## 文档

- [架构设计](docs/architecture.md) - 系统整体架构
- [核心机制](docs/core-mechanisms.md) - 直觉网络、记忆检索、执行循环
- [隐式反馈系统](docs/feedback-system.md) - 如何从用户行为推断满意度
- [开发路线图](docs/roadmap.md) - 实现计划

## 环境要求

- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- scikit-learn >= 1.3.0

## 开源协议

MIT

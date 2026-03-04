"""
Integration tests for OpenClaw Skill scripts.

Tests the CLI scripts as they would be called by OpenClaw,
verifying JSON output format and correctness.
"""

import json
import os
import subprocess
import sys
import pytest

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'skill', 'scripts')


def run_script(script_name: str, args: list[str], env_override: dict = None) -> dict:
    """Run a skill script and parse its JSON output."""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    cmd = [sys.executable, script_path] + args

    env = os.environ.copy()
    # Use a temp data dir to avoid polluting real data
    if env_override:
        env.update(env_override)

    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=60
    )

    if result.returncode != 0:
        pytest.fail(f"Script failed with code {result.returncode}:\n{result.stderr}")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        pytest.fail(f"Script output is not valid JSON:\n{result.stdout}")


class TestPredictScript:
    """Tests for predict.py CLI script."""

    def test_predict_returns_json(self):
        output = run_script("predict.py", ["--query", "今天天气怎么样"])

        assert "strategy" in output
        assert "confidence" in output
        assert "response" in output
        assert output["strategy"] in [
            "direct_answer", "search_required", "tool_use", "clarification"
        ]

    def test_predict_different_queries(self):
        q1 = run_script("predict.py", ["--query", "写一段Python代码"])
        q2 = run_script("predict.py", ["--query", "你好"])

        # Both should return valid results
        assert q1["strategy"] in [
            "direct_answer", "search_required", "tool_use", "clarification"
        ]
        assert q2["strategy"] in [
            "direct_answer", "search_required", "tool_use", "clarification"
        ]

    def test_stats(self):
        output = run_script("predict.py", ["--stats"])
        assert isinstance(output, dict)


class TestMemorizeScript:
    """Tests for memorize.py CLI script."""

    def test_store_episodic(self):
        output = run_script("memorize.py", [
            "--action", "store",
            "--type", "episodic",
            "--content", "用户问了天气，回答了晴天"
        ])
        assert output["status"] == "ok"
        assert output["type"] == "episodic"
        assert "id" in output

    def test_store_semantic(self):
        output = run_script("memorize.py", [
            "--action", "store",
            "--type", "semantic",
            "--subject", "Python",
            "--predicate", "是",
            "--object", "编程语言"
        ])
        assert output["status"] == "ok"
        assert output["type"] == "semantic"

    def test_store_procedural(self):
        output = run_script("memorize.py", [
            "--action", "store",
            "--type", "procedural",
            "--name", "代码审查",
            "--trigger", "review,审查,代码检查",
            "--action-desc", "检查代码质量和安全性"
        ])
        assert output["status"] == "ok"
        assert output["type"] == "procedural"

    def test_search(self):
        # Store first, then search
        run_script("memorize.py", [
            "--action", "store", "--type", "episodic",
            "--content", "讨论了数据库索引优化方案"
        ])

        output = run_script("memorize.py", [
            "--action", "search",
            "--query", "数据库优化",
            "--top-k", "3"
        ])
        assert "results" in output
        assert "total_found" in output

    def test_recent(self):
        output = run_script("memorize.py", [
            "--action", "recent", "--top-k", "5"
        ])
        assert "recent" in output
        assert "count" in output

    def test_store_missing_content_error(self):
        """Should fail gracefully when required args are missing."""
        script_path = os.path.join(SCRIPTS_DIR, "memorize.py")
        result = subprocess.run(
            [sys.executable, script_path, "--action", "store", "--type", "episodic"],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode != 0


class TestFeedbackScript:
    """Tests for feedback.py CLI script."""

    def test_feedback_from_followup(self):
        output = run_script("feedback.py", [
            "--user-input", "帮我优化SQL查询",
            "--response", "建议你添加索引",
            "--followup", "谢谢，速度快了很多"
        ])
        assert "strategy_used" in output
        assert "feedback" in output
        assert output["feedback"]["reward"] > 0

    def test_negative_feedback(self):
        output = run_script("feedback.py", [
            "--user-input", "查一下股票价格",
            "--response", "股价是100元",
            "--followup", "错了，没用"
        ])
        assert output["feedback"]["reward"] < 0

    def test_manual_feedback(self):
        output = run_script("feedback.py", [
            "--user-input", "测试输入",
            "--strategy", "direct_answer",
            "--reward", "0.8"
        ])
        assert "update" in output
        assert output["reward"] == 0.8

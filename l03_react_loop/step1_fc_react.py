"""
L03 · step1 — 【课程的做法】function calling + ReAct prompt
==========================================================
想法：用原生 function calling 执行工具，同时要求模型在 content 里写出 Thought。

⚠️ 这个文件是拿来"证伪"的。跑它不是为了得到答案，
   是为了亲眼看到：模型调工具时，Thought 根本不出来。

跑法：PYTHONIOENCODING=utf-8 python l03_react_loop/step1_fc_react.py
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

MODEL = "glm-4-flash"
MAX_STEPS = 8


# ── 工具 ────────────────────────────────────────────────────
def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculator(expression: str) -> str:
    # ⚠️ 白名单里【必须包含 %】。这是 L03 实测踩过的坑：
    #    漏掉 %，模型遇到「求余数」会死循环 8 轮，最后把「商的整数部分」当成余数。
    allowed = set("0123456789+-*/%.() ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含非法字符"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{e}"


TOOL_REGISTRY = {"get_current_time": get_current_time, "calculator": calculator}

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，如 '12 * 34'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"],
            },
        },
    },
]


# ── ReAct prompt：要求模型每步先写 Thought ──────────────────
REACT_SYSTEM_PROMPT = """你是一个会使用工具的智能助手。面对用户的问题，请严格按以下方式工作：

每一步，你都要先在回答里写出你的【思考】，然后再决定是否调用工具：
- 用"💭 Thought:" 开头，写明你这一步的推理。
- 如果需要调用工具来获取信息，在 Thought 之后正常调用工具。
- 如果已经可以回答用户了，用"✅ Final Answer:" 开头给出最终答案。

记住：每一步都要先写 Thought 再行动，让推理过程清晰可见。"""


def run(question: str, max_steps: int = MAX_STEPS):
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    stats = []  # (这一轮有没有 Thought, 这一轮调了几个工具)

    for step in range(1, max_steps + 1):
        print(f"\n{'━' * 56}\n🔄 第 {step} 轮")

        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        has_thought = bool(msg.content and msg.content.strip())
        n_calls = len(msg.tool_calls or [])
        stats.append((has_thought, n_calls))

        # ★ 诊断行：这一轮它到底说了什么、调了几个工具
        print(
            f"   💭 Thought 有内容吗？ {'有 ✅' if has_thought else '没有 ❌'}"
            f"（content = {repr(msg.content)[:60]}）"
        )
        print(f"   🔧 工具调用：{n_calls} 个")

        if n_calls:
            messages.append(msg.model_dump())
            for call in msg.tool_calls:
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                print(f"      Action: {call.function.name}({args})")
                result = str(TOOL_REGISTRY[call.function.name](**args))
                print(f"      Observation: {result}")
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
        else:
            print(f"   💬 最终回答：{msg.content}")
            return stats, msg.content

    print(f"\n⚠️ 走满 {max_steps} 轮还没结束，强制停止。")
    return stats, None


def main():
    print("=" * 60)
    print("L03 · step1 — function calling + ReAct prompt（课程的做法）")
    print("=" * 60)
    q = "现在几点？请把当前小时数和分钟数相加，再用结果除以 7，告诉我余数是多少。"
    print(f"\n👤 {q}")

    stats, ans = run(q)

    print("\n" + "=" * 60)
    print("📊 统计：每一轮 (有没有 Thought, 调了几个工具)")
    print("=" * 60)
    for i, (t, c) in enumerate(stats, 1):
        print(f"   第 {i} 轮：Thought={'有' if t else '无'}  工具={c}")
    n_thought = sum(1 for t, _ in stats if t)
    print(f"\n   ★ 有 Thought 的轮数：{n_thought} / {len(stats)}")
    print(f"   ★ 最终答案：{ans}")
    print(
        "\n   如果 Thought 全是 0 —— 说明 prompt 里「每步都要写 Thought」的要求被无视了。"
    )


if __name__ == "__main__":
    main()

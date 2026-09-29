"""
L06 · Plan-and-Execute vs ReAct —— 对照实验台
==============================================
课程说：复杂任务用 Plan-and-Execute，探索性任务用 ReAct。
本实验台用同一个复杂任务把这两个都跑一遍，用【数字】看谁更好。

同时对比两个版本的 Plan-and-Execute，让你看清它失败在哪：
  v1（课程风格）：规划 prompt 里直接写「输出格式为 {...}」，执行时每步新建 messages
  v2（修好）：  规划时把格式说明与任务隔离，执行时共享 messages

跑法：PYTHONIOENCODING=utf-8 python l06_planning/plan_vs_react.py
"""

import json
import os
import re
import sys

from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

MODEL = "glm-4-flash"

TASK = "帮我查北京、上海、广州三个城市的天气，找出最热的城市，算出最热和最冷的温差，最后给出穿衣建议。"

# 真值：北京 25 / 上海 28 / 广州 30 → 最热=广州(30)，最冷=北京(25)，温差=5
TRUTH = {"说出最热是广州": "广州", "算出温差 5": "5"}


# ════════════════════════════════════════════════════════════
# 工具
# ════════════════════════════════════════════════════════════
WEATHER = {"北京": ("晴", 25), "上海": ("多云", 28), "广州": ("雨", 30),
           "深圳": ("阴", 29), "杭州": ("晴", 26)}


def get_weather(city: str, unit: str = "摄氏度") -> str:
    """查天气（模拟数据，不联网）。"""
    if city not in WEATHER:
        return f"没有 {city} 的天气数据。支持：{list(WEATHER)}"
    cond, t = WEATHER[city]
    return f"{city}：{cond}，{t}°C"


def calculate(expression: str) -> str:
    """计算数学表达式。支持 + - * / % 和括号。"""
    allowed = set("0123456789+-*/%.() ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含不支持的字符"
    try:
        return str(eval(expression))
    except ZeroDivisionError:
        return "错误：除数不能为 0"
    except Exception as e:
        return f"计算错误：{e}"


TOOL_REGISTRY = {"get_weather": get_weather, "calculate": calculate}

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "查询指定城市的天气。当需要知道某城市温度/天气状况时使用。",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "城市名，如'北京'"}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "计算数学表达式。支持 + - * / % 和括号。需要精确计算时使用。",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "数学表达式，如 '30-25'"}},
            "required": ["expression"]}}},
]


def run_tool_calls(msg, calls_log):
    """执行一轮里的所有工具调用，返回要追加进 messages 的消息列表。"""
    out = [msg.model_dump()]
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        result = TOOL_REGISTRY[tc.function.name](**args)
        calls_log.append(tc.function.name)
        out.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return out


# ════════════════════════════════════════════════════════════
# 阶段 1：Plan —— 两个版本的 prompt
# ════════════════════════════════════════════════════════════
PLAN_BAD = """你是一个任务规划专家。请把下面的任务分解成清晰的执行步骤。

要求：
1. 输出一个 JSON 对象，格式为 {{"steps": ["步骤1", "步骤2", ...]}}
2. 每个步骤是一个具体的、可执行的动作
3. 步骤要覆盖完成任务需要的所有操作（包括查数据、计算、总结等）
4. 只输出 JSON，不要其他内容

任务：{task}

步骤计划："""

PLAN_GOOD = """你是一个任务规划专家。请把下面的任务分解成清晰的执行步骤。

【重要】你的输出格式要求如下，这些要求本身【不属于任务内容】，
绝对不要把"整理格式""输出JSON""用代码处理"之类的动作写进步骤列表。
步骤列表里只能有：为了完成用户任务、需要真正去做的业务动作。

<输出格式>
{{"steps": ["步骤1", "步骤2", ...]}}
</输出格式>

<用户任务>
{task}
</用户任务>

现在输出步骤列表（只输出 JSON）："""


def plan(task: str, good: bool = True):
    prompt = (PLAN_GOOD if good else PLAN_BAD).format(task=task)
    content = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content.strip()

    m = re.search(r"\{.*\}", content, re.DOTALL)     # 比课程的正则宽松，能容忍嵌套
    if m:
        try:
            return json.loads(m.group(0)).get("steps", [])
        except json.JSONDecodeError:
            pass
    return [content]


# ════════════════════════════════════════════════════════════
# 阶段 2：Execute —— 两个版本
# ════════════════════════════════════════════════════════════
def execute_isolated(task, steps):
    """v1：每一步【新建】messages。

    ★ 病根：工具调用的原始结果不跨步共享。后面的步骤只看到"总结后的文字"，
      于是它不确定自己查过没有 → 重复调工具。
    """
    completed = []
    calls = []
    for i, step in enumerate(steps, 1):
        ctx = (f"原始任务：{task}\n完整计划：{json.dumps(steps, ensure_ascii=False)}\n"
               f"已完成步骤的结果：{json.dumps(completed, ensure_ascii=False) if completed else '（还没有）'}\n\n"
               f"现在请执行【步骤 {i}】：{step}")
        messages = [{"role": "user", "content": ctx}]        # ← 每步都从零开始
        for _ in range(3):
            msg = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS_SPEC, tool_choice="auto"
            ).choices[0].message
            if msg.tool_calls:
                messages += run_tool_calls(msg, calls)
            else:
                completed.append(f"{step} → {msg.content}")
                break
        else:
            completed.append(f"{step} → （撞上限）")
    return calls, (completed[-1] if completed else "")


def execute_shared(task, steps):
    """v2：整个执行阶段【共享一份 messages】。

    ★ 修法：把工具结果留在上下文里，后面的步骤能直接看到原始数据，
      不需要重查。这也是 agent 框架里 session 的标准做法。
    """
    calls = []
    messages = [{"role": "user", "content": (
        f"请按下面的计划完成这个任务。你可以调用工具。\n\n"
        f"原始任务：{task}\n计划：{json.dumps(steps, ensure_ascii=False)}\n\n"
        f"请逐个完成，全部做完之后，用自然语言给出最终答案。"
    )}]
    for _ in range(12):
        msg = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SPEC, tool_choice="auto"
        ).choices[0].message
        if msg.tool_calls:
            messages += run_tool_calls(msg, calls)
        else:
            return calls, (msg.content or "")
    return calls, "（撞上限）"


# ════════════════════════════════════════════════════════════
# 基线：ReAct（边想边做，无预先计划）
# ════════════════════════════════════════════════════════════
def react(task, max_steps=12):
    calls = []
    messages = [{"role": "user", "content": task}]
    for _ in range(max_steps):
        msg = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SPEC, tool_choice="auto"
        ).choices[0].message
        if msg.tool_calls:
            messages += run_tool_calls(msg, calls)
        else:
            return calls, (msg.content or "")
    return calls, "（撞上限）"


# ════════════════════════════════════════════════════════════
# 对照
# ════════════════════════════════════════════════════════════
LEAK_WORDS = ["JSON", "json", "Python", "python", "格式化", "字符串", "编程语言", "输出格式"]


def score(answer: str) -> int:
    return sum(1 for v in TRUTH.values() if v in (answer or ""))


def run_plan_execute(good: bool, shared: bool):
    steps = plan(TASK, good=good)
    leaked = [s for s in steps if any(w in s for w in LEAK_WORDS)]
    calls, ans = (execute_shared if shared else execute_isolated)(TASK, steps)
    return steps, leaked, calls, ans


def compare(runs: int = 2):
    print("=" * 74)
    print("Plan-and-Execute vs ReAct —— 同一个复杂任务")
    print(f"任务：{TASK}")
    print(f"真值：最热=广州(30°C)  最冷=北京(25°C)  温差=5°C   每个变体跑 {runs} 次")
    print("=" * 74)

    variants = [
        ("P-E v1（课程风格）", lambda: run_plan_execute(good=False, shared=False)),
        ("P-E v2（修好）", lambda: run_plan_execute(good=True, shared=True)),
    ]

    for label, fn in variants:
        print(f"\n{'-' * 74}\n【{label}】\n{'-' * 74}")
        for i in range(1, runs + 1):
            steps, leaked, calls, ans = fn()
            print(f"\n  [{i}] 计划 {len(steps)} 步，元指令泄漏 {len(leaked)} 步")
            for s in steps:
                print(f"        {'⚠️' if s in leaked else '  '} {s[:64]}")
            print(f"      工具调用 {len(calls)} 次：{calls}")
            print(f"      终答 [{score(ans)}/2]：{(ans or '')[:80]!r}")

    print(f"\n{'-' * 74}\n【ReAct（基线）】\n{'-' * 74}")
    for i in range(1, runs + 1):
        calls, ans = react(TASK)
        print(f"\n  [{i}] 工具调用 {len(calls)} 次：{calls}")
        print(f"      终答 [{score(ans)}/2]：{(ans or '')[:100]!r}")


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    compare(runs)
    print("\n" + "=" * 74)
    print("看三件事：")
    print("  ① 计划里有几步是【关于输出格式】的？（那就是元指令泄漏）")
    print("  ② 工具调用多少次？重复查同一个城市了吗？")
    print("  ③ 最终交给用户的，是自然语言答案，还是一段 JSON 代码？")
    print("=" * 74)


if __name__ == "__main__":
    main()

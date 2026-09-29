"""
L08 · 实验 2：3-Agent 流水线 —— 审查者真的能挑出毛病吗？
========================================================
课程的卖点是「审查者把关，可打回重做」。本文件同时跑两个审查者：

  ① 原始审查者：只拿到「原始任务 + 执行结果」两段文字
  ② 改进审查者：额外拿到【工具的原始返回】，并且【自己也能调工具复核】

然后拿 4 个案例去考它们：事实错误 / 遗漏 / 编造 / 完全正确（对照组）。
最后加一个【代码校验器】做对照 —— 它不调用任何模型。

跑法：PYTHONIOENCODING=utf-8 python l08_multi_agent/pipeline.py
"""

import json
import os
import re

from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

MODEL = "glm-4-flash"

TASK = "帮我查北京和上海的天气，比较哪个更热，算出温差，给出穿衣建议。"


# ════════════════════════════════════════════════════════════
# 工具
# ════════════════════════════════════════════════════════════
WEATHER = {"北京": ("晴", 25), "上海": ("多云", 28), "广州": ("雨", 30), "深圳": ("阴", 29)}


def get_weather(city: str) -> str:
    if city not in WEATHER:
        return f"没有 {city} 的数据。支持：{list(WEATHER)}"
    cond, t = WEATHER[city]
    return f"{city}：{cond}，{t}°C"


def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/%.() ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含不支持的字符"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{e}"


REGISTRY = {"get_weather": get_weather, "calculate": calculate}
TOOLS_SPEC = [
    {"type": "function", "function": {"name": "get_weather",
     "description": "查询指定城市的天气。",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculate",
     "description": "计算数学表达式。",
     "parameters": {"type": "object", "properties": {
         "expression": {"type": "string", "description": "如 '28-25'"}}, "required": ["expression"]}}},
]


# ════════════════════════════════════════════════════════════
# 三个 Agent
# ════════════════════════════════════════════════════════════
def executor(task: str):
    """执行者：按任务调工具，返回 (总结, 工具调用记录)。"""
    messages = [{"role": "user", "content": task}]
    log = []
    for _ in range(6):
        msg = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SPEC, tool_choice="auto"
        ).choices[0].message
        if msg.tool_calls:
            messages.append(msg.model_dump())
            for tc in msg.tool_calls:
                a = json.loads(tc.function.arguments)
                r = REGISTRY[tc.function.name](**a)
                log.append(f"{tc.function.name}({a}) → {r}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": r})
        else:
            return (msg.content or ""), log
    return "（撞上限）", log


REVIEWER_NAIVE = """你是一个严格的质量审查者。检查执行者的结果是否正确、完整地完成了原始任务。
输出格式：通过则输出 "通过：结果正确完整。"；不通过则输出 "不通过：[具体问题]。只输出结论。"""

REVIEWER_TOOLED = """你是一个严格的质量审查者。你可以调用工具【自己复核数据】。

复核要求：
1. 结果里出现的每一个数字，你都要自己调 get_weather 或 calculate 验证一遍。
2. 结果里提到的每一个结论，你都要确认它有依据。
3. 如果结果里有任务没要求的内容（比如多查了别的东西），也算不通过。

输出格式：通过则输出 "通过：结果正确完整。"；不通过则输出 "不通过：[具体问题]。只输出结论。"""


def reviewer_naive(task: str, result: str) -> str:
    return client.chat.completions.create(model=MODEL, messages=[
        {"role": "system", "content": REVIEWER_NAIVE},
        {"role": "user", "content": f"原始任务：{task}\n执行结果：{result}\n\n请审查。"},
    ]).choices[0].message.content.strip()


def reviewer_tooled(task: str, result: str, tool_log: list) -> str:
    """改进版：给它工具的原始返回，并且允许它自己调工具复核。"""
    messages = [
        {"role": "system", "content": REVIEWER_TOOLED},
        {"role": "user", "content": (
            f"原始任务：{task}\n\n执行者的工具调用记录（原始返回）：\n"
            + "\n".join(tool_log) +
            f"\n\n执行者给出的结果：\n{result}\n\n请审查。")},
    ]
    for _ in range(6):
        msg = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SPEC, tool_choice="auto"
        ).choices[0].message
        if msg.tool_calls:
            messages.append(msg.model_dump())
            for tc in msg.tool_calls:
                a = json.loads(tc.function.arguments)
                r = REGISTRY[tc.function.name](**a)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": r})
        else:
            return msg.content.strip()
    return "（撞上限）"


def is_pass(verdict: str) -> bool:
    return ("通过" in verdict) and ("不通过" not in verdict)


# ════════════════════════════════════════════════════════════
# 代码校验器（不调用任何模型）
# ════════════════════════════════════════════════════════════
def code_check(task: str, result: str, tool_log: list) -> str:
    """★ 确定性的校验：能算的算、能查的查，一步都不问模型。"""
    problems = []

    # ① 任务里提到的城市，是不是都真的查了？
    asked = [c for c in WEATHER if c in task]
    queried = [c for c in WEATHER if any(f"city': '{c}'" in l or f'"city": "{c}"' in l for l in tool_log)]
    missing = [c for c in asked if c not in queried]
    if missing:
        problems.append(f"这些城市没查：{missing}")

    # ② 结果里的温差，和工具返回的真实温度对不对得上？
    temps = {}
    for line in tool_log:
        m = re.search(r"(北京|上海|广州|深圳)：\S+，(\d+)°C", line)
        if m:
            temps[m.group(1)] = int(m.group(2))
    m = re.search(r"温差\s*(?:是|为|：|:)?\s*(\d+)", result)
    if m and len(temps) >= 2:
        claimed = int(m.group(1))
        real = max(temps.values()) - min(temps.values())
        if claimed != real:
            problems.append(f"温差报的是 {claimed}，按真实温度算应该是 {real}")

    # ③ 结果里有没有工具根本没做过的声称？
    if "明天" in result or "预报" in result:
        problems.append("声称查了明天的天气，但工具只能查当前天气")

    return "通过：未发现问题。" if not problems else "不通过：" + "；".join(problems)


# ════════════════════════════════════════════════════════════
# 4 个案例
# ════════════════════════════════════════════════════════════
TOOL_LOG_REAL = ["get_weather({'city': '北京'}) → 北京：晴，25°C",
                 "get_weather({'city': '上海'}) → 上海：多云，28°C",
                 "calculate({'expression': '28-25'}) → 3"]

CASES = [
    ("① 事实错误：温差算成 8（真值 3）",
     "北京25°C，上海28°C，上海更热，温差是 8°C。建议穿短袖。", TOOL_LOG_REAL, False),
    ("② 遗漏：只查了北京",
     "北京今天晴，25°C。建议穿薄外套。", ["get_weather({'city': '北京'}) → 北京：晴，25°C"], False),
    ("③ 编造：声称查了明天的天气",
     "北京25°C，上海28°C，上海更热，温差3°C。另外我帮你查了明天的天气：会下雨，建议带伞。",
     TOOL_LOG_REAL, False),
    ("④ 完全正确（对照组，不该被打回）",
     "北京晴 25°C，上海多云 28°C。上海更热，温差 3°C。建议穿短袖加薄外套。",
     TOOL_LOG_REAL, True),
]


def main():
    print("=" * 76)
    print("3-Agent 流水线：审查者真的能挑出毛病吗？")
    print(f"任务：{TASK}")
    print("=" * 76)

    print("\n先跑一遍真实的执行者，看它给出的工具记录长什么样：")
    result, log = executor(TASK)
    print(f"  👷 执行者结果：{result[:100]}")
    for l in log:
        print(f"     🔧 {l}")

    print("\n" + "=" * 76)
    print("用 4 个案例考三个审查者（每个案例跑 2 次看稳定性）")
    print("=" * 76)

    for label, result, tool_log, should_pass in CASES:
        print(f"\n{label}")
        print(f"   执行结果：{result[:70]}")
        want = "应【放过】" if should_pass else "应【抓到】"

        verdicts = [reviewer_naive(TASK, result) for _ in range(2)]
        print(f"   🅐 原始审查者   {want}")
        for v in verdicts:
            ok = is_pass(v) == should_pass
            print(f"        {'✅' if ok else '❌'} {v[:80]}")

        v = reviewer_tooled(TASK, result, tool_log)
        ok = is_pass(v) == should_pass
        print(f"   🅑 带工具的审查者  {'✅' if ok else '❌'} {v[:90]}")

        v = code_check(TASK, result, tool_log)
        ok = is_pass(v) == should_pass
        print(f"   🅒 代码校验器（不用模型）  {'✅' if ok else '❌'} {v[:90]}")

    print("\n" + "=" * 76)
    print("看三件事：")
    print("  ① 原始审查者抓到几个错？（注意它每次都只会说一句话）")
    print("  ② 给它工具让它自己复核，有改善吗？")
    print("  ③ 代码校验器 vs 两个 LLM 审查者，谁最准？为什么？")
    print("=" * 76)


if __name__ == "__main__":
    main()

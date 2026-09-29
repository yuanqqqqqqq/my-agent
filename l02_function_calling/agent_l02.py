"""
L02 — Function Calling 深入：注册表 + 参数解析 + 错误兜底
========================================================
比 L01 的 step3 多了三样东西：

① 通用工具调度器（TOOL_REGISTRY 注册表）
    —— 加新工具只要注册两处，不用动 execute_function 的 if/elif（开闭原则）
② 参数解析兜底
    —— 模型给的 arguments 是【字符串】，可能是非法 JSON，必须 try
③ 错误当作"观察结果"喂回模型
    —— 工具炸了也不让 Agent 崩，把错误信息当 tool 结果发回去，让模型自己应对

跑法：PYTHONIOENCODING=utf-8 python l02_function_calling/agent_l02.py
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

MODEL = "glm-4-flash"

# ⚠️ 智谱实测：tool_choice 传 "none" 或指定函数【不报错但也不生效】（静默忽略）。
#    想禁用工具，唯一可靠的办法是【根本不传 tools】。所以这里只留 auto。
ZHIPU_TOOL_CHOICE = "auto"


# ════════════════════════════════════════════════════════════
# 第 1 步：工具本体
# ════════════════════════════════════════════════════════════
# ★ 原则：工具自己处理"可预期的错误"，返回人话；不要抛异常出去。
#   抛出的异常会被调度器兜住，但工具自己兜更精准。


def get_current_time() -> str:
    """获取当前时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculator(expression: str) -> str:
    """计算数学表达式。"""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "错误：表达式只允许数字和 + - * / . ( )，你给的是：" + expression
    try:
        return str(eval(expression))  # 教学用；生产环境绝不能用 eval
    except ZeroDivisionError:
        return "错误：除数不能为 0"
    except Exception as e:
        return f"计算错误：{e}"


def get_weather(city: str, unit: str = "摄氏度") -> str:
    """查天气（模拟数据，不联网）。"""
    weather_map = {
        "北京": ("晴", 25),
        "上海": ("多云", 28),
        "广州": ("雨", 30),
        "深圳": ("阴", 29),
    }
    if city not in weather_map:
        return f"没有 {city} 的天气数据。目前只支持：{list(weather_map.keys())}"
    condition, temp = weather_map[city]
    if unit == "华氏度":
        return f"{city}：{condition}，{temp * 9 / 5 + 32:.0f}°F"
    return f"{city}：{condition}，{temp}°C"


def string_length(text: str) -> str:
    """返回字符串的字符数。"""
    return f"'{text}' 有 {len(text)} 个字符"


# ════════════════════════════════════════════════════════════
# 第 2 步：注册表（函数名 → 真函数）+ 说明书（给模型看的 JSON Schema）
# ════════════════════════════════════════════════════════════
TOOL_REGISTRY = {
    "get_current_time": get_current_time,
    "calculator": calculator,
    "get_weather": get_weather,
    "string_length": string_length,
}

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间。用户问'现在几点''今天几号'时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式。需要精确的加减乘除时使用，不要心算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如'3 * (4 + 5)'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气。用户问'XX天气''XX下雨吗''气温多少'时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名，如'北京'、'上海'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["摄氏度", "华氏度"],
                        "description": "温度单位。用户没明说时用摄氏度。",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "string_length",
            "description": "数一个字符串有几个字符。用户问'XX有几个字''XX多长'时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要数长度的字符串"},
                },
                "required": ["text"],
            },
        },
    },
]


# ════════════════════════════════════════════════════════════
# 第 3 步：参数解析（★ L02 新增）
# ════════════════════════════════════════════════════════════
def parse_arguments(raw: str) -> tuple[dict, str | None]:
    """把模型给的 arguments 字符串解析成 dict。

    返回 (参数, 错误信息)：错误信息为 None 表示成功。
    ★ 解析失败不抛异常，而是把原因组织成人话返回 —— 因为这句话要喂回给模型看。
    """
    if not raw or not raw.strip():
        return {}, None  # 无参数工具会传空字符串

    try:
        args = json.loads(raw)
    except json.JSONDecodeError as e:
        return {}, f"参数解析失败：你输出的不是合法JSON（{e}）。你的原始输出是：{raw!r}"

    if not isinstance(args, dict):
        return (
            {},
            f"参数解析失败：需要 JSON 对象，你给的是{type(args).__name__}。原始输出：{raw!r}",
        )

    return args, None


# ════════════════════════════════════════════════════════════
# 第 4 步：通用工具调度器（★ L02 的核心）
# ════════════════════════════════════════════════════════════
def execute_function(name: str, arguments: dict) -> str:
    """按名字从注册表查出函数并执行，任何异常都转成"人话"返回。

    ★ 核心：绝不让工具的异常冒泡出 Agent 循环。
    返回的这句话会被当作 tool 结果喂回模型，模型据此决定重试还是认输。
    """
    # ① 工具名不存在（模型可能编一个没给过它的工具名）
    if name not in TOOL_REGISTRY:
        return f"错误：工具 '{name}'不存在。可用工具：{list(TOOL_REGISTRY.keys())}"

    func = TOOL_REGISTRY[name]

    # ② 调用时兜住所有异常
    try:
        return str(func(**arguments))
    except TypeError as e:
        # 参数对不上：缺必填 / 多了不认识的参数
        return f"参数错误：{e}。你传的参数是：{arguments}"
    except Exception as e:
        return f"工具执行失败：{type(e).__name__}: {e}"


# ════════════════════════════════════════════════════════════
# 第 5 步：Agent 循环（和 L01 同构，只是接上了新调度器）
# ════════════════════════════════════════════════════════════
def run_agent(question: str, max_steps: int = 6, show_raw: bool = False) -> str | None:
    messages = [
        {
            "role": "system",
            "content": (
                "你是严谨的助手。涉及实时时间用 get_current_time，涉及计算用calculator，"
                "涉及天气用get_weather。工具报错时如实告诉用户，不要自己编造数据。"
            ),
        },
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        print(f"\n{'─' * 56}\n🔄 第 {step} 步")

        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice=ZHIPU_TOOL_CHOICE,
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append(msg.model_dump())

            for call in msg.tool_calls:
                name = call.function.name
                raw = call.function.arguments

                if show_raw:
                    print(f"📦 原始 arguments（类型{type(raw).__name__}）：{raw!r}")

                args, err = parse_arguments(raw)
                if err:
                    result = err  # ←解析失败，把原因当结果喂回去
                    print(f"⚠️ {err}")
                else:
                    print(f"🤔 调用 {name}({args})")
                    result = execute_function(name, args)

                print(f"🔧 结果：{result}")
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
        else:
            print(f"💬 最终回答：{msg.content}")
            return msg.content

    print(f"⚠️ 走满 {max_steps} 步仍未结束，强制停止。")
    return None


# ════════════════════════════════════════════════════════════
# 主流程：三个实验
# ════════════════════════════════════════════════════════════
EXPERIMENTS = [
    (
        "实验 1：多工具并行（看调度器怎么分发）",
        "北京和上海今天天气怎么样？'你好世界'有几个字？",
    ),
    ("实验 2：错误兜底（工具内部报错）", "帮我算一下 10 / 0 等于多少？"),
    (
        "实验 3：参数对不上（触发 TypeError 兜底）",
        "算一下 1 + 1，结果保留两位小数（precision=2）",
    ),
]


def main():
    for title, q in EXPERIMENTS:
        print("\n\n" + "═" * 56)
        print(title)
        print("═" * 56)
        print(f"👤 {q}")
        run_agent(q, show_raw=True)


if __name__ == "__main__":
    main()

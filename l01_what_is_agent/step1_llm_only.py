
"""
L01 · step1 — 先看看"没有工具"的 LLM 能做什么
=============================================
目的：亲眼看到大模型的根本局限——它只会说，不会做，而且不知道自己不会。

这个脚本做了件"作弊"的事：先用 Python 算出【本机真实答案】打印出来，
再让模型答同样的问题。两下一对比，模型的错就无处可藏。

跑法：PYTHONIOENCODING=utf-8 python l01_what_is_agent/step1_llm_only.py
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

MODEL = "glm-4-flash"
WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def main():
    now = datetime.now()

    # ── 第一步：先用 Python 算出"标准答案"（绝对准）────────
    truth = {
        "现在几点了？请精确到秒。": now.strftime("%Y-%m-%d %H:%M:%S"),
        "今天是几月几号？星期几？": f"{now.strftime('%Y-%m-%d')} {WEEKDAYS[now.weekday()]}",
        "帮我精确算一下 1234567 * 7654321 等于多少？": str(1234567 * 7654321),
    }

    print("=" * 62)
    print("标准答案（Python 算的，绝对准）")
    print("=" * 62)
    for q, a in truth.items():
        print(f"  {q}")
        print(f"    → {a}")

    # ── 第二步：让模型答同样的问题（它没有工具）────────────
    print("\n" + "=" * 62)
    print("模型答的（没有工具，只能凭训练数据猜）")
    print("=" * 62)
    for q in truth:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": q}],
        )
        print(f"\n👤 用户：{q}")
        print(f"🤖 模型：{resp.choices[0].message.content}")

    print("\n" + "=" * 62)
    print("对比一下：哪几个答对了？答错的，要给它配什么才能答对？")
    print("=" * 62)


if __name__ == "__main__":
    main()
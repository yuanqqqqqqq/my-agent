"""
L08 · 实验 1：Agent 接力传话，信息在哪一步丢？
==============================================
课程说多 Agent 协作的核心难题是「Agent 间怎么通信」。

★ 但实测发现：损耗不来自「传递次数」，来自【任务目标里有没有"删减"】。

四条链，同样 6 条事实、同样 4 跳，只换【转述时被要求做什么】：
  A 纯转述「一个字都不能丢」        → 预期无损
  C 转述且「30 字以内」            → 强制压缩，预期丢
  D 「只保留你认为对方需要的」      → 让 Agent 自己判断，看它敢不敢删
  E 「润色成正式商务邮件」          → 改变文体，预期无损

跑法：PYTHONIOENCODING=utf-8 python l08_multi_agent/relay_lab.py
"""

import os

from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

MODEL = "glm-4-flash"
HOPS = 4
CHAINS_PER_MODE = 2

HANDOFF = """客户名：张伟
订单号：A-7391
金额：12,480 元
截止日期：10月15日
特殊要求：不含 logo
联系人电话：138-0000-1234"""

# 检测关键词：不用整句匹配，否则措辞一变就误报（L05 的教训）
FACTS = {
    "客户名": ["张伟"],
    "订单号": ["A-7391", "A7391"],
    "金额": ["12,480", "12480", "12 480"],
    "截止日期": ["10月15日", "10-15", "10/15", "十月十五"],
    "特殊要求": ["logo", "LOGO"],
    "电话": ["138-0000-1234", "13800001234"],
}

MODES = {
    "A 纯转述「一个字都不能丢」":
        "你是项目经理。请把下面这份任务交接单【完整地】转述给你的同事，一个字都不能丢。\n\n{content}",
    "C 转述且「必须 30 字以内」":
        "你是项目经理。请把下面这份任务交接单转述给你的同事，【必须控制在 30 字以内】。\n\n{content}",
    "D 「只保留你认为对方真正需要的」":
        "你是项目经理。请把下面这份任务交接单转述给同事，"
        "【只保留你认为对方执行时真正需要的信息】。\n\n{content}",
    "E 「润色成一封正式商务邮件」":
        "你是项目经理。请把下面这份交接单整理成一封【正式商务邮件】发给同事。\n\n{content}",
}


def survived(text: str) -> int:
    return sum(1 for keys in FACTS.values() if any(k in text for k in keys))


def run_chain(prompt_template: str) -> list[int]:
    content = HANDOFF
    chain = []
    for _ in range(HOPS):
        content = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt_template.format(content=content)}],
        ).choices[0].message.content
        chain.append(survived(content))
    return chain


def main():
    print("=" * 72)
    print(f"Agent 接力传话：{HOPS} 跳 × {CHAINS_PER_MODE} 条链   （共 {len(FACTS)} 条事实）")
    print("只换「转述时被要求做什么」—— 别的都一样")
    print("=" * 72)

    for label, tmpl in MODES.items():
        rows = [run_chain(tmpl) for _ in range(CHAINS_PER_MODE)]
        avg = [round(sum(r[i] for r in rows) / CHAINS_PER_MODE, 1) for i in range(HOPS)]
        bar = "  ".join(f"{v}" for v in avg)
        print(f"\n  【{label}】")
        print(f"     起 6.0 →  {bar}     （每跳一档）")

    print("\n" + "=" * 72)
    print("结论（自己填）：")
    print("  · 哪几条链无损？它们共同点是什么？")
    print("  · 哪条链丢了？它和别的链的差别在哪？")
    print("=" * 72)


if __name__ == "__main__":
    main()

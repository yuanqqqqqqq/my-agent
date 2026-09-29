"""
L09 · 毕业项目：带【代码校验层】的研究助手
==========================================
给一个研究主题，Agent 自主：规划要点 → 多轮检索 → 提交结构化报告 → 代码校验 → 不合格打回重做。

★ 和课程版毕业项目的区别（也是这个项目能进简历的理由）：
  课程版：Agent 生成报告，直接给用户。
  本项目：Agent 生成报告后，**先用代码校验它**，不过关就带着问题打回重做。
         校验全部是确定性的（引用真实性 / 数字有出处 / 要点覆盖 / 非空壳）。

为什么不用 LLM 当审查者？——L08 实测：原始审查者在 4 个案例里漏掉了最危险的 2 个，
它的输出永远是"通过：结果正确完整。"。代码校验器 4/4 全对。

用法：
  python research_agent/agent.py "研究主题"                  # 研究一个主题，终端看报告
  python research_agent/agent.py "研究主题" --out r.md       # 导出 Markdown
  python research_agent/agent.py "研究主题" --json           # 输出纯 JSON（可被程序消费）
  python research_agent/agent.py -i                          # 交互模式：连续追问，带上下文
  python research_agent/agent.py                             # 不带参数则跑内置演示主题

（都要加 PYTHONIOENCODING=utf-8，否则 Windows 上 emoji 会撞 GBK）
"""

import json
import os
import re
import sys

# 让 `python research_agent/agent.py` 也能 import 到包（直接跑脚本时父目录不在 sys.path）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx                              # noqa: E402
from dotenv import load_dotenv            # noqa: E402
from zhipuai import ZhipuAI               # noqa: E402

from research_agent.tools import TOOLS_SPEC, execute_tool        # noqa: E402
from research_agent.verify import verify                          # noqa: E402

MODEL = "glm-4-flash"
MAX_STEPS = 10          # 工具调用上限（L01 实测：循环没有天然的终止保证）
MAX_REWRITE = 2         # 报告校验不过时，最多打回几次

load_dotenv()

# ★ 显式绕开系统代理建客户端。实测踩过的坑：
#   httpx 默认 trust_env=True，会去读 Windows 注册表里的系统代理
#   （HKCU\...\Internet Settings\ProxyServer）。开着 Clash 时一切正常，
#   但【Clash 一停，注册表里的代理设置还在】—— 于是所有智谱请求被塞给一个
#   已经没人监听的 127.0.0.1:7897，全部 APIConnectionError，
#   而且报错只说"Connection error"，完全指不到真正原因。
#   智谱是国内 API，直连本来就好使（实测 200），也不该绕道境外代理。
#   ⚠️ 这条只救智谱。联网搜索（ddgs 用 primp 直连）走网络层，Clash 停了照样不通。
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"),
                 http_client=httpx.Client(trust_env=False, timeout=60))


# ════════════════════════════════════════════════════════════
# 结构化输出：规划和报告都用 function calling 提交
# ════════════════════════════════════════════════════════════
# ★ 为什么用 FC 而不是"让模型输出 JSON 字符串"？
#   L06 实测：在 prompt 里写"输出格式为 {...}"，模型会把这条【格式要求当成任务步骤】
#   写进计划里（元指令泄漏，3/3 次）。用 schema 就没有可抄的东西了。

PLAN_SPEC = [{"type": "function", "function": {
    "name": "submit_plan",
    "description": "提交研究计划。开始调研前先调用它。",
    "parameters": {"type": "object", "properties": {
        "points": {"type": "array", "items": {"type": "string"},
                   "description": "要研究的要点列表，3~5 条，每条一个具体问题。"
                                  "只写业务要点，不要写'整理格式''输出报告'这类动作。"}}},
    "required": ["points"]}}]

REPORT_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "submit_report",
            "description": "提交最终研究报告。调研完成后调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "一段话概述结论。如果资料不足，直接写'没有找到相关资料'。",
                    },
                    "points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                # ★ 设计决策：原本是 point + evidence 两个字段，
                                #   实测模型会把两者写在一起、让 evidence 留空，
                                #   于是被校验层以"没有 evidence"打回。
                                #   根因是 schema 违背了任务的自然形状（L04）：
                                #   写研究要点时，"结论"和"依据"在模型眼里是一条。
                                #   合并成一个字段，并明确要求它包含原文事实。
                                "point": {
                                    "type": "string",
                                    "description": (
                                        "一条研究要点。必须同时写清【结论】和【支撑它的原文事实/数字】，"
                                        "不要只写结论。例如：'默认情况下路径参数是字符串类型，"
                                        "所以 /items/3 会返回 {\"item_id\": \"3\"}（数字 3 的字符串）'"
                                    ),
                                },
                                "source": {
                                    "type": "string",
                                    "description": "来源标识，必须是工具返回里 [来源: xxx] 中 xxx 的原样复制",
                                },
                            },
                            "required": ["point", "source"],
                        },
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "本次报告用到的所有来源标识",
                    },
                },
                "required": ["summary", "points", "sources"],
            },
        },
    }
]

SYSTEM_PROMPT = """你是研究助手。工作流程：

1. 接到主题后，先调用 submit_plan 提交研究要点（3~5 条）。
2. 然后逐条调研：用 search_kb 查本地知识库、search_web 联网搜索、calculate 算数。
3. 调研充分后，调用 submit_report 提交报告。
4. 如果报告被校验打回，按打回意见补充调研后重新提交。

铁律：
- 报告里的每一个数字和事实，都必须来自工具返回的原文。不要用你自己的记忆补充。
- 每条要点都要标来源。来源必须**原样复制**工具返回里的标识，**不要改写成中文描述**。

  工具返回长这样：  [来源: fastapi_docs/request-files.md]
  你要填的 source：  fastapi_docs/request-files.md
  ❌ 不要填成：      FastAPI 官方文档        ← 这是改写，不是来源

- 资料确实不足时，如实写"没有找到相关资料"，不要编造。
- 同一条查询不要重复调用；要换就换关键词。
"""


# ════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════
MAX_STALL = 3           # 连续几次「不调工具只想说话」就放弃引导


def research(topic: str, *, use_verify: bool = True, quiet: bool = False,
             history: list | None = None):
    """跑一次研究。返回 (报告, 统计)。

    use_verify=False 时【关掉整个校验层】，报告生成后直接返回 ——
    eval.py 做 A/B 对照用的。

    history: 之前几轮的对话（多轮追问用）。格式是标准的 messages 列表，
             由 chat() 维护。传进来后模型就能看到前面的研究和报告。
    """
    def log(*a):
        if not quiet:
            print(*a)

    t0 = __import__("time").time()
    log("=" * 74)
    log(f"🔬 研究主题：{topic}")
    log("=" * 74)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += (history or [])                     # ★ 多轮：带上之前的对话
    messages.append({"role": "user", "content": topic})

    retrieved_sources: set[str] = set()       # 本次真实检索到的来源
    tools_called: set[str] = set()            # 本次调用过哪些工具（供'有没有尽力找'校验）
    tool_output_all: list[str] = []           # 本次所有工具返回的原文（供数字溯源）
    plan_points: list[str] = []
    report: dict | None = None
    rewrites = 0
    steps = 0
    stalled = 0           # 连续「不调工具」的次数，见下面的 MAX_STALL

    while steps < MAX_STEPS:
        steps += 1
        # PLAN_SPEC / REPORT_SPEC 本身已经是"工具列表"，别再包一层 []（会变成嵌套数组，API 报 400）
        specs = TOOLS_SPEC + (PLAN_SPEC if not plan_points else []) + REPORT_SPEC
        msg = client.chat.completions.create(
            model=MODEL, messages=messages, tools=specs, tool_choice="auto"
        ).choices[0].message

        if not msg.tool_calls:
            # 模型没调工具，而是想直接说话。
            #
            # ★ 实测过的坑：问"我昨天中午吃了什么"这种【根本不需要任何工具】的问题时，
            #   模型会连续拒绝走流程、只想直接回答。而原来的代码每次都回同一句
            #   "请调用 submit_report"，推 9 次都没用 → 撞步数上限、什么都没产出。
            #
            #   所以：引导有次数上限。超过就承认它是对的 —— 把自然语言收下当答案。
            stalled += 1
            if stalled >= MAX_STALL:
                log(f"\n（模型连续 {stalled} 次不走结构化流程，改为直接采用它的回答）")
                return ({"summary": msg.content or "", "points": [], "sources": [], "_freeform": True},
                        {"retrieved": len(retrieved_sources), "tools": len(tools_called),
                         "tool_names": sorted(tools_called), "rewrites": rewrites,
                         "steps": steps, "plan_points": len(plan_points),
                         "elapsed": round(__import__("time").time() - t0, 1),
                         "retrieved_set": sorted(retrieved_sources),
                         "verified": None, "problems": ["(非结构化回答)"]})
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user", "content":
                             "请调用 submit_report 提交结构化报告（或先 submit_plan 提交计划）。"
                             "如果这个问题确实不需要查任何资料、也没有可引用的来源，"
                             "就直接在 submit_report 里写清楚原因（summary）。"})
            continue
        stalled = 0

        messages.append(msg.model_dump())
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            # ── 提交计划 ──
            if name == "submit_plan":
                plan_points = args.get("points") or []
                log(f"\n📋 研究计划（{len(plan_points)} 条）：")
                for i, p in enumerate(plan_points, 1):
                    log(f"     {i}. {p}")
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": f"计划已收到，共 {len(plan_points)} 条。开始调研。"})
                continue

            # ── 提交报告 ──
            if name == "submit_report":
                report = args
                stats = {"retrieved": len(retrieved_sources), "tools": len(tools_called),
                         "tool_names": sorted(tools_called), "rewrites": rewrites,
                         "steps": steps, "plan_points": len(plan_points),
                         "elapsed": round(__import__("time").time() - t0, 1),
                         "retrieved_set": sorted(retrieved_sources)}

                if not use_verify:
                    log("\n（校验层已关闭，直接采用这份报告）")
                    stats.update(verified=None, problems=["(未校验)"])
                    return report, stats

                ok, problems = verify(report, retrieved=retrieved_sources,
                                      tool_output="\n".join(tool_output_all),
                                      required_points=plan_points,
                                      tools_called=tools_called,
                                      topic=topic)
                if ok:
                    log("\n✅ 校验通过（引用真实性 / 数字出处 / 要点覆盖 / 非空壳 / 尽力程度）")
                    stats.update(verified=True, problems=[])
                    return report, stats

                rewrites += 1
                log(f"\n❌ 校验不通过（第 {rewrites} 次）：")
                for p in problems:
                    log(f"     {p}")
                if rewrites > MAX_REWRITE:
                    log(f"\n⚠️ 打回 {MAX_REWRITE} 次仍未通过，返回最后一次报告。")
                    stats.update(verified=False, problems=problems, rewrites=rewrites)
                    return report, stats
                messages.append({"role": "tool", "tool_call_id": tc.id, "content":
                                 "报告校验不通过：\n" + "\n".join(problems) +
                                 "\n\n请补充调研后重新提交。注意：来源必须是工具返回里出现过的，"
                                 "数字必须有出处，计划里的要点都要覆盖。"})
                continue

            # ── 普通工具 ──
            tools_called.add(name)
            result = execute_tool(name, args)
            tool_output_all.append(result)
            # ★ 登记所有合法来源标识。工具返回里同时给了【短标识】和【完整链接】，
            #   模型引用哪个都算数 —— 否则它用了 URL 就会被误判成"编造来源"。
            for m in re.findall(r"\[来源:\s*([^\]\s]+)\]", result):
                retrieved_sources.add(m)
            for m in re.findall(r"链接:\s*(\S+)", result):
                retrieved_sources.add(m)
            label = {"search_web": "🌐 联网搜索", "search_kb": "📚 本地知识库",
                     "calculate": "🔢 计算"}.get(name, "🔧")
            log(f"\n  {label} {name}({str(args)[:70]})")
            log(f"     → {result[:150].replace(chr(10), ' ')}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    log(f"\n⚠️ 达到最大步数 {MAX_STEPS}，停止。")
    return report, {"retrieved": len(retrieved_sources), "tools": len(tools_called),
                    "tool_names": sorted(tools_called), "rewrites": rewrites,
                    "steps": steps, "plan_points": len(plan_points),
                    "elapsed": round(__import__("time").time() - t0, 1),
                    "retrieved_set": sorted(retrieved_sources),
                    "verified": None, "problems": ["(撞上限)"]}


def _clean(s: str) -> str:
    """显示时把模型可能带上的 [来源: ...] 外壳剥掉，避免打印成 [来源: [来源: xxx]]。"""
    return re.sub(r"^\[?来源\s*[:：]\s*|\]$", "", (s or "").strip()).strip()


def to_markdown(topic: str, report: dict | None, stats: dict) -> str:
    """把报告渲染成 Markdown —— 可以直接存文件、贴进笔记、发给别人。"""
    if not report:
        return f"# {topic}\n\n（没有产出报告）\n"
    # 模型坚持不配合流程、直接自由回答时，summary 里就是它的原话
    if report.get("_freeform"):
        return f"# {topic}\n\n{report.get('summary', '')}\n\n---\n\n> 本次为直接回答，未走结构化调研流程。\n"

    lines = [f"# {topic}", ""]
    if s := report.get("summary"):
        lines += ["## 概述", "", s, ""]
    if pts := report.get("points"):
        lines += ["## 要点", ""]
        for i, p in enumerate(pts, 1):
            # 序号单独加粗，正文不加粗 —— 否则加粗跨度会套住代码块，渲染会坏
            lines.append(f"### {i}. ")
            lines.append(p.get("point", ""))
            lines.append("")
            if p.get("source"):
                lines.append(f"> 来源：`{_clean(p['source'])}`")
                lines.append("")
    if src := report.get("sources"):
        lines += ["## 全部来源", ""]
        lines += [f"- `{_clean(s)}`" for s in src]
        lines.append("")

    v = {True: "✅ 通过", False: "❌ 未通过", None: "（未校验）"}[stats.get("verified")]
    lines += ["---", "",
              f"*校验：{v} ｜ 工具调用 {stats.get('tools', 0)} 种 ｜ "
              f"打回 {stats.get('rewrites', 0)} 次 ｜ 耗时 {stats.get('elapsed', 0)}s*"]
    return "\n".join(lines) + "\n"


def render(report: dict | None, topic: str):
    if not report:
        print("\n（没有产出报告）")
        return
    print("\n" + "=" * 74)
    print(f"📄 研究报告：{topic}")
    print("=" * 74)
    if report.get("_freeform"):
        print(f"\n{report.get('summary', '')}\n")
        print("（本次模型直接回答，未走结构化调研流程）")
        return
    print(f"\n【概述】{report.get('summary', '')}\n")
    for i, p in enumerate(report.get("points") or [], 1):
        print(f"{i}. {p.get('point', '')}")
        if p.get("source"):
            print(f"   来源：{_clean(p['source'])}")
    if src := report.get("sources"):
        print("\n【全部来源】\n" + "\n".join(f"  · {_clean(s)}" for s in src))


# ════════════════════════════════════════════════════════════
# 多轮追问
# ════════════════════════════════════════════════════════════
def chat():
    """交互模式：连续追问同一个话题，Agent 带着前面几轮的上下文继续研究。

    ★ 记忆的实现就是最朴素的那种（L05 实测：模型自己没记忆，
      你传多少 messages，它就记得多少）：把之前每轮的「主题 + 报告」留在 history 里。
    """
    print("=" * 74)
    print("研究助手 · 交互模式")
    print("  输入研究主题开始；出报告后可以继续追问（会带着上下文）。")
    print("  输入 exit / quit 退出。")
    print("=" * 74)

    history: list = []
    while True:
        try:
            q = input("\n🔬 ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit", ""):
            break

        report, stats = research(q, history=history)
        render(report, q)

        # 把这一轮记进 history —— 只留「主题 + 报告要点」，不留全部工具流水（会撑爆上下文）
        history.append({"role": "user", "content": q})
        if report and not report.get("_freeform"):
            pts = "\n".join(f"- {p.get('point', '')}" for p in (report.get("points") or []))
            srcs = "、".join(_clean(s) for s in (report.get("sources") or []))
            summary = f"（上一轮研究报告）\n概述：{report.get('summary', '')}\n要点：\n{pts}\n来源：{srcs}"
        else:
            summary = f"（上一轮回答）{report.get('summary', '') if report else '（无）'}"
        history.append({"role": "assistant", "content": summary})

        # 只保留最近 3 轮，避免上下文无限膨胀（L05 的滑动窗口）
        if len(history) > 6:
            history = history[-6:]


def main():
    argv = sys.argv[1:]
    if "-i" in argv or "--interactive" in argv:
        chat()
        return

    out = None
    if "--out" in argv:
        i = argv.index("--out")
        out = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    as_json = "--json" in argv
    if as_json:
        argv = [a for a in argv if a != "--json"]

    topic = " ".join(argv).strip() or "FastAPI 是怎么处理文件上传的？"
    # ★ --json 必须输出【纯 JSON】—— 工具日志混进去的话，管道就没法用了
    report, stats = research(topic, quiet=as_json)

    if as_json:
        print(json.dumps({"topic": topic, "report": report, "stats": stats},
                         ensure_ascii=False, indent=2))
    else:
        render(report, topic)

    if out:
        md = to_markdown(topic, report, stats)
        with open(out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\n📝 报告已写入 {out}（{len(md)} 字）")


if __name__ == "__main__":
    main()

"""
校验层 —— 纯代码，一个模型都不调
==================================
这是本项目的核心差异化。L08 实测：
  · LLM 当审查者 = 橡皮图章（4 个案例漏掉最危险的 2 个）
  · 代码校验器 4/4 全对，零 token，可复现

四种校验，全部可判定：
  ① 引用真实性 —— 报告里写的来源，必须在【本次真实检索到】的来源里
  ② 数字可溯源 —— 报告里的数字，必须在工具返回的原文里出现过
  ③ 覆盖度     —— 研究要点清单里的每一条，报告都要提到
  ④ 空壳检测   —— 不能只有概述没有要点

★ 注意一个反直觉的点：**正确拒答不是失败**（L07 的教训）。
  「知识库里没有找到」是一个合格的产出，不该被拦。
"""

import re

# 报告里出现数字时的常见形态：12,480 / 25 / 3.5 / 286K
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _is_declined(report: dict) -> bool:
    """判断是不是一个"合格的拒答"（L07：拒答不是失败）。

    ★★ 判定依据是【结构】，不是【措辞】—— 这是从一次实测事故里改出来的。

    最早的版本是拿关键词表去匹配文本：
        re.search(r"(没有找到|未找到|资料不足|无法确认|没有相关资料)", text)
    看起来够用，实际上【两个方向都会错】，8 个真实样本错判 6 个：

      误杀（合法拒答被判成"作答"→ 拖去跑全套校验 → 必挂）：
        "无法找到关于公司下周团建安排的资料。"        ← 表里只有"无法确认"
        "这个问题不需要查任何资料，也没有可引用的来源。"
        "资料库中没有这个信息" / "抱歉，我无法回答这个问题"
      漏放（真答案被判成"拒答"→ 直接放行，不校验）：
        "FastAPI 处理文件上传的方式是通过 File 和 Form，但关于上传文件时
         可能遇到的问题和解决方案没有找到相关资料。"
        ↑ 一个货真价实的答案，只因子句里带了"没有找到相关资料"就被放行

    ★ 为什么关键词表注定修不好：模型的表达是开放的，而表是封闭的。
      今天补上"无法找到"，明天就冒出来"未能检索到""暂时没有这方面的资料"。
      而【措辞】和【结论】本来就不是一回事 —— 漏放的根因正是这个。

    ★ 正确依据是 schema 已经给好的结构：报告必须把结论写进 points。
      points 为空 = 模型自己声明"我没有拿到任何结论" = 拒答。
      这与它用哪句话来表达无关，也不受子句里出现什么词影响。
    """
    return not (report.get("points") or [])


def _norm_source(s: str) -> str:
    """归一化来源标识，再比较。

    ★★ 为什么必须归一化 —— 实测踩过的坑：
       我们告诉模型「来源填 [来源: xxx] 里的 xxx」，但模型常常把【整个
       [来源: xxx] 连方括号一起】抄进来。这是【格式不符】，不是【编造来源】。
       直接字符串相等会把它误判成"引用了不存在的来源"，连环拦截 3 次。

       ⚠️ 这是本项目里第 2 次同类失败（L07 的引用校验也误杀过"正确拒答"）。
          校验层最常见的两种错：**漏放** 和 **误杀**。误杀更隐蔽 ——
          日志上看是"校验生效了"，实际是在拒绝正确答案。
    """
    s = (s or "").strip()
    m = re.match(r"^\[?来源\s*[:：]\s*(.+?)\]?$", s)
    if m:
        s = m.group(1)
    return s.strip().strip("[]").strip().rstrip("/")


# ════════════════════════════════════════════════════════════
# ① 引用真实性
# ════════════════════════════════════════════════════════════
def check_citations(report: dict, retrieved: set[str]) -> list[str]:
    """报告里声称的来源，必须真的检索到过。

    ★ 比较前先归一化两边（见 _norm_source），否则会把"格式不符"误判成"编造来源"。
    """
    problems = []
    claimed_raw = list(report.get("sources") or [])
    for p in report.get("points") or []:
        if s := p.get("source"):
            claimed_raw.append(s)
    claimed = {_norm_source(s) for s in claimed_raw if s}

    if not claimed:
        problems.append("① 报告里一条来源都没标，无法核实")

    real = {_norm_source(s) for s in retrieved}
    ghosts = sorted(c for c in claimed if c not in real)
    if ghosts:
        problems.append(f"① 引用了不存在的来源：{ghosts}（本次并没有检索到它们）")
    return problems


# ════════════════════════════════════════════════════════════
# ② 数字可溯源
# ════════════════════════════════════════════════════════════
def check_numbers(report: dict, tool_output: str) -> list[str]:
    """报告里的数字，必须在工具返回的原文里出现过。

    ★ 这条治的是 L06 实测的病：模型编了个 22.0 去算温差。
      任何数字都必须能在工具输出里找到出处。
    """
    haystack = tool_output.replace(",", "")
    problems, bad = [], []
    for p in report.get("points") or []:
        blob = f"{p.get('point', '')} {p.get('evidence', '')}"   # evidence 已合并进 point，留着兼容
        for n in _NUM.findall(blob):
            if len(n) < 2:
                continue                      # 单个数字（如"3 个要点"）不深究
            if n.replace(",", "") not in haystack:
                bad.append(n)
    if bad:
        problems.append(f"② 这些数字在工具返回里找不到出处：{sorted(set(bad))}")
    return problems


# ════════════════════════════════════════════════════════════
# ③ 覆盖度
# ════════════════════════════════════════════════════════════
def _cjk_windows(text: str) -> set[str]:
    """中文按 2 字滑窗切词（没有分词器，滑窗能覆盖到「检索增强」这类词）。

    ★ 为什么不能直接 re.findall(r"[一-鿿]{2,}", ...)：
      那会把整段连续中文当成【一个词】——「上传文件的路径参数设置」变成一条 9 字长串，
      要求报告里出现一模一样的整句才算覆盖，几乎永远不覆盖 → 无限打回。
    """
    words: set[str] = set()
    for seg in re.findall(r"[一-鿿]+", text):
        words |= {seg[i:i + 2] for i in range(len(seg) - 1)}
    return words


def check_coverage(report: dict, required_points: list[str]) -> list[str]:
    """研究计划里的要点，报告至少要覆盖【大多数】。

    ★ 这条治的是 L08 Q2 的病：多 Agent 流水线把关键约束传丢了。
      终点检查（确定性）比每跳都传可靠。

    ★ 阈值：覆盖不足一半才拦。要求 100% 覆盖会在真实任务里无限打回——
      模型把计划拆成 4 条、报告写了 3 条，剩下 1 条只是措辞没对上。
      完整性远不如「编造来源 / 编数字」危险，后者有 ①② 两道硬校验守着。
    """
    if not required_points:
        return []
    blob = (report.get("summary", "") +
            " " + " ".join(str(p) for p in (report.get("points") or [])))
    missing = []
    for pt in required_points:
        # 用要点里的关键词判断有没有提到（避免整句匹配的坑，见 L05）
        kws = _cjk_windows(pt) | set(re.findall(r"[A-Za-z][A-Za-z0-9_+#.-]{2,}", pt))
        if kws and not any(k in blob for k in kws):
            missing.append(pt)
    if len(missing) * 2 > len(required_points):
        return [f"③ 研究计划里的要点没覆盖：{missing}"]
    return []


# ════════════════════════════════════════════════════════════
# ⑥ 限定词检查 —— 私有问题不能用公开资料搪塞
# ════════════════════════════════════════════════════════════
# 实测（第 2 步加 fetch_page 之后回归出来的）：
#   问"我们公司下周的团建安排是什么"，报告引用真实率 100%，
#   但引用的全是"团建活动怎么策划"这类公开文章 —— 和"我们公司"毫无关系。
#
#   ★ 校验层原本能验证"来源是真的"，验证不了"来源是相关的"。
#     fetch_page 扩大了模型能找到"看似相关材料"的面，把这个盲区放大了。
PRIVATE_HINTS = ("我们", "咱们", "本公司", "贵公司", "本团队", "本部门", "我的", "本人")


def check_scope(report: dict, topic: str, retrieved: set[str]) -> list[str]:
    """问题问的是私有信息时，报告不能拿公开资料冒充答案。"""
    if not any(h in (topic or "") for h in PRIVATE_HINTS):
        return []
    blob = (report.get("summary", "") + " "
            + " ".join(str(p) for p in (report.get("points") or [])))
    if _is_declined(report):
        return []                       # 已拒答，合格

    # 有作答 —— 判断依据是不是"私有来源"（本地知识库的标识形如 fastapi_docs/xxx）
    claimed = {_norm_source(s) for s in (report.get("sources") or [])}
    for p in report.get("points") or []:
        if s := p.get("source"):
            claimed.add(_norm_source(s))
    has_private = any(c and "/" in c and not c.startswith("http") for c in claimed)

    # 也没显式声明"没有针对性的信息" → 拦
    disclaimed = re.search(r"(没有找到|未找到|未提供|没有.*具体|仅供参考|通用做法)", blob)
    if not has_private and not disclaimed:
        return ["⑥ 问题问的是「我们/我」的私有信息，但报告用的是公开资料，"
                "且没有说明这一点。请如实说明「没有找到针对贵方的具体信息」，"
                "或引用本地知识库。"]

    # 没查本地知识库就答私有问题，也算没尽力
    if not has_private and not any(not r.startswith("http") and "/" in _norm_source(r)
                                   for r in retrieved):
        return ["⑥ 问题问的是私有信息，但你一次本地知识库都没查。请先查 search_kb。"]
    return []


# ════════════════════════════════════════════════════════════
# ④ 空壳检测
# ════════════════════════════════════════════════════════════
def check_substance(report: dict, min_len: int = 25) -> list[str]:
    """要点必须"有实质内容"，不能是一句空话。

    ★ evidence 字段已合并进 point，所以这里改成看【要点够不够长、够不够具体】。
    """
    points = report.get("points") or []
    if not points:
        return ["④ 报告没有任何要点，只有概述"]
    thin = [i for i, p in enumerate(points, 1)
            if len((p.get("point") or "").strip()) < min_len]
    if thin and len(thin) == len(points):
        return [f"④ 所有要点都太短（<{min_len} 字），看不出结论背后的依据"]
    return []


# ════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════
def verify(report: dict, *, retrieved: set[str], tool_output: str,
           required_points: list[str],
           tools_called: set[str] | None = None,
           topic: str = "") -> tuple[bool, list[str]]:
    # ★★ 第一件事：判断它是不是一个【合格的拒答】。
    #    拒答没有要点、没有引用、覆盖不全 —— 全是正常的，不该被拦。
    #    ⚠️ 这个坑我在 L07 修过一次（当时是"引用校验"误杀拒答），
    #       写这一层时又犯了一遍。三态判定（有据 / 拒答 / 无据作答）必须写在最前面。
    blob = (report.get("summary", "") + " "
            + " ".join(str(p) for p in (report.get("points") or [])))
    if _is_declined(report):
        # ★★★ 但"拒答"也要分两种：
        #     ① 真的找过了，资料确实不存在 → 合格，放行
        #     ② 只查了一个来源就放弃      → 不合格，打回
        #     实测：问"RAG 的核心思路"（本地库没有），它只查了本地库就写
        #     "没有找到相关资料"。**没编造是对的，但没尽力找。**
        #     校验层当初放行了它 —— 因为「有没有依据」和「有没有尽力」是两回事。
        if tools_called is not None and "search_web" not in tools_called:
            return False, [
                "⑤ 你只查了本地知识库就判断'没有资料'。"
                "请先调用 search_web 联网搜一次，再决定是否真的没有。"
            ]
        return True, []

    problems = (check_citations(report, retrieved)
                + check_numbers(report, tool_output)
                + check_coverage(report, required_points)
                + check_substance(report)
                + check_scope(report, topic, retrieved))
    return (not problems), problems


if __name__ == "__main__":
    # 自检：4 个案例，看能不能都判对
    LOG = ("[来源: https://fastapi.tiangolo.com]\nFastAPI 教程\n"
           "FastAPI is a modern web framework. 安装：uv add fastapi\n\n"
           "[来源: kb://fastapi_docs/request-files.md]\n上传文件用 UploadFile")
    REAL_SOURCES = {"https://fastapi.tiangolo.com", "kb://fastapi_docs/request-files.md"}

    GOOD_POINT = ("FastAPI 是一个现代 Web 框架，用 Python 类型声明来定义参数，"
                  "原文：FastAPI is a modern web framework")
    CASES = [
        ("① 引用不存在的来源",
         {"summary": "FastAPI 介绍",
          "points": [{"point": GOOD_POINT, "source": "https://fake.com"}]},
         "FastAPI", False),
        ("② 数字查无出处",
         {"summary": "性能对比",
          "points": [{"point": "据称性能高出 999% 以上，原文：very fast", "source": "https://fastapi.tiangolo.com"}]},
         "FastAPI", False),
        ("③ 要点没覆盖",
         {"summary": "只讲了框架",
          "points": [{"point": GOOD_POINT, "source": "https://fastapi.tiangolo.com"}]},
         "上传文件", False),
        ("④ 要点太短（新规则）",
         {"summary": "FastAPI 简介", "sources": ["https://fastapi.tiangolo.com"],
          "points": [{"point": "它是框架", "source": "https://fastapi.tiangolo.com"}]},
         "FastAPI", False),
        ("⑤ 合格的拒答（不该被拦）",
         {"summary": "没有找到相关资料", "points": [], "sources": []},
         "上传文件", True),
        # ↓ ⑦⑧ 是同一个 bug 的两半，缺一个就测不出另一半：
        #   旧的关键词表实现在 ⑦ 上"误杀"、在 ⑧ 上"漏放"。
        ("⑦ 换了说法的拒答 —— 旧实现会误杀（不该被拦）",
         {"summary": "无法找到关于公司下周团建安排的资料。",
          "points": [], "sources": []},
         "团建", True),
        ("⑧ 真答案的子句里带了「没有找到」—— 旧实现会当拒答放行（该拦）",
         {"summary": "FastAPI 简介", "sources": ["https://fastapi.tiangolo.com"],
          "points": [{"point": GOOD_POINT + "，不过官方文档里没有找到更细的说明",
                      "source": "https://fake.com"}]},   # ← 假来源，必须被抓出来
         "FastAPI", False),
        ("对照组：完整报告",
         {"summary": "FastAPI 简介", "sources": ["https://fastapi.tiangolo.com"],
          "points": [{"point": GOOD_POINT, "source": "https://fastapi.tiangolo.com"}]},
         "FastAPI", True),
    ]
    print("=" * 70)
    print("校验层自检")
    print("=" * 70)
    for label, report, required, should_pass in CASES:
        ok, problems = verify(report, retrieved=REAL_SOURCES,
                              tool_output=LOG, required_points=[required])
        got_pass = ok
        mark = "✅" if got_pass == should_pass else "❌"
        want = "该放行" if should_pass else "该拦截"
        print(f"\n{mark} {label}   （{want}）")
        print(f"    {'通过' if ok else '拦截'}：{problems if problems else '无问题'}")

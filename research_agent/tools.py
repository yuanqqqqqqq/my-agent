"""
工具层 —— 每个工具都自带【有效性校验】
========================================
实测教训（L02/L03/L09）：「工具返回了东西」不等于「返回的东西有用」。

  · 旧包 duckduckgo_search：问 FastAPI 教程 → 返回星座运势，而且【不报错】
  · tool_choice="none" 被静默忽略（不报错也不生效）
  · calculator 白名单漏 % → 返回"非法字符"，逼模型绕路

所以本层的每个工具都必须自己判断"这次调用到底有没有用"，把结论写进返回值，
让模型（和后面的校验层）能看见。
"""

import re
from urllib.parse import urlparse

from ddgs import DDGS

# 复用 RAG 项目（已 pip install -e 装好）。
# ★ 懒加载：只在真的调用 search_kb 时才 import rag。
#   好处：没装 my-rag 的人也能 import 本包、跑联网研究、跑校验层；
#        只有点到 search_kb 时才收到一句「请先安装 my-rag」，而不是 ImportError 崩掉。
_rag = None  # 缓存 (client, retrieve, get_collection)


def _load_rag():
    """懒加载 my-rag（PyPI 上没有，需单独 pip install -e ../my-rag）。"""
    global _rag
    if _rag is None:
        from rag.client import get_client
        from rag.retrieve import retrieve
        from rag.store import get_collection
        _rag = (get_client(), retrieve, get_collection)
    return _rag


# ════════════════════════════════════════════════════════════
# 相关性判断（纯代码，不调模型）
# ════════════════════════════════════════════════════════════
def keywords(text: str) -> set[str]:
    """从文本里抠出关键词：英文单词 + 中文 2 字以上的片段。

    ⚠️ 中文按【2 字滑窗】切，不是按词切 —— 我们没有分词器，
       用滑窗能保证「检索增强」这样的词被覆盖到。
    """
    text = text.lower()
    words = set(re.findall(r"[a-z][a-z0-9_+#.-]{1,}", text))
    cjk = re.findall(r"[一-鿿]+", text)
    for seg in cjk:
        words |= {seg[i:i + 2] for i in range(max(1, len(seg) - 1))}
    return words


def relevance(query: str, text: str) -> float:
    """查询关键词在结果里的命中比例。0~1。

    ★ 这是本项目的核心防守：搜索引擎经常返回"成功但无关"的结果，
      用这个打分把它们挡在上下文外面。
    """
    kws = keywords(query)
    if not kws:
        return 1.0
    hit = sum(1 for k in kws if k in text.lower())
    return hit / len(kws)


# ════════════════════════════════════════════════════════════
# 工具 1：联网搜索（带相关性过滤）
# ════════════════════════════════════════════════════════════
def search_web(query: str, max_results: int = 5) -> str:
    """联网搜索。过滤掉不相关的结果，并在返回里说明过滤情况。"""
    try:
        with DDGS(timeout=12) as d:
            raw = list(d.text(query, max_results=max_results))
    except Exception as e:
        return f"搜索失败（{type(e).__name__}）：{e}。可以换个关键词重试。"

    if not raw:
        return f"搜索「{query}」返回 0 条结果。建议换关键词重试。"

    kept, dropped = [], 0
    for r in raw:
        title = r.get("title") or ""
        body = r.get("body") or ""
        href = r.get("href") or ""
        score = relevance(query, title + " " + body)
        if score < 0.34:          # ★ 阈值：命中不到 1/3 关键词的算无关
            dropped += 1
            continue
        # ★ 来源标识要【短、好抄】——实测：给长 URL，模型会把它"自然化"成
        #   "FastAPI 官方文档"这种中文描述，然后被校验层拦下。工具的返回格式
        #   决定了模型能不能用对，这是工具设计的一部分（L04）。
        # ★ 但 fetch_page 需要完整链接，所以【两个都给】：短标识用于引用，
        #   完整链接用于抓正文。校验层会把两者都登记为合法来源。
        site = urlparse(href).netloc or href
        kept.append(f"[来源: {site}]\n链接: {href}\n{title}\n{body[:400]}")

    if not kept:
        return (f"搜索「{query}」原始返回 {len(raw)} 条，"
                f"但全部被判为【不相关】（相关性 < 0.34），已过滤。"
                f"请换更具体的关键词重试。")

    head = f"搜到 {len(kept)} 条相关结果" + (f"（另有 {dropped} 条不相关已过滤）" if dropped else "")
    return head + "\n\n---\n\n".join([""] + kept)


# ════════════════════════════════════════════════════════════
# 工具 2：查本地知识库（复用 my-rag）
# ════════════════════════════════════════════════════════════
_KB_CACHE: dict[str, object] = {}


def search_kb(query: str, top_k: int = 3, index: str = "fastapi_docs") -> str:
    """在本地知识库检索。返回带来源的片段。

    ★ 设计要点（L07）：直接用 retrieve()，不要用 ask()。
      ask() = 检索 + 生成 —— 把会生成的东西喂给会生成的模型，会污染信息。
    """
    try:
        client, retrieve, get_collection = _load_rag()
    except Exception as e:
        return (f"search_kb 依赖 my-rag（PyPI 上没有），请先安装：pip install -e ../my-rag。"
                f"原始错误：{e}")

    try:
        if index not in _KB_CACHE:
            _KB_CACHE[index] = get_collection(index)
        col = _KB_CACHE[index]
    except Exception as e:
        return f"打开索引 {index} 失败：{e}"

    hits = retrieve(client, col, query, strategy="rerank", top_k=top_k)
    if not hits:
        return f"知识库 {index} 里没有检索到「{query}」相关内容。"

    blocks = [f"[来源: {index}/{h.get('source', '?')}]\n{h.get('text', '')[:500]}"
              for h in hits]
    return f"知识库命中 {len(blocks)} 条\n\n---\n\n".join([""] + blocks)


# ════════════════════════════════════════════════════════════
# 工具 3：抓取网页正文（治"只看到 400 字摘要，读不到全文"）
# ════════════════════════════════════════════════════════════
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")

# 抓正文时要去掉的整块标签——留着它们，正文会被导航/广告淹没
_STRIP_TAGS = ["script", "style", "noscript", "nav", "header", "footer",
               "aside", "form", "iframe", "svg", "button"]


def fetch_page(url: str, max_chars: int = 3000) -> str:
    """抓取网页正文，返回纯文本。抓不到就返回带原因的人话，让模型换一条路。

    ★ 为什么需要它：search_web 只返回 400 字摘要。摘要不够时模型只能反复搜，
      实测会把步数预算烧光（eval 里第 3 题就是这么撞上限的）。
    """
    import requests
    from bs4 import BeautifulSoup

    if not re.match(r"^https?://", url or ""):
        return f"URL 格式不对：{url!r}。必须是 http:// 或 https:// 开头。"

    try:
        resp = requests.get(url, timeout=12, headers={"User-Agent": _UA})
    except Exception as e:
        return f"抓取失败（{type(e).__name__}）：{e}。可以换一个链接试试。"

    if resp.status_code != 200:
        return f"抓取失败：HTTP {resp.status_code}。可以换一个链接试试。"

    ctype = resp.headers.get("Content-Type", "")
    if "html" not in ctype.lower():
        return f"这个链接不是网页（Content-Type: {ctype or '未知'}），抓不了正文。"

    # ★ 中文站点常把编码声明写错，resp.text 会变乱码 —— 交给 bs4 从 meta 里判
    resp.encoding = resp.apparent_encoding or resp.encoding
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))

    if len(text) < 200:
        return (f"这个页面抓到的正文只有 {len(text)} 字，可能是纯 JS 渲染的页面。"
                f"换一个链接试试。")

    site = urlparse(url).netloc
    truncated = "（已截断）" if len(text) > max_chars else ""
    return (f"[来源: {site}]\n{url}\n{text[:max_chars]}{truncated}")


# ════════════════════════════════════════════════════════════
# 工具 4：计算
# ════════════════════════════════════════════════════════════
def calculate(expression: str) -> str:
    """计算数学表达式。支持 + - * / % ** 和括号。

    ★ 白名单【必须包含 %】—— L03 实测：漏了它，模型会死循环 8 轮，
      最后把「商的整数部分」当成余数。
    """
    allowed = set("0123456789+-*/%.() ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含不支持的字符。支持 + - * / % ** ( ) 和数字"
    try:
        return str(eval(expression))
    except ZeroDivisionError:
        return "错误：除数不能为 0"
    except Exception as e:
        return f"计算错误：{e}"


# ════════════════════════════════════════════════════════════
# 注册表 + 说明书
# ════════════════════════════════════════════════════════════
TOOL_REGISTRY = {
    "search_web": search_web,
    "fetch_page": fetch_page,
    "search_kb": search_kb,
    "calculate": calculate,
}

# ★ description 按 L04 的写法：做什么 + 什么时候用 + 【什么时候不用】+ 参数怎么填
TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "search_web",
        "description": (
            "联网搜索公开信息。当需要最新资讯、公开资料、或本地知识库没有的内容时使用。"
            "注意：工具会自动过滤不相关结果，如果返回'全部被判为不相关'，"
            "说明关键词太宽泛，请换更具体的词重试。"
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string",
                      "description": "搜索关键词。请提取核心术语，不要照抄用户的整句话。"},
            "max_results": {"type": "integer", "description": "最多取几条，默认 5"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_page",
        "description": (
            "抓取某个网页的正文全文。当 search_web 返回的摘要信息不够、"
            "需要读到完整内容时使用。参数用 search_web 返回里的完整链接。"
            "注意：只对 http(s) 网页有效，抓不到会返回原因。"
        ),
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "要抓取的网页完整链接，如 https://..."},
            "max_chars": {"type": "integer", "description": "最多返回多少字符，默认 3000"}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "search_kb",
        "description": (
            "在本地 FastAPI 官方文档库里检索。当问题涉及 FastAPI 的用法、接口、配置时使用，"
            "这比联网搜索更可靠。注意：非 FastAPI 的问题不要用这个工具。"
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "检索关键词，提取核心术语"},
            "top_k": {"type": "integer", "description": "返回几条，默认 3"},
            "index": {"type": "string", "description": "索引名，默认 fastapi_docs"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "计算数学表达式。支持 + - * / % 和括号。需要精确计算时使用。",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string",
                           "description": "数学表达式，如 '12*34'、'17%5'、'(1+2)*3'"}},
            "required": ["expression"]}}},
]


def execute_tool(name: str, args: dict) -> str:
    """调度器：任何异常都转成人话返回，绝不让工具异常冒泡出 Agent 循环（L02）。"""
    if name not in TOOL_REGISTRY:
        return f"错误：工具 '{name}' 不存在。可用工具：{list(TOOL_REGISTRY)}"
    try:
        return str(TOOL_REGISTRY[name](**args))
    except TypeError as e:
        return f"参数错误：{e}。你传的参数是：{args}"
    except Exception as e:
        return f"工具执行失败：{type(e).__name__}: {e}"


if __name__ == "__main__":
    # 工具自检（L03 的教训：白名单/阈值这类硬编码，靠测试防，不靠仔细防）
    cases = [("1+2", "3"), ("17%5", "2"), ("2**10", "1024"),
             ("10/0", "错误：除数不能为 0"), ("abc", "错误：表达式包含不支持的字符")]
    ok = True
    for expr, want in cases:
        got = calculate(expr)
        if not got.startswith(want[:6]):
            print(f"❌ calculate({expr!r}) = {got!r}，期望 {want!r}")
            ok = False
    print("✅ 工具自检通过" if ok else "❌ 自检失败")
    print(f"\n相关性自检：")
    print(f"  查询'FastAPI 教程' vs 结果'FastAPI 教程 | 菜鸟教程' → {relevance('FastAPI 教程', 'FastAPI 教程 | 菜鸟教程'):.2f}")
    print(f"  查询'FastAPI 教程' vs 结果'Free Daily Horoscopes'  → {relevance('FastAPI 教程', 'Free Daily Horoscopes'):.2f}")

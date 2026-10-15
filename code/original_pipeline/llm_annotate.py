#!/usr/bin/env python3
"""
llm_annotate.py —— 用 OpenAI API 按编码本 v2（方案C）标注评论

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 API key 放在哪：同目录下的  .env  文件（照着 .env.example 复制一份改），格式：
     OPENAI_API_KEY=sk-proj-你的key
 本脚本只从 .env 或环境变量读 key，不会打印、不会写进数据库或日志。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

编码本 v2 的来历：v1 沿用谣言研究 SDQC 思路设计了"信任维度"，500 条试标注中带信任立场的
相关评论不足 2%，无法支撑统计分析；经开放式 + 轴心编码归纳后改为下面的两层结构，
原信任维度保留为第二层的一个类别。v1 提示词存档于 8.分析与图表/prompts/。

每条评论输出：
  on_topic  是否实质性地讨论子弹库帛书及其流失/归还/追索（相关帖子下的闲聊、表情、跑题内容为 false）
  tone      情感基调：喜悦自豪 / 愤慨痛惜 / 中性理性 / 质疑异见
  function  话语功能：情感表态 / 追索诉求与国家能力 / 文物知识与归还细节 / 历史与文明论战 / 处置过程的信任与怀疑
（on_topic 为 false 时 tone、function 都是"不适用"）

合规：只描述话语现象，不判断怀疑内容真假；只发送评论正文和帖子主题，不发送昵称和用户 ID。

结果写入 db/annotations.sqlite 的 labels 表，主键 = (平台, 评论, 编码本, 模型, 轮次)，
同一条评论可以被不同模型 / 不同轮次重复标注（用于信度检验），互不覆盖。

用法：
    python3 llm_annotate.py --scope pilot500 --run-tag pilot_v2                 # 用 v1 试标注的同一批评论
    python3 llm_annotate.py --scope pilot500 --run-tag pilot_v2 --model gpt-5.6-terra  # 换模型做信度对比
    python3 llm_annotate.py --scope core --run-tag full_v2                       # 事件后相关评论全量
    python3 llm_annotate.py --scope pre --run-tag full_v2                        # 事件前相关评论（研究问题2的基线）
    python3 llm_annotate.py --scope core --limit 20 --dry-run                    # 只看提示词
"""
import argparse
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "heritage.sqlite")
ANN_DB_PATH = os.path.join(BASE_DIR, "db", "annotations.sqlite")
CODEBOOK = "v2"

NA = "不适用"
TONES = ["喜悦自豪", "愤慨痛惜", "中性理性", "质疑异见"]
FUNCTIONS = ["情感表态", "追索诉求与国家能力", "文物知识与归还细节", "历史与文明论战", "处置过程的信任与怀疑"]

# 每百万 token 价格（美元）：输入 / 缓存输入 / 输出，来源 platform.openai.com/docs/pricing（2026-09）
PRICES = {
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-5.6-sol": (4.00, 0.40, 20.00),
}

# (评论, on_topic, tone, function)。标 * 的来自 v1 试标注里真实判错过的情况
FEW_SHOT = [
    ("太好了！终于回家了，这就是我们中华文明生生不息的证明！", True, "喜悦自豪", "情感表态"),
    ("强盗！偷了几十年才还，还要我们感谢？", True, "愤慨痛惜", "情感表态"),
    ("还有第一卷《四时令》在美国，什么时候能要回来？", True, "中性理性", "追索诉求与国家能力"),
    ("国家强大了文物才能回家，以前谁理你", True, "喜悦自豪", "追索诉求与国家能力"),
    ("返还的《五行令》《攻守占》是残帛，没回来的《四时令》才是完整的", True, "中性理性", "文物知识与归还细节"),
    ("楚帛书里的伏羲女娲创世记载，比中原文献早得多", True, "中性理性", "历史与文明论战"),
    ("怎么证明运回来的就是真品？谁去验证的？", True, "质疑异见", "处置过程的信任与怀疑"),
    ("希望公开完整的鉴定报告和运输记录", True, "中性理性", "处置过程的信任与怀疑"),
    ("一块破布而已，有必要这么吹吗", True, "质疑异见", "文物知识与归还细节"),
    ("感谢你的分享[赞][赞]", False, NA, NA),                           # * 谢的是博主
    ("三叔，快出来写续集", False, NA, NA),                              # * 聊《盗墓笔记》
    ("大英博物馆那批汝窑来上博展览，你敢扣吗", False, NA, NA),           # * 别的文物、别国机构
    ("这个视频的背景音乐叫什么？", False, NA, NA),
]

SYSTEM_PROMPT = f"""你是社会科学研究助理，正在给关于"长沙子弹库战国帛书"的中文社交媒体评论做内容编码。

事件背景：子弹库帛书1942年出土于长沙子弹库楚墓，1946年被美国人柯强骗走带出国。2024年9月央视纪录片《帛书传奇》播出。2025年5月，美国史密森尼学会国立亚洲艺术博物馆将第二、三卷《五行令》《攻守占》返还中国；同年10月入藏湖南省博物馆（"归湘"）。第一卷《四时令》仍在美国。评论可能发表于归还前或归还后。

每条评论编三个字段：

【on_topic】是否实质性地讨论子弹库帛书、它的流失与归还、或由此引出的文物追索话题。宁严勿宽，以下都判 false：
- 没有实质内容：纯表情、纯点赞、"沙发""来了""多着呢"之类短语、纯话题标签
- 对着博主说话：感谢分享、催更、夸账号
- 聊影视小说和流行文化：《盗墓笔记》、三叔、吴邪、稻米等，即使帛书是小说原型
- 说的是别的文物或别国博物馆（大英博物馆、圆明园兽首、马王堆、敦煌等），没有落回子弹库帛书
- 只提长沙、湖南、博物馆参观攻略等周边信息
归还前讨论帛书流失、呼吁追索，同样判 true。

【tone 情感基调】四选一（on_topic 为 false 填"{NA}"）：
- 喜悦自豪：欢迎回家、激动感动、民族自豪、对归还或相关各方表示感谢
- 愤慨痛惜：谴责盗掠者和外国、痛惜流失与损毁、屈辱记忆、"抢回来"之类的愤怒
- 中性理性：陈述事实、提问求解、知识性讨论，没有明显情绪
- 质疑异见：怀疑、讽刺主流叙事、淡化事件意义、唱反调
反讽按实际态度归类，不按字面。

【function 话语功能】五选一（on_topic 为 false 填"{NA}"），看评论主要在"说什么"：
- 情感表态：主要就是表达情绪或态度，没有更多实质内容
- 追索诉求与国家能力：要求追回第一卷或其他流失文物、谈还有多少文物在海外、把归还归因于国力、外交、法律
- 文物知识与归还细节：帛书本身的内容、价值、研究，归还的卷次、残损、运输、展出、保存，流失经过的具体情节
- 历史与文明论战：延伸到更宽的古史、楚文化起源、夏商周、伪史论、中西文明比较、国际政治等争论
- 处置过程的信任与怀疑：针对中方机构接收、鉴定、保管、展出这批帛书的信任、怀疑、要求公开透明，或为中方机构辩护。对外国机构的谴责不属于这一类
一条评论涉及多个功能时，按优先级取一个：处置过程的信任与怀疑 > 追索诉求与国家能力 > 历史与文明论战 > 文物知识与归还细节 > 情感表态。

原则：只描述话语本身，不判断怀疑内容是否属实，不表达你自己的立场。帖子主题只用于理解上下文，编码对象是评论本身。

示例：
{chr(10).join(f'评论："{t}" → on_topic={str(o).lower()}, tone={tn}, function={fn}' for t, o, tn, fn in FEW_SHOT)}

每批评论都带编号和 comment_id，请为每一条输出一条编码，comment_id 原样照抄，不要遗漏。"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"labels": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "comment_id": {"type": "string"},
            "on_topic": {"type": "boolean"},
            "tone": {"type": "string", "enum": TONES + [NA]},
            "function": {"type": "string", "enum": FUNCTIONS + [NA]},
        },
        "required": ["comment_id", "on_topic", "tone", "function"],
        "additionalProperties": False,
    }}},
    "required": ["labels"],
    "additionalProperties": False,
}


def load_dotenv():
    """读取同目录 .env 里的 OPENAI_API_KEY 等配置（只放进环境变量，不打印）"""
    for path in (os.path.join(BASE_DIR, "pipeline", ".env"), os.path.join(BASE_DIR, ".env")):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def open_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("ATTACH DATABASE ? AS ann", (ANN_DB_PATH,))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ann.labels (
            platform TEXT, comment_id TEXT, codebook TEXT, model TEXT, run_tag TEXT,
            on_topic INTEGER, tone TEXT, function TEXT, annotated_at TEXT,
            PRIMARY KEY (platform, comment_id, codebook, model, run_tag)
        )
    """)
    conn.commit()
    return conn


SCOPES = {
    "core": "c.post_relevant = 1 AND c.is_post_event = 1",
    "pre": "c.post_relevant = 1 AND c.is_post_event = 0",
}


def fetch_targets(conn, scope, model, run_tag, limit=None):
    """取待标注评论，跳过同一 (编码本, 模型, 轮次) 下已标过的"""
    if scope in SCOPES:
        where, params = SCOPES[scope], []
        join_extra = ""
    elif scope.startswith("sample:"):  # 预先抽好、存在 ann.samples 表里的样本（见 8.分析与图表/scripts/make_sample.py）
        where, params = "s.name = ?", [scope.split(":", 1)[1]]
        join_extra = "JOIN ann.samples s ON s.platform = c.platform AND s.comment_id = c.comment_id"
    else:  # 复用某一轮 v1 试标注的同一批评论，便于前后对比
        where, params = "v1.run_tag = ?", [scope]
        join_extra = "JOIN ann.annotations v1 ON v1.platform = c.platform AND v1.comment_id = c.comment_id"
    rows = conn.execute(f"""
        SELECT c.platform, c.comment_id, c.text, p.text AS post_text
        FROM comments c {join_extra}
        LEFT JOIN posts p ON p.platform = c.platform AND p.post_id = c.post_id
        LEFT JOIN ann.labels l ON l.platform = c.platform AND l.comment_id = c.comment_id
             AND l.codebook = ? AND l.model = ? AND l.run_tag = ?
        WHERE {where} AND l.comment_id IS NULL AND c.text IS NOT NULL AND TRIM(c.text) != ''
        ORDER BY c.platform, c.comment_id
    """, [CODEBOOK, model, run_tag] + params).fetchall()
    return rows[:limit] if limit else rows


def build_user_prompt(batch):
    return "\n".join(
        f'[{i}] comment_id={cid}\n    帖子主题：{(post or "")[:60].replace(chr(10), " ")}'
        f'\n    评论：{(text or "").replace(chr(10), " ")[:500]}'
        for i, (_, cid, text, post) in enumerate(batch, 1))


class QuotaExhausted(Exception):
    """余额耗尽：OpenAI 不开放余额查询，只能靠 insufficient_quota 报错识别"""


def annotate_batch(client, model, batch, stats, lock):
    import openai
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": build_user_prompt(batch)}],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "labels", "schema": OUTPUT_SCHEMA, "strict": True}},
            )
        except openai.RateLimitError as e:
            if "insufficient_quota" in str(e) or "no credits" in str(e).lower():
                raise QuotaExhausted(str(e)[:200])
            time.sleep(15 * (attempt + 1))
            continue
        except openai.AuthenticationError as e:
            sys.exit(f"认证失败：{str(e)[:200]}\n请检查 pipeline/.env 里的 OPENAI_API_KEY。")
        except openai.BadRequestError as e:
            sys.exit(f"请求被拒（400）：{str(e)[:300]}")
        except (openai.APIConnectionError, openai.InternalServerError, openai.APITimeoutError):
            time.sleep(10 * (attempt + 1))
            continue

        u = resp.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        with lock:
            stats["in"] += u.prompt_tokens - cached
            stats["cached"] += cached
            stats["out"] += u.completion_tokens
            stats["served"][resp.model] = stats["served"].get(resp.model, 0) + 1
        content = resp.choices[0].message.content
        return json.loads(content)["labels"] if content else []
    return []


def validate(item, batch_ids):
    if item.get("comment_id") not in batch_ids:
        return None
    on, tone, fn = item.get("on_topic"), item.get("tone"), item.get("function")
    if on is False:
        return (0, NA, NA)
    if on is True and tone in TONES and fn in FUNCTIONS:
        return (1, tone, fn)
    return None  # 自相矛盾（相关却填"不适用"），不入库，重跑时补标


def cost(stats, model):
    p_in, p_cached, p_out = PRICES.get(model, PRICES["gpt-5.6-luna"])
    return (stats["in"] * p_in + stats["cached"] * p_cached + stats["out"] * p_out) / 1e6


def run(scope, run_tag, model, limit=None, batch_size=20, concurrency=5, dry_run=False, quiet=False):
    conn = open_db()
    rows = fetch_targets(conn, scope, model, run_tag, limit)
    log = (lambda *a: None) if quiet else print
    log(f"[{run_tag}|{model}|{scope}] 待标注 {len(rows)} 条")
    if not rows or dry_run:
        if dry_run and rows:
            print("\n--- system ---\n" + SYSTEM_PROMPT + "\n\n--- 第一批 user ---\n" + build_user_prompt(rows[:batch_size]))
        return {"done": 0, "cost": 0.0}

    import openai
    # Accept-Encoding: identity 绕开本机 httpx2 解压缩 bug
    client = openai.OpenAI(timeout=180, max_retries=2, default_headers={"Accept-Encoding": "identity"})
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    stats = {"in": 0, "cached": 0, "out": 0, "served": {}}
    lock = threading.Lock()
    done = skipped = 0
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [(pool.submit(annotate_batch, client, model, b, stats, lock), b) for b in batches]
            for n, (fut, batch) in enumerate(futs, 1):
                ids = {r[1]: r[0] for r in batch}
                good = {}
                for item in fut.result():
                    v = validate(item, ids)
                    if v:
                        good[item["comment_id"]] = v
                now = datetime.now(timezone.utc).isoformat()
                conn.executemany("INSERT OR REPLACE INTO ann.labels VALUES (?,?,?,?,?,?,?,?,?)",
                                 [(ids[c], c, CODEBOOK, model, run_tag, *v, now) for c, v in good.items()])
                conn.commit()
                done += len(good)
                skipped += len(batch) - len(good)
                if n % 25 == 0 or n == len(batches):
                    log(f"  批次 {n}/{len(batches)} | 写入 {done} | 跳过 {skipped} | ${cost(stats, model):.3f}")
    except QuotaExhausted as e:
        print(f"余额不足，已停止，已保存 {done} 条。充值后重跑会自动续上。{e}", file=sys.stderr)
    spent = cost(stats, model)
    log(f"  完成：写入 {done}，跳过 {skipped}，{(time.time() - t0) / 60:.1f} 分钟，${spent:.3f}，服务模型 {stats['served']}")
    conn.close()
    return {"done": done, "skipped": skipped, "cost": spent, "served": stats["served"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", required=True, help="core / pre / 某个 v1 轮次名（如 pilot500）")
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--model", default=None, help="覆盖 .env 里的 OPENAI_ANNOTATE_MODEL")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    load_dotenv()
    model = args.model or os.environ.get("OPENAI_ANNOTATE_MODEL", "gpt-5.6-luna")
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("没找到 OPENAI_API_KEY：把 pipeline/.env.example 复制为 pipeline/.env，填上 key。")
    run(args.scope, args.run_tag, model, args.limit, args.batch_size, args.concurrency, args.dry_run)


if __name__ == "__main__":
    main()

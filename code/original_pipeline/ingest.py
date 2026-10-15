#!/usr/bin/env python3
"""
ingest.py —— 清洗/合并/去重五平台原始评论数据，写入统一 SQLite 数据库

用法：
    cd "项目根目录（7.数据清洗与标注 的上一级）"
    python3 "7.数据清洗与标注/pipeline/ingest.py"

只读取项目根目录下 6.爬取的评论数据/ 里的原始 JSON，不修改任何源文件；
输出写入 7.数据清洗与标注/db/heritage.sqlite（如已存在会先自动备份改名，不覆盖）。
"""
import json
import os
import sqlite3
from datetime import datetime, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 7.数据清洗与标注/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # 项目根目录，6.爬取的评论数据 在这一级
DATA_DIR = os.path.join(PROJECT_ROOT, "6.爬取的评论数据")
DB_PATH = os.path.join(BASE_DIR, "db", "heritage.sqlite")

CST = timezone(timedelta(hours=8))
# 事件基准日期：国家文物局在驻美使馆接收史密森尼学会返还子弹库帛书《五行令》《攻守占》的
# 交接仪式日期，来自知乎内容原文确认（"5月16日，国家文物局...接收...史密森尼学会...返还的子弹库帛书"）
EVENT_DATE = datetime(2025, 5, 16, tzinfo=CST)
# 相关度判定（严格版）。三类已实测的假阳性必须排除：
# 1. 其他帛书：搜"帛书"在B站采到的818条全是帛书老子/马王堆，0条属于本事件
# 2. 史密森尼本身：它是庞大机构（国家动物园、十几座博物馆），单独出现时在各平台
#    主要指大熊猫和博物馆游记，知乎4010条/B站3000条/小红书125条严格判定全为0%
# 3. "楚帛"会匹配楚帛画（人物御龙帛画等其他文物），所以用完整的"楚帛书"
# 帛书传奇：央视专讲子弹库帛书的纪录片（2024年9月开播），相关讨论是研究问题2的事件前基线
STRONG_EVENT_TERMS = ["子弹库", "楚帛书", "战国帛书", "长沙帛书", "帛书传奇"]
EVENT_CONTEXT_TERMS = ["返还", "归还", "追索", "回归", "回家", "归湘"]
# 不排除《盗墓笔记》：大量真实讨论子弹库帛书的帖子会以"盗墓笔记中战国帛书的原型"开头，
# 排除会误杀；纯粉丝内容本身不含事件专名，已经会被判为不相关。
OTHER_RELIC_TERMS = ["马王堆", "老子", "道德经", "周易", "纵横家", "天文气象", "五十二病方", "帛画"]

PLATFORM_DIRS = {
    "douyin": "1.douyin/json",
    "bilibili": "2.bilibili",
    "xhs": "3.xhs/json",
    "weibo": "4.weibo/json",
    "zhihu": "5.zhihu/json",
}

# 各平台字段名差异映射（评论用于关联帖子的 id 字段名不同，如 aweme_id/video_id/note_id/content_id）
POST_ID_FIELD = {"douyin": "aweme_id", "bilibili": "video_id", "xhs": "note_id",
                  "weibo": "note_id", "zhihu": "content_id"}
COMMENT_TIME_FIELD = {"douyin": "create_time", "bilibili": "create_time", "xhs": "create_time",
                       "weibo": "create_time", "zhihu": "publish_time"}
COMMENT_TIME_MS = {"xhs"}  # 小红书评论时间戳是毫秒，其余平台是秒
COMMENT_LIKE_FIELD = {"douyin": "like_count", "bilibili": "like_count", "xhs": "like_count",
                       "weibo": "comment_like_count", "zhihu": "like_count"}
COMMENT_NICK_FIELD = {"douyin": "nickname", "bilibili": "nickname", "xhs": "nickname",
                       "weibo": "nickname", "zhihu": "user_nickname"}
POST_TIME_FIELD = {"douyin": "create_time", "bilibili": "create_time", "xhs": "time",
                    "weibo": "create_time", "zhihu": "created_time"}
POST_TIME_MS = {"xhs"}  # 小红书帖子时间戳同样是毫秒
POST_TEXT_FIELDS = {  # 拼接后用于关键词相关度判断、并存入 posts.text
    "douyin": ["title", "desc"],
    "bilibili": ["title", "desc"],
    "xhs": ["title", "desc"],
    "weibo": ["content"],
    "zhihu": ["title", "desc", "content_text"],
}


def load_json_files(base, rel_dir, prefix):
    """读取目录下所有 {prefix}_*.json（只扫当前层，不进 backup 子目录）。"""
    items = []
    d = os.path.join(base, rel_dir)
    for fn in sorted(os.listdir(d)):
        if fn.startswith(prefix + "_") and fn.endswith(".json"):
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                items.extend(json.load(f))
    return items


def to_epoch_seconds(value, is_ms):
    if value in (None, ""):
        return None
    try:
        v = int(value)
    except (ValueError, TypeError):
        return None
    if is_ms or v > 10 ** 12:  # 兜底：万一某条记录本身就是毫秒但平台默认是秒
        v = v // 1000
    return v


def is_relevant(text):
    """帖子是否在讲这次子弹库帛书回归事件（而不是别的帛书/别的文物归还事件）。"""
    if not text:
        return False
    names_this_relic = "子弹库" in text or "楚帛书" in text
    if any(term in text for term in OTHER_RELIC_TERMS) and not names_this_relic:
        return False
    if any(term in text for term in STRONG_EVENT_TERMS):
        return True
    # 史密森尼/帛书 单独出现都不够，必须同时有"帛书"和归还语境
    return "帛书" in text and any(term in text for term in EVENT_CONTEXT_TERMS)


def build_post_index(platform, contents):
    """post_id -> {keyword, timestamp, relevant}，供评论 join 缺失的 source_keyword 用"""
    idx = {}
    id_field = POST_ID_FIELD[platform]
    time_field = POST_TIME_FIELD[platform]
    is_ms = platform in POST_TIME_MS
    text_fields = POST_TEXT_FIELDS[platform]
    for p in contents:
        pid = p.get(id_field)
        if pid is None:
            continue
        text = "".join(str(p.get(tf, "") or "") for tf in text_fields)
        idx[pid] = {
            # 指定ID深挖(detail_)的帖子记录没有搜索词，保留同一帖子在搜索记录里的关键词
            "keyword": p.get("source_keyword") or idx.get(pid, {}).get("keyword"),
            "timestamp": to_epoch_seconds(p.get(time_field), is_ms),
            "relevant": is_relevant(text),
        }
    return idx


def process_comments(platform, comments, post_idx):
    id_field = POST_ID_FIELD[platform]
    time_field = COMMENT_TIME_FIELD[platform]
    is_ms = platform in COMMENT_TIME_MS
    like_field = COMMENT_LIKE_FIELD[platform]
    nick_field = COMMENT_NICK_FIELD[platform]

    rows = []
    seen_ids = set()
    dup_count = 0
    event_ts = EVENT_DATE.timestamp()

    for c in comments:
        cid = c.get("comment_id")
        if cid is None:
            continue
        if cid in seen_ids:
            dup_count += 1
            continue
        seen_ids.add(cid)

        post_id = c.get(id_field)
        post_info = post_idx.get(post_id, {})

        direct_kw = c.get("source_keyword")
        keyword = direct_kw or post_info.get("keyword")
        if direct_kw:
            keyword_source = "direct"
        elif post_info.get("keyword"):
            keyword_source = "post_join"
        else:
            keyword_source = None

        ts = to_epoch_seconds(c.get(time_field), is_ms)
        try:
            like_count = int(c.get(like_field) or 0)
        except (ValueError, TypeError):
            like_count = 0

        rows.append({
            "platform": platform,
            "post_id": post_id,
            "comment_id": cid,
            "parent_comment_id": c.get("parent_comment_id"),
            "text": c.get("content"),
            "timestamp": ts,
            "timestamp_iso": datetime.fromtimestamp(ts, tz=CST).isoformat() if ts else None,
            "like_count": like_count,
            "anon_user_id": c.get("creator_hash"),
            "nickname_masked": c.get(nick_field),
            "crawl_round": 1,
            "keyword": keyword,
            "keyword_source": keyword_source,
            "post_relevant": post_info.get("relevant"),
            "is_post_event": (1 if ts >= event_ts else 0) if ts is not None else None,
        })

    return rows, dup_count


def process_posts(platform, contents, post_idx):
    id_field = POST_ID_FIELD[platform]
    text_fields = POST_TEXT_FIELDS[platform]
    event_ts = EVENT_DATE.timestamp()

    rows = []
    seen = set()
    for p in contents:
        pid = p.get(id_field)
        if pid is None or pid in seen:
            continue
        seen.add(pid)
        text = " ".join(str(p.get(tf, "") or "") for tf in text_fields).strip()
        info = post_idx.get(pid, {})
        ts = info.get("timestamp")
        rows.append({
            "platform": platform,
            "post_id": pid,
            "text": text,
            "keyword": p.get("source_keyword"),
            "timestamp": ts,
            "timestamp_iso": datetime.fromtimestamp(ts, tz=CST).isoformat() if ts else None,
            "anon_user_id": p.get("creator_hash"),
            "is_relevant": 1 if info.get("relevant") else 0,
            "is_post_event": (1 if ts >= event_ts else 0) if ts is not None else None,
        })
    return rows


def main():
    os.makedirs(os.path.join(BASE_DIR, "db"), exist_ok=True)
    if os.path.exists(DB_PATH):
        backup = DB_PATH + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(DB_PATH, backup)
        print(f"已备份旧数据库到: {backup}\n")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE comments (
        platform TEXT, post_id TEXT, comment_id TEXT, parent_comment_id TEXT,
        text TEXT, timestamp INTEGER, timestamp_iso TEXT, like_count INTEGER,
        anon_user_id TEXT, nickname_masked TEXT, crawl_round INTEGER,
        keyword TEXT, keyword_source TEXT, post_relevant INTEGER, is_post_event INTEGER,
        PRIMARY KEY (platform, comment_id)
    );
    CREATE TABLE posts (
        platform TEXT, post_id TEXT, text TEXT, keyword TEXT,
        timestamp INTEGER, timestamp_iso TEXT, anon_user_id TEXT,
        is_relevant INTEGER, is_post_event INTEGER,
        PRIMARY KEY (platform, post_id)
    );
    CREATE INDEX idx_comments_platform ON comments(platform);
    CREATE INDEX idx_comments_keyword ON comments(keyword);
    CREATE INDEX idx_comments_post_event ON comments(is_post_event);
    CREATE INDEX idx_comments_relevant ON comments(post_relevant);
    CREATE INDEX idx_posts_platform ON posts(platform);
    """)

    print(f"事件基准日期: {EVENT_DATE.date()} （国家文物局接收仪式，据知乎内容原文确认）")
    print(f"事件专名词: {STRONG_EVENT_TERMS}")
    print(f"排除的其他文物: {OTHER_RELIC_TERMS}")
    print(f"数据源目录: {DATA_DIR}\n")

    grand_total = 0
    for platform, rel_dir in PLATFORM_DIRS.items():
        # search_ = 关键词搜索模式，detail_ = 指定ID深挖模式；search 在前，同一评论去重时保留带搜索词的那条
        contents = (load_json_files(DATA_DIR, rel_dir, "search_contents")
                    + load_json_files(DATA_DIR, rel_dir, "detail_contents"))
        comments = (load_json_files(DATA_DIR, rel_dir, "search_comments")
                    + load_json_files(DATA_DIR, rel_dir, "detail_comments"))
        post_idx = build_post_index(platform, contents)

        comment_rows, dup_count = process_comments(platform, comments, post_idx)
        post_rows = process_posts(platform, contents, post_idx)

        cur.executemany(
            """INSERT OR IGNORE INTO comments
               (platform, post_id, comment_id, parent_comment_id, text, timestamp, timestamp_iso,
                like_count, anon_user_id, nickname_masked, crawl_round, keyword, keyword_source,
                post_relevant, is_post_event)
               VALUES (:platform, :post_id, :comment_id, :parent_comment_id, :text, :timestamp,
                :timestamp_iso, :like_count, :anon_user_id, :nickname_masked, :crawl_round,
                :keyword, :keyword_source, :post_relevant, :is_post_event)""",
            comment_rows,
        )
        cur.executemany(
            """INSERT OR IGNORE INTO posts
               (platform, post_id, text, keyword, timestamp, timestamp_iso, anon_user_id,
                is_relevant, is_post_event)
               VALUES (:platform, :post_id, :text, :keyword, :timestamp, :timestamp_iso,
                :anon_user_id, :is_relevant, :is_post_event)""",
            post_rows,
        )

        n_relevant = sum(1 for r in comment_rows if r["post_relevant"])
        n_irrelevant = sum(1 for r in comment_rows if r["post_relevant"] is False)
        n_unknown = sum(1 for r in comment_rows if r["post_relevant"] is None)
        n_post_event = sum(1 for r in comment_rows if r["is_post_event"] == 1)
        n_no_keyword = sum(1 for r in comment_rows if not r["keyword"])

        print(f"[{platform}] 评论 {len(comment_rows)} 条唯一（跳过同批次内重复 {dup_count} 条），"
              f"帖子 {len(post_rows)} 条")
        print(f"    帖子相关={n_relevant}  帖子不相关={n_irrelevant}  未匹配到帖子={n_unknown}  |  "
              f"事件后(≥2025-05-16)={n_post_event}  |  缺关键词={n_no_keyword}")

        grand_total += len(comment_rows)

    conn.commit()
    conn.close()

    print(f"\n=== 合计写入 {grand_total} 条唯一评论 ===")
    print(f"数据库文件: {DB_PATH}")
    print("\n提示: comments 表里 post_relevant / is_post_event 是标记列，不是删除，"
          "后续按需用 WHERE 过滤即可，原始 JSON 完全未改动。")


if __name__ == "__main__":
    main()

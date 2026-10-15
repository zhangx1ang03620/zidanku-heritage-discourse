#!/usr/bin/env python3
"""
build_dataset.py —— 合成分析用数据集 data/analysis_dataset.csv

一行一条"相关帖子下、有正文、已用编码本 v2 标注"的评论（主分析标签 = gpt-5.6-luna / full_v2）。
不含用户 ID 和昵称。

字段说明中容易踩坑的两处：
  is_reply   微博一级评论的 parent_comment_id 等于它自己的 ID（不是 0），必须排除这种情况，
             否则微博会被误判为 100% 回复（实际约 4%）。
  coverage   该帖子我们采到的评论数 / 平台显示的评论数。抖音、小红书默认按热度返回评论，
             每帖只取前若干条，覆盖率低的帖子样本偏向高赞评论；互动量分析用它做稳健性检验。
"""
import glob
import json
import os
import sqlite3

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = os.path.join(ROOT, "7.数据清洗与标注", "db", "heritage.sqlite")
ANN_DB = os.path.join(ROOT, "7.数据清洗与标注", "db", "annotations.sqlite")
RAW = os.path.join(ROOT, "6.爬取的评论数据")
OUT = os.path.join(ROOT, "8.分析与图表", "data", "analysis_dataset.csv")

RAW_DIRS = {"douyin": "1.douyin/json", "bilibili": "2.bilibili", "xhs": "3.xhs/json",
            "weibo": "4.weibo/json", "zhihu": "5.zhihu/json"}
POST_ID = {"douyin": "aweme_id", "bilibili": "video_id", "xhs": "note_id", "weibo": "note_id", "zhihu": "content_id"}
PLATFORM_COMMENT_COUNT = {"douyin": "comment_count", "bilibili": "video_comment", "xhs": "comment_count",
                          "weibo": "comments_count", "zhihu": "comment_count"}

# 阶段划分依据见实验日志第 3 节
PHASES = [
    ("P1", "Background", None, "2024-09-01"),
    ("P2", "Documentary & advocacy", "2024-09-01", "2025-05-16"),
    ("P3", "Return peak", "2025-05-16", "2025-06-01"),
    ("P4", "Consolidation", "2025-06-01", "2025-10-01"),
    ("P5", "Return to Hunan", "2025-10-01", "2025-12-01"),
    ("P6", "Long tail", "2025-12-01", None),
]

EN = {
    "platform": {"bilibili": "Bilibili", "douyin": "Douyin", "xhs": "Xiaohongshu", "weibo": "Weibo", "zhihu": "Zhihu"},
    "tone": {"喜悦自豪": "Joy & pride", "愤慨痛惜": "Indignation & grief", "中性理性": "Neutral/informational",
             "质疑异见": "Skepticism & dissent"},
    "function": {"情感表态": "Affective expression", "追索诉求与国家能力": "Restitution claims & state capacity",
                 "文物知识与归还细节": "Artifact knowledge & return details",
                 "历史与文明论战": "Historical & civilizational debate",
                 "处置过程的信任与怀疑": "Institutional trust & distrust"},
}


def platform_comment_counts():
    """从原始帖子 JSON 读平台显示的评论数（同一帖子多次出现取最大值）"""
    rows = []
    for plat, d in RAW_DIRS.items():
        for f in glob.glob(os.path.join(RAW, d, "*_contents_*.json")):
            for p in json.load(open(f, encoding="utf-8")):
                try:
                    n = int(str(p.get(PLATFORM_COMMENT_COUNT[plat]) or 0).replace(",", ""))
                except ValueError:
                    n = 0
                rows.append((plat, str(p.get(POST_ID[plat])), n))
    df = pd.DataFrame(rows, columns=["platform", "post_id", "platform_comment_count"])
    return df.groupby(["platform", "post_id"], as_index=False).platform_comment_count.max()


def assign_phase(ts):
    for code, _, start, end in PHASES:
        if (start is None or ts >= start) and (end is None or ts < end):
            return code
    return None


def main():
    conn = sqlite3.connect(DB)
    conn.execute("ATTACH DATABASE ? AS ann", (ANN_DB,))
    df = pd.read_sql("""
        SELECT c.platform, c.post_id, c.comment_id, c.parent_comment_id, c.timestamp_iso, c.like_count,
               c.keyword, c.is_post_event, LENGTH(c.text) AS text_len,
               l.on_topic, l.tone, l.function
        FROM comments c
        JOIN ann.labels l ON l.platform = c.platform AND l.comment_id = c.comment_id
             AND l.codebook = 'v2' AND l.model = 'gpt-5.6-luna' AND l.run_tag = 'full_v2'
        WHERE c.post_relevant = 1
    """, conn)

    pid = df.parent_comment_id.fillna("").astype(str)
    df["is_reply"] = (~pid.isin(["", "0"]) & (pid != df.comment_id.astype(str))).astype(int)
    df["date"] = df.timestamp_iso.str[:10]
    df["month"] = df.timestamp_iso.str[:7]
    df["phase"] = df.date.map(assign_phase)
    df["post_key"] = df.platform + ":" + df.post_id.astype(str)

    counts = platform_comment_counts()
    # 分子是该帖子实际采到的全部评论（含无正文的），不是已标注数——后者只是子集，会低估覆盖率
    collected = pd.read_sql("SELECT platform, post_id, COUNT(*) AS collected FROM comments GROUP BY platform, post_id", conn)
    df = df.merge(counts, on=["platform", "post_id"], how="left").merge(collected, on=["platform", "post_id"], how="left")
    df["coverage"] = np.where(df.platform_comment_count > 0, df.collected / df.platform_comment_count, np.nan)

    for col in ("platform", "tone", "function"):
        df[col + "_en"] = df[col].map(EN[col])
    df["phase_en"] = df.phase.map({c: f"{c} {n}" for c, n, _, _ in PHASES})

    df = df.drop(columns=["parent_comment_id"])
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    on = df[df.on_topic == 1]
    print(f"分析数据集 {len(df)} 条（其中 on_topic {len(on)} 条）→ {OUT}")
    print("  各平台:", df.platform.value_counts().to_dict())
    print("  各阶段 on_topic:", on.phase.value_counts().sort_index().to_dict())
    print(f"  有平台评论数的帖子占比 {df.platform_comment_count.gt(0).mean():.0%}，"
          f"覆盖率≥0.8 的评论占比 {(df.coverage >= 0.8).mean():.0%}")


if __name__ == "__main__":
    main()

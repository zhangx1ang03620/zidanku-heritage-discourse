#!/usr/bin/env python3
"""
make_sample.py —— 抽取固定随机种子的分层样本，写入 annotations.sqlite 的 samples 表

sens3000：每个平台 600 条，从"相关帖子下、有文字"的评论里（事件前 + 事件后）随机抽取，
用 gpt-5.6-terra 再标一遍，做主分析（gpt-5.6-luna）的敏感性检验。
按平台等额抽样，是为了每个平台都有足够样本检验"跨平台差异换模型后是否仍成立"。
"""
import os
import sqlite3

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = os.path.join(ROOT, "7.数据清洗与标注", "db", "heritage.sqlite")
ANN_DB = os.path.join(ROOT, "7.数据清洗与标注", "db", "annotations.sqlite")


def main(name="sens3000", per_platform=600, seed=20260921):
    conn = sqlite3.connect(DB)
    df = pd.read_sql("""SELECT platform, comment_id FROM comments
                        WHERE post_relevant = 1 AND text IS NOT NULL AND TRIM(text) != ''""", conn)
    sample = (df.groupby("platform", group_keys=False)
                .apply(lambda g: g.sample(n=min(per_platform, len(g)), random_state=seed)))
    ann = sqlite3.connect(ANN_DB)
    ann.execute("CREATE TABLE IF NOT EXISTS samples (name TEXT, platform TEXT, comment_id TEXT, "
                "PRIMARY KEY (name, platform, comment_id))")
    ann.execute("DELETE FROM samples WHERE name = ?", (name,))
    ann.executemany("INSERT INTO samples VALUES (?,?,?)",
                    [(name, r.platform, r.comment_id) for r in sample.itertuples()])
    ann.commit()
    print(f"样本 {name}: {len(sample)} 条", sample.platform.value_counts().to_dict())


if __name__ == "__main__":
    main()

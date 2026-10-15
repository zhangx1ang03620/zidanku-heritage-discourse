#!/usr/bin/env python3
"""manifest.py —— 标注运行清单（可复现用）：每个 (编码本, 模型, run_tag) 的条数、时间范围、相关率。
输出 tables/run_manifest.csv。花费见 data/*_run.log。"""
import os
import sqlite3

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ANN = os.path.join(ROOT, "7.数据清洗与标注", "db", "annotations.sqlite")

df = pd.read_sql("""
    SELECT codebook, model, run_tag, COUNT(*) AS n, AVG(on_topic) AS on_topic_rate,
           MIN(annotated_at) AS first_at, MAX(annotated_at) AS last_at
    FROM labels GROUP BY 1, 2, 3 ORDER BY first_at""", sqlite3.connect(ANN))
df["on_topic_rate"] = df["on_topic_rate"].round(3)
out = os.path.join(os.path.dirname(HERE), "tables", "run_manifest.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")
print(df.to_string(index=False))

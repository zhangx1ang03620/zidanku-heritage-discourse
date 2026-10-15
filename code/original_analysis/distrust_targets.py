#!/usr/bin/env python3
"""
distrust_targets.py —— 【探索性，全量结果出来后才提出，不属于冻结计划】
"处置过程的信任与怀疑"类评论（事件后，n≈251）指向的对象是什么？
luna 与 terra 各标一次，报告二者一致性；结果写 annotations.sqlite 的 distrust_targets 表和 tables/T13。
API key 只从 pipeline/.env 读进环境变量，不打印。
"""
import json
import os
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
PIPE = os.path.join(os.path.dirname(OUT), "7.数据清洗与标注", "pipeline")
sys.path.insert(0, PIPE)
import llm_annotate as la  # noqa: E402

TARGETS = ["国内保管与归属", "真伪与鉴定", "过程透明与交易代价", "外方动机", "官方宣传与叙事", "其他"]
STANCES = ["怀疑或不信任", "信任或为机构辩护", "调侃无明确立场"]
PROMPT = f"""下面是关于"长沙子弹库战国帛书"从美国归还中国（2025年5月）的社交媒体评论，它们已被判定为涉及"处置过程的信任与怀疑"。
请判断每条评论的【指向对象】和【立场】。
指向对象（单选）：
- 国内保管与归属：文物归还后存放在哪里（国家博物馆/北京 vs 湖南/湖南博物院）、会不会被"借走不还"、保管是否安全（如火灾）
- 真伪与鉴定：怀疑归还的是不是真品、是否被调包、如何鉴定
- 过程透明与交易代价：是否花钱买回、谈判条件、信息是否公开
- 外方动机：美方/持有机构为何归还、是否另有目的
- 官方宣传与叙事：对官方媒体报道、宣传口径本身的怀疑或评价
- 其他：以上都不是
立场（单选）：怀疑或不信任 / 信任或为机构辩护 / 调侃无明确立场
逐条返回 id、target、stance。"""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["labels"], "properties": {"labels": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "target", "stance"],
                               "properties": {"id": {"type": "string"}, "target": {"type": "string", "enum": TARGETS},
                                              "stance": {"type": "string", "enum": STANCES}}}}}}


def call(client, model, batch, stats, lock):
    # 批内用短编号 1..n（长 id 模型容易抄错），返回后映射回原 id
    user = "\n".join(f"[{k + 1}] {t}" for k, (_, t) in enumerate(batch))
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {"name": "t", "schema": SCHEMA, "strict": True}})
    with lock:
        stats["in"] += resp.usage.prompt_tokens
        stats["out"] += resp.usage.completion_tokens
    idmap = {str(k + 1): i for k, (i, _) in enumerate(batch)}
    out = []
    for x in json.loads(resp.choices[0].message.content)["labels"]:
        key = x["id"].strip("[] ")
        if key in idmap:
            out.append({**x, "id": idmap[key]})
    return out


def main():
    la.load_dotenv()
    import openai
    client = openai.OpenAI(timeout=180, max_retries=3, default_headers={"Accept-Encoding": "identity"})
    df = pd.read_csv(os.path.join(OUT, "data", "institutional_trust_comments.csv"))
    df["id"] = df.platform + ":" + df.comment_id.astype(str)
    items = list(zip(df.id, df.text.astype(str).str[:300]))
    batches = [items[i:i + 25] for i in range(0, len(items), 25)]
    conn = sqlite3.connect(la.ANN_DB_PATH)
    conn.execute("DROP TABLE IF EXISTS distrust_targets")
    conn.execute("CREATE TABLE IF NOT EXISTS distrust_targets (id TEXT, model TEXT, target TEXT, stance TEXT, "
                 "PRIMARY KEY (id, model))")
    total = 0.0
    for model in ("gpt-5.6-luna", "gpt-5.6-terra"):
        stats, lock = {"in": 0, "out": 0}, threading.Lock()
        with ThreadPoolExecutor(4) as ex:
            res = [x for r in ex.map(lambda b: call(client, model, b, stats, lock), batches) for x in r]
        conn.executemany("INSERT OR REPLACE INTO distrust_targets VALUES (?,?,?,?)",
                         [(x["id"], model, x["target"], x["stance"]) for x in res])
        conn.commit()
        p = la.PRICES[model]
        c = stats["in"] / 1e6 * p[0] + stats["out"] / 1e6 * p[2]
        total += c
        print(f"{model}: {len(res)}/{len(items)} 条，${c:.3f}")
    lab = pd.read_sql("SELECT * FROM distrust_targets", conn).pivot(index="id", columns="model", values=["target", "stance"])
    lab = lab.dropna()
    k_t = cohen_kappa_score(lab["target"]["gpt-5.6-luna"], lab["target"]["gpt-5.6-terra"])
    k_s = cohen_kappa_score(lab["stance"]["gpt-5.6-luna"], lab["stance"]["gpt-5.6-terra"])
    print(f"luna vs terra：target κ={k_t:.3f}，stance κ={k_s:.3f}（n={len(lab)}）")
    tab = pd.crosstab(lab["target"]["gpt-5.6-terra"], lab["stance"]["gpt-5.6-terra"], margins=True)
    tab.to_csv(os.path.join(OUT, "tables", "T13_distrust_targets_terra.csv"), encoding="utf-8-sig")
    lum = pd.crosstab(lab["target"]["gpt-5.6-luna"], lab["stance"]["gpt-5.6-luna"], margins=True)
    lum.to_csv(os.path.join(OUT, "tables", "T13b_distrust_targets_luna.csv"), encoding="utf-8-sig")
    print(tab.to_string())
    print(lum.to_string())
    print(f"合计花费 ${total:.3f}")


if __name__ == "__main__":
    main()

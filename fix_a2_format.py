"""修复A2.csv格式: 数字索引 -> 物品ID (uid,prediction)"""
import os, sys, csv
import pandas as pd

REC_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "A推荐", "A推荐")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# 加载物品映射
item_df = pd.read_csv(os.path.join(REC_DATA, "item.csv"))
all_items = sorted(item_df["iid"].unique())
idx2item = {idx + 1: iid for idx, iid in enumerate(all_items)}  # 0是padding

# 读取当前错误的A2.csv (user_id,rec_1,...,rec_10)
rows = []
with open(os.path.join(OUTPUT_DIR, "A2.csv"), "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        uid = row[0]
        pred_indices = [int(x) for x in row[1:]]
        # 转换为物品ID
        items = [idx2item.get(idx, "i000001") for idx in pred_indices if idx > 0]
        while len(items) < 10:
            items.append("i000001")
        rows.append((uid, items[:10]))

# 写入正确格式
with open(os.path.join(OUTPUT_DIR, "A2.csv"), "w", encoding="utf-8") as f:
    f.write("uid,prediction\n")
    for uid, items in rows:
        f.write(f'{uid},"{",".join(items)}"\n')

print(f"修复完成: {len(rows)}行, 格式: uid,prediction")
print(f"示例: {rows[0][0]},\"{','.join(rows[0][1])}\"")

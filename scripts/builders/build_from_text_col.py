import json
import re
import pandas as pd


df = pd.read_csv("data/unprocessed/text_columns.csv")

def normalize_item(s):
    if not s or s == "None":
        return None
    # CamelCase → whitespace
    s = re.sub(r'(?<!^)(?=[A-Z])', ' ', s)
    # lowercase
    s = s.lower().strip()
    
    return s

def process_col(cell):
    if pd.isna(cell):
        return []
    if not isinstance(cell, str):
        return []
    items = cell.split(",")
    normalized = [normalize_item(x) for x in items]
    return list(set(filter(None, normalized)))

cols = df.columns.to_list()
result = {}
for col in cols:
    df[col] = df[col].apply(process_col)
    lst = list(set(item for sublist in df[col] for item in sublist if item != "see remarks"))
    result[col] = lst
with open("data/processed/text_features_from_db.json", "w") as f:
    json.dump(result, f, indent = 2)
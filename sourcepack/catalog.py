"""检查资料容器，不解释业务规则。"""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import math

def read_table(path, table):
    """读取一张资料表并验证字段、基本类型和记录键。"""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        names = [c["name"] for c in table["columns"]]
        if reader.fieldnames != names:
            raise ValueError("字段不匹配：" + str(path))
        result = list(reader)
    seen = set()
    for index, row in enumerate(result, 2):
        for column in table["columns"]:
            value = row[column["name"]]
            if value is None:
                raise ValueError(f"列数不匹配：{index}")
            if value == "" and column.get("nullable", False):
                continue
            try:
                kind = column["type"]
                if kind == "number":
                    if not math.isfinite(float(value)):
                        raise ValueError()
                elif kind == "integer":
                    int(value)
                elif kind == "boolean" and value not in {"true", "false"}:
                    raise ValueError()
                elif kind == "date":
                    datetime.date.fromisoformat(value)
                elif kind == "datetime":
                    stamp = datetime.datetime.fromisoformat(value)
                    if stamp.tzinfo is None:
                        raise ValueError()
            except (TypeError, ValueError):
                raise ValueError(f"类型不匹配：{index}:{column['name']}") from None
        key = tuple(row[name] for name in table.get("key", []))
        if key:
            if "" in key or key in seen:
                raise ValueError(f"记录键缺失或重复：{index}")
            seen.add(key)
    return result

def inspect(root):
    """核对资料摘要及声明的引用关系。"""
    root = Path(root).resolve()
    catalog = json.loads((root / "data/catalog.json").read_text(encoding="utf-8"))
    records, report = {}, []
    for table in catalog["tables"]:
        path = (root / table["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("文件路径无效")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != table["sha256"]:
            raise ValueError("资料摘要不一致：" + table["name"])
        data = read_table(path, table)
        if len(data) != table["row_count"]:
            raise ValueError("记录数不一致：" + table["name"])
        records[table["name"]] = data
        report.append({"name": table["name"], "rows": len(data), "sha256": digest})
    for table in catalog["tables"]:
        for reference in table.get("references", []):
            allowed = {r[reference["target_column"]] for r in records[reference["target_table"]]}
            for row in records[table["name"]]:
                value = row[reference["column"]]
                if value and value not in allowed:
                    raise ValueError("资料引用不存在：" + table["name"])
    return {"title": catalog["title"], "status": "ok", "tables": report}

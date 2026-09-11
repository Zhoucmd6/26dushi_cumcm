"""只整理和检查 C 题输入；不假定区间起止，不预测、不优化、不填写模板。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import date, datetime, time, timedelta
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LABELS = [f"{i // 6:02d}:{i % 6 * 10:02d}" for i in range(1, 144)] + ["24:00"]
EXPECTED_DATES = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(365)]
INPUT_FILES = ["C题.pdf", *[f"附件/附件{i}.xlsx" for i in range(1, 5)],
               *[f"附件/附件5/{n}.xlsx" for n in ("result1", "result2", "result3", "result4-2", "result4-3")]]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(source):
    """保存准确副本；已有副本如与来源不同，停止并提示人工处理。"""
    source = source.resolve()
    for relative in INPUT_FILES:
        src, dst = source / relative, ROOT / "data/raw" / relative
        if not src.is_file():
            raise FileNotFoundError(f"缺少输入：{src}")
        if dst.exists() and sha256(src) != sha256(dst):
            raise ValueError(f"来源与已有副本不同，请先核实版本：{relative}")
    for relative in INPUT_FILES:
        src, dst = source / relative, ROOT / "data/raw" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
    manifest = {"source_directory": str(source), "files": [
        {"file": p, "sha256": sha256(ROOT / "data/raw" / p)} for p in INPUT_FILES]}
    (ROOT / "data/raw/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_snapshot():
    manifest_path = ROOT / "data/raw/manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("首次运行请传入 --source，保存输入副本及来源记录。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if sha256(ROOT / "data/raw" / item["file"]) != item["sha256"]:
            raise ValueError(f"原始副本已改变：{item['file']}；请先核实。")


def rows(filename, sheet):
    wb = openpyxl.load_workbook(ROOT / "data/raw/附件" / filename, read_only=True, data_only=True)
    try:
        return list(wb[sheet].values)
    finally:
        wb.close()


def clock_label(value):
    if isinstance(value, (datetime, time)):
        return value.strftime("%H:%M")
    value = str(value).strip()
    if value in ("0:00+1", "00:00+1", "24:00"):
        return "24:00"
    h, m = value.split(":")
    return f"{int(h):02d}:{int(m):02d}"


def date_label(value):
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d")
    return datetime.strptime(str(value).strip(), "%Y-%m-%d").date().isoformat()


def number(value, location):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"数值缺失或不是有限数：{location}: {value!r}")
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(name, headers, data):
    with (ROOT / "data/processed" / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)


def read_daily(filename, sheet):
    table = rows(filename, sheet)
    require(len(table) == 366, f"{filename}/{sheet} 应有 365 个日期")
    labels = [clock_label(v) for v in table[0][1:]]
    require(labels == EXPECTED_LABELS, f"{filename}/{sheet} 时间标签不符合预期")
    dates = [date_label(r[0]) for r in table[1:]]
    require(dates == EXPECTED_DATES, f"{filename}/{sheet} 日期不连续或存在重复")
    data = [[number(v, f"{filename}/{sheet} 行{ridx}列{cidx}")
             for cidx, v in enumerate(r[1:], 2)] for ridx, r in enumerate(table[1:], 2)]
    require(all(len(r) == 144 for r in data), f"{filename}/{sheet} 每天应有 144 个数值")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="桌面 C题 文件夹；首次运行需要")
    args = parser.parse_args()
    for folder in ("data/raw", "data/processed", "reports", "outputs/figures", "outputs/results"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    if args.source:
        snapshot(args.source)
    verify_snapshot()

    baseline = rows("附件1.xlsx", "Sheet1")
    require(len(baseline) == 145, "附件1应有 144 条数据")
    require([clock_label(r[0]) for r in baseline[1:]] == EXPECTED_LABELS, "附件1时间标签异常")
    base = [[i, clock_label(r[0]), *[number(v, f"附件1 行{i+1}") for v in r[1:]]]
            for i, r in enumerate(baseline[1:], 1)]
    require(all(len(r) == 5 for r in base), "附件1字段数异常")

    load = read_daily("附件2.xlsx", "小区负载")
    pv = read_daily("附件2.xlsx", "光伏发电实际功率")
    prices = read_daily("附件4.xlsx", "Sheet1")
    actual = [[d, i + 1, t, load[j][i], pv[j][i]]
              for j, d in enumerate(EXPECTED_DATES) for i, t in enumerate(EXPECTED_LABELS)]
    price_points = [[d, i + 1, t, prices[j][i]]
                    for j, d in enumerate(EXPECTED_DATES) for i, t in enumerate(EXPECTED_LABELS)]

    source_forecast = rows("附件3.xlsx", "Sheet1")
    require(len(source_forecast) == 1461, "附件3应有 1460 批预报")
    require(list(source_forecast[0][2:]) == [f"预报{i}小时" for i in range(1, 25)], "附件3预报列异常")
    forecasts, seen, current_date, filled = [], set(), None, 0
    for ridx, row in enumerate(source_forecast[1:], 2):
        if row[0] not in (None, ""):
            current_date = date_label(row[0])
        else:
            filled += 1
        require(current_date is not None, f"附件3第{ridx}行日期无法确定")
        issued_clock = clock_label(row[1])
        require(issued_clock in ("00:00", "06:00", "12:00", "18:00"), "预报发布时间异常")
        key = (current_date, issued_clock)
        require(key not in seen, f"重复预报批次：{key}")
        seen.add(key)
        issued = datetime.fromisoformat(f"{current_date}T{issued_clock}:00")
        require(len(row[2:]) == 24, f"附件3第{ridx}行不是24个预报值")
        for lead, value in enumerate(row[2:], 1):
            forecasts.append([issued.isoformat(), lead, (issued + timedelta(hours=lead)).isoformat(),
                              number(value, f"附件3 行{ridx}预报{lead}小时")])
    require(seen == {(d, t) for d in EXPECTED_DATES for t in ("00:00", "06:00", "12:00", "18:00")},
            "预报日期或发布批次不完整")

    # 所有输入检查成功后再输出；不做时段偏移、插值或未来数据切分。
    write_csv("baseline_points.csv", ["point_index", "time_label", "price_yuan_per_kwh", "load_kw", "pv_forecast_kw"], base)
    write_csv("actual_points.csv", ["date", "point_index", "time_label", "load_kw", "pv_actual_kw"], actual)
    write_csv("price_points.csv", ["date", "point_index", "time_label", "price_yuan_per_kwh"], price_points)
    write_csv("pv_forecasts.csv", ["issued_at", "lead_hours", "valid_at", "pv_forecast_kw"], forecasts)

    template_notes = []
    for name in ("result1", "result2", "result3", "result4-2", "result4-3"):
        wb = openpyxl.load_workbook(ROOT / f"data/raw/附件/附件5/{name}.xlsx", read_only=True, data_only=True)
        try:
            sheet = wb["计划购电量"]
            first, last = (sheet["A2"].value, sheet["A145"].value) if name == "result1" else (sheet["B1"].value, sheet["EO1"].value)
            template_notes.append({"file": name + ".xlsx", "first_interval": first, "last_interval": last})
        finally:
            wb.close()

    summary = {
        "baseline_points": len(base), "actual_points": len(actual), "price_points": len(price_points),
        "forecast_batches": len(seen), "forecast_points": len(forecasts),
        "forecast_date_labels_filled": filled,
        "date_start": EXPECTED_DATES[0], "date_end": EXPECTED_DATES[-1],
        "checks": ["原始副本SHA-256一致", "全年日期连续且无重复", "每日时间标签一致", "数值均有限且无缺失", "预报批次完整且无重复"],
        "template_labels": template_notes,
        "ranges": {
            "load_kw": [min(r[3] for r in actual), max(r[3] for r in actual)],
            "pv_actual_kw": [min(r[4] for r in actual), max(r[4] for r in actual)],
            "price_yuan_per_kwh": [min(r[3] for r in price_points), max(r[3] for r in price_points)]},
        "unresolved": ["输入时间点如何对应10分钟区间", "模板首尾时间标签", "充放电效率与接口口径", "调减购电款是否退还及多次调整结算", "未来电价何时可知", "跨日及期末储电要求"]}
    (ROOT / "reports/data_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# C题数据检查结果", "", "只读检查并保留原件副本；尚未预测、优化或填写结果模板。", "",
             "| 检查对象 | 结果 |", "|---|---|",
             f"| 附件1 | {len(base)} 个时间点，3列数值完整 |",
             f"| 附件2 | 365天 × 144个时间点 = {len(actual)} 行，负载和实际光伏完整 |",
             f"| 附件3 | {len(seen)} 批 × 24小时 = {len(forecasts)} 个预报值 |",
             f"| 附件4 | {len(price_points)} 个电价数值完整 |",
             "| 日期、时间和重复记录 | 检查通过 |", "| 原始副本 | 与来源的SHA-256一致 |", "",
             "上述检查不代表数据已通过物理合理性审查，也不代表建模口径已确定。", "",
             "## 数据处理说明", "",
             f"- 附件3的 {filled} 个空白日期标签按同一天的第一条记录补齐，不是填补预测数值。",
             "- 00:10到24:00暂时保留为原始采样标签。没有自动决定它们代表前10分钟还是后10分钟。",
             "- 预报保留发布时间 issued_at、提前量 lead_hours 和目标时刻 valid_at。决策时只能使用 issued_at 不晚于当前时刻的预报。",
             "- 12月31日发布的跨年预报保留；没有为其虚构2026年实际值。",
             "- 没有进行插值、功率转电量、训练集划分或模型求解。", "",
             "## 结果模板疑点", "", "| 文件 | 首段标签 | 末段标签 |", "|---|---|---|"]
    lines += [f"| {r['file']} | {r['first_interval']} | {r['last_interval']} |" for r in template_notes]
    lines += ["", "这些标签与题面当天00:00到24:00的要求存在疑似错位。已保留原样，需要核实后再生成最终填表程序。", "",
              "## 与建模手确认", ""] + [f"- {x}。" for x in summary["unresolved"]]
    (ROOT / "reports/数据检查.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("baseline_points", "actual_points", "forecast_batches", "forecast_points", "checks")}, ensure_ascii=False, indent=2))
    print("输出位置：", ROOT)


if __name__ == "__main__":
    main()

from pathlib import Path
import argparse
import json

from forecasting import read_dataset
from reporting import build_payload
from run_pipeline import write_json, write_csv, ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    args=parser.parse_args()
    dataset=read_dataset(ROOT/"data/processed")
    config=json.loads((args.run/"config.json").read_text(encoding="utf-8"))["config"]
    payload=build_payload(args.run,dataset,config)
    write_json(args.run/"workbook_payload.json",payload)
    target=args.run/"deliverables"
    target.mkdir(exist_ok=True)
    write_csv(target/"第二问每日费用与电量.csv",payload["q2"]["daily"])
    write_json(target/"指定日期结果.json",payload["q2"]["selected_dates"])
    print(json.dumps({"payload":str(args.run/"workbook_payload.json"),"q2_dates":len(payload["q2"]["purchases"]),
                      "q2_battery_rows":len(payload["q2"]["battery"]),"emergency_segments":len(payload["q2"]["emergency"])},ensure_ascii=False))


if __name__=="__main__":
    main()

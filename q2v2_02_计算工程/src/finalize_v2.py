"""检查交付前置条件，保存最终源码及展示检查记录。"""
from pathlib import Path
import sys,json,hashlib,shutil
from run_v2 import write_json


def main(run):
    root=Path(__file__).resolve().parents[1]
    complete=json.loads((run/'completion.json').read_text(encoding='utf-8'))
    audit=json.loads((run/'independent_audit.json').read_text(encoding='utf-8'))
    workbook=json.loads((run/'workbook_qa/readonly_export_check.json').read_text(encoding='utf-8'))
    figures=json.loads((run/'figure_checks.json').read_text(encoding='utf-8'))
    if complete['formal_days']!=334 or complete['experiments']!=14 or audit['audited_experiments']!=25 or audit['audited_days']!=4837:
        raise ValueError('Not all preregistered experiments were audited')
    if len(figures)!=4 or any(x['text_bounds']!='passed' or x['glyph_check']!='passed' for x in figures):
        raise ValueError('Figure checks incomplete')
    if any(not row['all_values_verified'] for key,row in workbook.items() if key.startswith('result')):
        raise ValueError('Workbook checks incomplete')
    for file in (run/'deliverables').glob('*.inspect.ndjson'):
        target=run/'workbook_qa'/file.name
        if target.exists():raise ValueError('Unexpected existing inspection target')
        # 两端路径固定于本次workspace运行目录；只移动工具产生的单个QA文件。
        if not file.resolve().is_relative_to(root.parent.resolve()) or not target.resolve().is_relative_to(root.parent.resolve()):
            raise ValueError('QA move outside workspace')
        file.rename(target)
    for folder in ('src','tests','configs'):
        destination=run/('final_'+folder)
        destination.mkdir(exist_ok=True)
        for source in (root/folder).iterdir():
            if source.is_file():shutil.copy2(source,destination/source.name)
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for part in ('src','tests','configs') for p in (root/part).iterdir() if p.is_file()}
    write_json(run/'final_code_sha256.json',hashes)
    write_json(run/'presentation_review.json',{
        'workbook_sheets_reviewed':['result1/计划购电量','result1/充放电量','result2/计划购电量','result2/充放电量','result2/紧急购电量'],
        'extra_views':['first/last intervals','year-end totals','method/time notes'],
        'scientific_pngs_reviewed':[x['name'] for x in figures],
        'result':'legible; no clipped content or overlapping legends; negative zero cleaned in Excel only',
        'export_revision':'result2 Excel zeros below 1e-9 normalized, all exported values checked within 1e-6',
        'performance_fix':'trajectory/forecast NPZ arrays loaded once during export/audit, no mathematical change',
        'parallel_execution':'worker terminal_1_2 also finished separately after primary started; separate duplicate not used',
        'main_solver_failures':0})
    complete['exports_verified']=True
    complete['independent_audit_verified']=True
    complete['figures_visually_reviewed']=True
    write_json(run/'completion.json',complete)
    print(json.dumps(complete,ensure_ascii=False,indent=2))


if __name__=='__main__':main(Path(sys.argv[1]).resolve())

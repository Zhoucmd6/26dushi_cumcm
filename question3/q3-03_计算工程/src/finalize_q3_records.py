"""逐项核验输入/数值源码哈希和完整实验注册表，再封存本次真实记录。"""
import json,hashlib,shutil,sys
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
from run_q3 import ROOT,write_json
from audit_q3 import audit_run
from audit_q3_calibration import audit_calibration

run=Path(sys.argv[1]);j=lambda p:json.loads(p.read_text(encoding='utf-8'))
cfg=j(run/'run_config.json');registry=j(run/'experiment_registry.json')
for relative,expected in cfg['input_sha256'].items():
    if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:raise ValueError('Input changed: '+relative)
for name,expected in cfg['numerical_code_sha256'].items():
    if hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()!=expected:raise ValueError('Numerical source changed: '+name)
audit=audit_run(run)
calibration_audit=audit_calibration(run)
if set(audit)!=set(registry):raise ValueError('Incomplete or unexpected experiments')
routes={};attempts=[];completed=0
for name,policy in registry.items():
    folder=run/'experiments'/name
    if j(folder/'policy.json')!=policy:raise ValueError('Policy mismatch: '+name)
    updates=j(folder/'updates.json');completed+=len(updates)
    routes[name]=dict(Counter(x['solver_route'] for x in updates))
    for u in updates:
        if u['solver_route']!='milp':attempts.append({'experiment':name,'date':u['date'],'hour':u['decision_hour'],'route':u['solver_route'],'attempts':u['solver_attempts']})
test=j(run/'test_results.json')
if not test['successful']:raise ValueError('Tests failed')
book=j(run/'workbook_qa/openpyxl_readonly_check.json')
if abs(book['total_fee_yuan']-j(run/'experiments/main/summary.json')['total_fee_yuan'])>1e-5:raise ValueError('Workbook fee mismatch')
figures=j(run/'figure_qa.json')
if len(figures)!=7 or any(x['glyph_check']!='passed' or x['text_bounds']!='passed' for x in figures):raise ValueError('Figure QA failed')
record={'input_hash_checks':len(cfg['input_sha256']),'numerical_code_sha256':cfg['numerical_code_sha256'],
    'completed_annual_experiments':len(audit),'unique_annual_policies':len({json.dumps(p,sort_keys=True) for p in registry.values()}),
    'completed_rolling_solves':completed,'january_calibration_candidates':len(j(run/'selection.json')['all_candidates']),
    'january_risk_comparators':len(j(run/'january_risk_validation.json')),'solver_routes':routes,'solver_fallbacks':attempts,
    'unit_tests':test,'calibration_audit':calibration_audit,'max_audit_residual':max(x['max_residual'] for x in audit.values()),
    'workbook_total_fee_yuan':book['total_fee_yuan'],'figure_count':len(figures),'finalized_at_utc':datetime.now(timezone.utc).isoformat()}
write_json(run/'completion_record.json',record)
shutil.copytree(ROOT/'src',run/'final_code_snapshot',dirs_exist_ok=True)
shutil.copytree(ROOT/'tests',run/'final_tests',dirs_exist_ok=True)
files=[p for p in (run/'deliverables').rglob('*') if p.is_file()]
write_json(run/'deliverable_manifest.json',[{'path':str(p.relative_to(run/'deliverables')),'bytes':p.stat().st_size,
          'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files])
print(json.dumps({k:v for k,v in record.items() if k not in ('solver_routes','numerical_code_sha256','solver_fallbacks')},ensure_ascii=False))

"""保存本次真实测试数量、日志和结果。"""
from pathlib import Path
import sys,unittest,json
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[1];run=Path(sys.argv[1]);run.mkdir(parents=True,exist_ok=True)
suite=unittest.defaultTestLoader.discover(str(root/'tests'))
with (run/'unit_tests.log').open('w',encoding='utf-8') as log:
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
record={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'skipped':len(result.skipped),'successful':result.wasSuccessful(),'time_utc':datetime.now(timezone.utc).isoformat()}
(run/'test_results.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record))
if not result.wasSuccessful():sys.exit(1)

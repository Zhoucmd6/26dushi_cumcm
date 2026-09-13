"""Bounded independent processes; each experiment has its own output directory."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,json,os,subprocess,sys
from run_q4 import definitions

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='full_q4_20260913');p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];run=root/'outputs'/a.run
    subprocess.run([sys.executable,str(root/'src/run_q4.py'),'--run',a.run,'--selection-only'],check=True)
    selection=json.loads((run/'selection.json').read_text('utf-8'))
    env=os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):env[key]='1'
    env['PYTHONIOENCODING']='utf-8';env['PYTHONDONTWRITEBYTECODE']='1'
    def execute(name):
        with (run/(name+'.log')).open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,str(root/'src/run_q4.py'),'--run',a.run,'--experiment',name],
                           cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        return name
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for job in as_completed([pool.submit(execute,name) for name in definitions(selection)]):
            print('FINISHED',job.result(),flush=True)

if __name__=='__main__':main()

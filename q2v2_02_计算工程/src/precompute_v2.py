"""并行预计算互相独立的后段实验；仅把完整结果原子发布给主程序恢复使用。

避免两个进程同时写同一实验：工作目录与主运行分开；主目录已经存在时
保留独立工作副本，不写入该目录。数值输入和函数与主入口完全相同。
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from run_v2 import experiment, load_cache, ROOT
from forecasting import read_dataset
from forecasting_v2 import MODELS
from microgrid_core import Battery


def main(run, group):
    manifest=json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['code'].items():
        if hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Parallel worker numeric code does not match running snapshot')
    config=manifest['config']
    selected=json.loads((run/'selection.json').read_text(encoding='utf-8'))
    data=read_dataset(ROOT/'data/processed')
    if data.sources!=manifest['inputs']:raise ValueError('Input mismatch')
    model=selected['chosen_model'];mult=selected['chosen_terminal_multiplier']
    initial=selected['formal_common_initial_kwh']
    cache=load_cache(data,model,config,run/'forecasts',365)
    stage=(ROOT/'outputs'/(run.name+'_parallel')/group).resolve()
    days=list(range(31,365))
    if group=='seed_eta':
        jobs=[('seed_20260913',{'seed':20260913}),('seed_20260914',{'seed':20260914})]
    elif group=='sensitivity':
        jobs=[('residual_0_8',{'residual_scale':.8}),('residual_1_2',{'residual_scale':1.2}),
              ('terminal_0_8',{'terminal_multiplier':mult*.8}),('terminal_1_2',{'terminal_multiplier':mult*1.2})]
    else:raise ValueError('Unknown bounded worker group')
    workspace=ROOT.parent.resolve()

    def publish(name):
        source=(stage/name).resolve();target=(run/name).resolve()
        if not source.is_relative_to(workspace) or not target.is_relative_to(workspace):
            raise ValueError('Move must stay inside version2 workspace')
        if target.exists():
            print(f'{name}: primary already exists; retained separate worker copy',flush=True)
            return
        try:source.rename(target)
        except FileExistsError:
            print(f'{name}: primary started concurrently; worker copy retained',flush=True)
            return
        print(f'{name}: completed worker result published',flush=True)

    for name, kw in jobs:
        if (run/name/'summary.json').exists():continue
        args={'terminal_multiplier':mult,**kw}
        experiment(data,cache,config,stage/name,days,initial,final_target=6000.,**args)
        publish(name)
    if group=='seed_eta':
        alternate=Battery(np.sqrt(.9),np.sqrt(.9))
        p1=load_cache(data,MODELS[0],config,run/'forecasts',365)
        warm=experiment(data,p1,config,stage/'warmup_roundtrip90',range(31),6000.,battery=alternate)
        publish('warmup_roundtrip90')
        experiment(data,cache,config,stage/'roundtrip90',days,warm['terminal_kwh'],
                   battery=alternate,terminal_multiplier=mult,final_target=6000.)
        publish('roundtrip90')


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),sys.argv[2])

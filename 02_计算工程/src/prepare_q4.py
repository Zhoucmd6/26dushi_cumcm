from pathlib import Path
import json
import hashlib
import numpy as np
from q4_forecasting import read_inputs, select_price, supply_predictions, make_cache

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'outputs/prepared'
    folder.mkdir(parents=True,exist_ok=True)
    ds,prices,official=read_inputs(ROOT/'data/processed')
    selection=select_price(prices,ds.prices)
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')
                   if p.name in ('q4_forecasting.py','forecasting.py','forecasting_v2.py','q3_forecasting.py')}
    signature={'inputs':ds.sources,'prediction_sources':source_hashes,'seed':20260912}
    if (folder/'manifest.json').exists():
        if json.loads((folder/'manifest.json').read_text('utf-8'))!=signature:
            raise ValueError('Prediction source/input changed; use a new prepared directory')
    (folder/'manifest.json').write_text(json.dumps(signature,ensure_ascii=False,indent=2),encoding='utf-8')
    if (folder/'supply.npz').exists():
        with np.load(folder/'supply.npz') as z:supply={k:z[k] for k in z.files}
    else:
        supply=supply_predictions(ds,official,progress=lambda s:print(s,flush=True))
        np.savez_compressed(folder/'supply.npz',**supply)
    for key,warmup in [('formal',False),('warmup',True)]:
        cache=make_cache(ds,prices,supply,selection,warmup=warmup)
        np.savez_compressed(folder/f'{key}.npz',**cache)
    (folder/'price_selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(selection,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()

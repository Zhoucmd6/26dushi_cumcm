"""Verify the slim delivery without archived prediction caches or solver logs."""
from pathlib import Path
import csv,hashlib,io,json,zipfile
from forecasting import interval_label
from q4_forecasting import read_inputs

PROJECT=Path(__file__).resolve().parents[1]
PACKAGE=PROJECT.parent
RESULT=PACKAGE/'01_论文成果'
REVIEW=PACKAGE/'03_复核记录'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    manifest=json.loads((REVIEW/'manifest.json').read_text('utf-8'))
    prediction=json.loads((REVIEW/'预测来源manifest.json').read_text('utf-8'))
    for name,digest in manifest['inputs'].items():assert sha(PROJECT/'data/processed'/name)==digest,name
    for name,digest in {**manifest['numeric_code'],**prediction['prediction_sources']}.items():
        assert sha(PROJECT/'src'/name)==digest,name
    assert sha(PROJECT/'model/model-q4.tex')==manifest['model']
    saved=json.loads((REVIEW/'workbook_verification.json').read_text('utf-8'))
    for r in saved:assert sha(RESULT/r['file'])==r['sha256'],r['file']
    index=PACKAGE/'交付校验清单.json'
    if index.exists():
        for r in json.loads(index.read_text('utf-8'))['files']:assert sha(PACKAGE/r['path'])==r['sha256'],r['path']
    ds,prices,_=read_inputs(PROJECT/'data/processed')
    cases=[]
    with zipfile.ZipFile(RESULT/'10分钟完整轨迹.zip') as z:
        assert z.testzip() is None
        for mode in ('q42','q43'):
            maxima={};cash=0.;last=None;count=0
            def check(key,value):
                value=abs(float(value));maxima[key]=max(maxima.get(key,0.),value)
                assert value<2e-5,(mode,key,value)
            rows=csv.DictReader(io.TextIOWrapper(z.open(mode+'_10分钟完整轨迹.csv'),encoding='utf-8-sig'))
            for number,row in enumerate(rows):
                day,t=divmod(number,144);day+=31
                assert row['date']==str(ds.dates[day]) and row['interval']==interval_label(t)
                q,r,c,d,e,u,s,sn,p=[float(row[k]) for k in ('Q_kwh','R_kwh','C_kwh','D_kwh','E_kwh','U_kwh','S_start_kwh','S_end_kwh','price_yuan_per_kwh')]
                check('price',p-prices[day,t]);net=(ds.load_kw[day,t]-ds.pv_kw[day,t])/6
                check('state',sn-s-.9*c+d/.9)
                check('state_bounds',max(0,1200-s,1200-sn,s-10800,sn-10800))
                check('flow_bounds',max(0,c-5000/6,d-5000/6,-min(q,r,c,d,e,u)))
                check('exclusive',min(c,d))
                check('emergency',e-max(net+c-d-r,0));check('unused',u-max(r+d-c-net,0))
                check('continuity',s-(6000 if last is None else last));last=sn
                if mode=='q42' or t<36:check('fixed_commitment',q-r)
                normal=p*q if mode=='q42' else p*(q+1.5*max(r-q,0)-.5*max(q-r,0))
                actual=normal+5*p*e
                check('normal_bill',normal-float(row['normal_cost_yuan']))
                check('emergency_bill',5*p*e-float(row['emergency_cost_yuan']))
                check('actual_bill',actual-float(row['cash_cost_yuan']))
                cash+=actual;count+=1
            assert count==334*144
            check('year_end',max(6000-last,0))
            expected=next(r['cash_cost_yuan'] for r in saved if r['file']==('result4-2.xlsx' if mode=='q42' else 'result4-3.xlsx'))
            check('annual_bill',cash-expected)
            cases.append({'mode':mode,'intervals_checked':count,'cash_cost_yuan':expected,
                          'maximum_error':max(maxima.values()),'checks':maxima})
    result={'inputs_match':True,'numerical_sources_match':True,'workbooks_unchanged':True,
            'csv_zip_crc':'passed','executed_trajectories':cases,'tolerance':2e-5}
    (REVIEW/'精简验证.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

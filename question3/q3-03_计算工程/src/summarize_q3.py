"""汇总第三问对照、预报增量价值、概率检验和论文图表。"""
from pathlib import Path
import argparse,json,hashlib,platform
import numpy as np
import pandas as pd
import scipy,matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from forecasting import read_dataset
from run_q3 import ROOT,write_json,write_csv
from audit_q3 import audit_run
from scientific_style import apply_style,plt,COLORS,panel,time_axis,step,note,save_figure

def label(name):
    return {'main':'B3 条件尺度（风险中性）','point_official':'B1 原始预报点预测','scale_pooled':'B2 统一尺度','scale_exponential':'指数尺度','scale_binned':'分箱尺度',
            'cvar90':'CVaR 90%, λ=0.1','cvar95':'CVaR 95%, λ=0.1','terminal_zero':'无日末软目标',
            'terminal_hard':'每日硬目标6000','terminal_soft':'正软终端目标',
            'cvar90_lambda005':'CVaR 90%, λ=0.05','cvar95_lambda005':'CVaR 95%, λ=0.05',
            'hourly_mean':'小时均值解释','efficiency_roundtrip90':'往返效率90%',
            'probability_selected':'概率评分选参','global_cash_selected':'全候选费用选参',
            'settlement_alternative':'备选结算：下调另罚'}.get(name,name)

def mask_label(mask):
    selected=[str(h) for j,h in enumerate([6,12,18]) if mask&(1<<j)]
    return '0'+(' + '+', '.join(selected) if selected else '')+' 时'

def collect(run):
    summaries=[]
    for p in sorted((run/'experiments').glob('*/summary.json')):
        item=json.loads(p.read_text(encoding='utf-8'));policy=item.pop('policy')
        summaries.append({'name':p.parent.name,**item,**policy})
    write_json(run/'experiment_summary.json',summaries)
    out=run/'deliverables';out.mkdir(exist_ok=True)
    fields=['name','mode','gamma','total_fee_yuan','normal_fee_yuan','emergency_fee_yuan','emergency_kwh',
            'daily_emergency_cvar90_yuan','daily_emergency_cvar95_yuan','unused_kwh','initial_kwh','terminal_kwh','runtime_s',
            'adjust_up_kwh','adjust_down_kwh','adjusted_intervals','throughput_kwh','solver_runtime_total_s','max_solver_gap_yuan']
    write_csv(out/'第三问全部方案对比.csv',[{k:r[k] for k in fields} for r in summaries])
    records={r['name']:r for r in summaries}
    expected=set(json.loads((run/'experiment_registry.json').read_text(encoding='utf-8')))
    if not expected.issubset(records):raise RuntimeError('Unfinished experiments: '+str(sorted(expected-set(records))))
    rows=[];maskdaily={}
    for mask in range(8):
        name='main' if mask==7 else f'info_mask{mask}'
        maskdaily[mask]=pd.read_csv(run/'experiments'/name/'daily.csv')['total_fee_yuan'].to_numpy()
    # 配对7日移动块bootstrap；保留日差相关性，区间仅作这一个年份的探索性描述。
    rng=np.random.default_rng(20260912);n=334;block=7
    starts=rng.integers(0,n-block+1,size=(2000,int(np.ceil(n/block))))
    sample=(starts[:,:,None]+np.arange(block)[None,None,:]).reshape(2000,-1)[:,:n]
    for mask in range(8):
        for j,hour in enumerate([6,12,18]):
            if mask&(1<<j):continue
            dest=mask|(1<<j);delta=maskdaily[mask]-maskdaily[dest];sample_means=delta[sample].mean(axis=1)
            lo,hi=np.quantile(sample_means,[.025,.975])
            rows.append({'base_mask':mask,'added_hour':hour,'new_mask':dest,'total_saving_yuan':float(delta.sum()),
                         'mean_daily_saving_yuan':float(delta.mean()),'mean_daily_ci_low_yuan':float(lo),
                         'mean_daily_ci_high_yuan':float(hi),'days_with_saving':int((delta>1e-6).sum()),
                         'bootstrap_block_days':7,'bootstrap_draws':2000})
    write_csv(out/'预报时刻边际价值.csv',rows)
    write_csv(out/'预报时刻组合对比.csv',[{'mask':m,'forecast_hours':mask_label(m),'total_fee_yuan':float(maskdaily[m].sum()),
                                         'saving_vs_midnight_only_yuan':float((maskdaily[0]-maskdaily[m]).sum())} for m in range(8)])
    return records,rows,maskdaily

def figures(run,records,marginal,maskdaily):
    out=run/'deliverables/科研图表';out.mkdir(parents=True,exist_ok=True);apply_style();qa=[]
    a=dict(np.load(run/'experiments/main/dispatch.npz'));ds=read_dataset(ROOT/'data/processed')
    daily=pd.read_csv(run/'experiments/main/daily.csv');scores=pd.read_csv(run/'annual_probability_scores.csv')
    with PdfPages(out/'第三问科研图表.pdf') as pdf:
        fig,axs=plt.subplots(4,2,figsize=(9,10));fig.subplots_adjust(left=.10,right=.96,top=.925,bottom=.09,hspace=.58,wspace=.33)
        for j,date in enumerate(['2025-03-20','2025-06-21','2025-09-23','2025-12-21']):
            i=list(daily.date).index(date);ax=axs[j,0]
            step(ax,a['Q'][i],color=COLORS['muted'],ls='--',label='0点计划 Q')
            step(ax,a['R'][i],color=COLORS['purchase'],label='最终正常购电 R')
            step(ax,a['E'][i],color=COLORS['emergency'],lw=1,label='紧急购电 E')
            panel(ax,chr(97+2*j),date+'：购电');ax.set_ylabel('电量 (kWh / 10 min)');time_axis(ax)
            ax=axs[j,1];ax.plot(np.arange(145)/6,a['S'][i],color=COLORS['state'],label='电池内部储量 S')
            ax.axhline(1200,color=COLORS['rule'],lw=.7,ls=':');ax.axhline(10800,color=COLORS['rule'],lw=.7,ls=':')
            for h in [6,12,18]:ax.axvline(h,color=COLORS['rule'],lw=.6,ls='--')
            ax.set_ylim(0,12000);ax.set_ylabel('储电量 (kWh)');time_axis(ax);panel(ax,chr(98+2*j),date+'：储能')
        handles,names=axs[0,0].get_legend_handles_labels();h2,n2=axs[0,1].get_legend_handles_labels()
        fig.legend(handles+h2,names+n2,ncol=4,fontsize=8,loc='upper center',bbox_to_anchor=(.53,.993))
        note(fig,'主方案：B3条件尺度、风险中性随机滚动调度。每6小时执行新计划；电量按10分钟区间展示，未作平滑。')
        qa.append(save_figure(fig,out,'01_指定日期购电与储能',pdf=pdf))

        fig,axs=plt.subplots(1,2,figsize=(9,3.8));fig.subplots_adjust(left=.09,right=.97,top=.85,bottom=.24,wspace=.36)
        monthly=daily.groupby(pd.to_datetime(daily.date).dt.month)[['normal_fee_yuan','emergency_fee_yuan']].sum()/1e4
        ax=axs[0];ax.bar(monthly.index,monthly.normal_fee_yuan,color=COLORS['planned_cost'],width=.66,label='正常结算')
        ax.bar(monthly.index,monthly.emergency_fee_yuan,bottom=monthly.normal_fee_yuan,color=COLORS['emergency'],width=.66,label='紧急费用')
        ax.set_xticks(monthly.index);ax.set_xlabel('月份');ax.set_ylabel('费用 (万元)');ax.legend(ncol=2,fontsize=7);panel(ax,'a','各月实际购电费用')
        ax=axs[1]
        for name,color,marker in [('main',COLORS['purchase'],'o'),('cvar90',COLORS['pv'],'s'),('cvar95',COLORS['state'],'D')]:
            item=records[name];x=item['total_fee_yuan']/1e4;y=item['daily_emergency_cvar95_yuan']/1e4
            ax.scatter(x,y,color=color,marker=marker,s=48,label=label(name))
        ax.set_xlabel('2—12月总费用 (万元)');ax.set_ylabel('日紧急费用 CVaR95 (万元)');ax.ticklabel_format(useOffset=False);ax.legend(fontsize=7,loc='best');panel(ax,'b','实际费用与尾部风险')
        note(fig,'右图按334个已实现的日紧急费用重算CVaR95。该指标不等于各次优化目标内CVaR之和。',y=.025)
        qa.append(save_figure(fig,out,'02_月度费用与风险权衡',pdf=pdf))

        fig,axs=plt.subplots(1,2,figsize=(9,4.3));fig.subplots_adjust(left=.17,right=.97,top=.85,bottom=.25,wspace=.52)
        ax=axs[0];values=np.array([maskdaily[m].sum() for m in range(8)])/1e4
        ax.barh(np.arange(8),values,color=[COLORS['purchase'] if m==7 else COLORS['planned_cost'] for m in range(8)],height=.64)
        ax.set_yticks(np.arange(8),[mask_label(m) for m in range(8)]);ax.invert_yaxis();ax.set_xlabel('2—12月总费用 (万元)');ax.set_xlim(0,values.max()*1.18)
        for j,v in enumerate(values):ax.text(v+4,j,f'{v:.2f}',va='center',fontsize=7)
        panel(ax,'a','可用预报时刻的8种组合')
        ax=axs[1];chosen=[next(r for r in marginal if r['new_mask']==7 and r['added_hour']==h) for h in [6,12,18]]
        for j,row in enumerate(chosen):
            mean=row['mean_daily_saving_yuan'];lo=row['mean_daily_ci_low_yuan'];hi=row['mean_daily_ci_high_yuan']
            ax.errorbar(mean,j,xerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o',color=[COLORS['pv'],COLORS['purchase'],COLORS['state']][j],capsize=3)
        ax.axvline(0,color=COLORS['rule'],lw=.8);ax.set_yticks(range(3),['增加6点预报','增加12点预报','增加18点预报']);ax.invert_yaxis();ax.set_xlabel('平均每日节省费用 (元)');panel(ax,'b','保留其余预报时的增量价值')
        note(fig,'各组合均每天重新优化4次，仅屏蔽新预报。区间为配对7日移动块bootstrap 95%区间（2000次），仅作探索性描述。')
        qa.append(save_figure(fig,out,'03_预报信息增量价值',pdf=pdf))

        fig,axs=plt.subplots(1,2,figsize=(9,3.8));fig.subplots_adjust(left=.09,right=.97,top=.85,bottom=.25,wspace=.35)
        for mode,color,marker,text in [('pooled',COLORS['purchase'],'o','合并尺度'),('exponential',COLORS['state'],'s','指数尺度'),('binned',COLORS['pv'],'D','分箱尺度')]:
            frame=scores[(scores['mode']==mode)&(scores['subset']=='daylight')].sort_values('issue_hour')
            axs[0].plot(frame.issue_hour,frame.crps,marker=marker,color=color,label=text)
            axs[1].plot(frame.issue_hour,frame.coverage*100,marker=marker,color=color,label=text)
        for ax in axs:ax.set_xticks([0,6,12,18]);ax.set_xlabel('预报发布时间 (h)')
        axs[0].set_ylabel('CRPS (kW)，越低越好');axs[0].legend(fontsize=7);panel(axs[0],'a','实际光伏为正时的概率评分')
        axs[1].axhline(90,color=COLORS['muted'],ls='--',lw=.8,label='标称90%');axs[1].set_ylabel('90%区间实际覆盖率 (%)');axs[1].set_ylim(50,100);axs[1].legend(fontsize=7);panel(axs[1],'b','样本外区间覆盖检验')
        note(fig,'每次只评价当天剩余时域；不同发布时间对应的目标时段不同，不能将横轴趋势直接解释为提前量的因果效果。')
        qa.append(save_figure(fig,out,'04_概率预测样本外检验',pdf=pdf))
        from q3_extra_figures import extra_figures
        qa.extend(extra_figures(run,records,out,pdf))
    write_json(run/'figure_qa.json',qa)

from q3_report import report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');parser.add_argument('--no-figures',action='store_true');args=parser.parse_args()
    run=Path(args.run);records,marginal,maskdaily=collect(run);report(run,records,marginal,maskdaily)
    if not args.no_figures:figures(run,records,marginal,maskdaily)

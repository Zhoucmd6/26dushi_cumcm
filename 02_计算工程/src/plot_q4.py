"""Publication figures from audited data; 450 dpi PNG, editable SVG and PDF."""
from pathlib import Path
import argparse,csv,json
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from scientific_style import apply_style,save_figure,step,time_axis
from q4_forecasting import read_inputs,scenarios
from run_q4 import ROOT,load_cache,write_json,write_csv
from structural_checks import DATES

C2='#0072B2';C3='#009E73';RED='#D55E00';PURPLE='#8C6BB1';GRAY='#63717D'
LABELS={'q42_point':'点预测','q42_pair':'成对场景 ρ=1','q42_independent':'独立边际 ρ=0（主方案）',
        'q42_gamma0':'主方案 γ=0','q43_point':'点预测','q43_no_price_correction':'不修正日内电价',
        'q43_common':'共同动作 K=1（主方案）','q43_lookahead_k2':'分组前瞻 K=2',
        'q43_lookahead_k3':'分组前瞻 K=3','q43_gamma0':'主方案 γ=0'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='full_q4_20260913');arg=parser.parse_args()
    run=ROOT/'outputs'/arg.run;out=run/'deliverables/论文用图';out.mkdir(parents=True,exist_ok=True)
    selection=json.loads((run/'selection.json').read_text('utf-8'))
    summaries={p.parent.name:json.loads(p.read_text('utf-8')) for p in run.glob('q4*/summary.json')}
    daily={name:list(csv.DictReader((run/name/'daily.csv').open(encoding='utf-8-sig'))) for name in summaries}
    ds,prices,_=read_inputs(ROOT/'data/processed');cache=load_cache('formal');qa=[];captions=[]
    arrays={}
    for mode in ('q42','q43'):
        with np.load(run/selection[mode+'_main']/'trajectory.npz') as z:arrays[mode]={k:z[k] for k in z.files}
    months=np.array([d.month for d in ds.dates[31:]]);x=np.arange(2,13)
    def values(name,key):return np.array([float(r[key]) for r in daily[name]])
    def axstyle(ax,title,ylabel=None):
        ax.set_title(title,loc='left',pad=10);ax.grid(axis='y',color='#DDE3E8',lw=.5,zorder=0)
        ax.set_axisbelow(True)
        if ylabel:ax.set_ylabel(ylabel)
    def header(fig,title,subtitle):
        fig.text(.09,.974,title,fontsize=13,fontweight='bold',va='top')
        fig.text(.09,.974-23/(fig.get_figheight()*72),subtitle,fontsize=8,color=GRAY,va='top')
    def save(fig,name,caption,pdf):
        qa.append(save_figure(fig,out,name,pdf=pdf));captions.append((name,caption))
    apply_style()
    with PdfPages(out/'第四问图集.pdf') as pdf:
        # 1. Price information, units and uncertainty.
        fig,axs=plt.subplots(1,2,figsize=(8.1,3.8));fig.subplots_adjust(left=.09,right=.975,bottom=.18,top=.78,wspace=.3)
        header(fig,'电价预测与信息更新','2025年2—12月评价；右侧按已执行时段汇总，不重复计算滚动预测的重叠部分。')
        day=[str(d) for d in ds.dates].index('2025-06-21');row=day-31
        step(axs[0],prices[day],color='#242C33',label='实际电价',linewidth=1.)
        step(axs[0],arrays['q42']['forecast_price'][row],color=C2,linestyle='--',label='零点场景均价')
        step(axs[0],arrays['q43']['forecast_price'][row],color=C3,label='滚动场景均价')
        lo=[];hi=[]
        for issue in range(4):
            _,p,prob,_,_=scenarios(ds,prices,cache,day,issue,mode='q43')
            lo.extend(np.quantile(p[:,:36],.1,axis=0,method='inverted_cdf'));hi.extend(np.quantile(p[:,:36],.9,axis=0,method='inverted_cdf'))
        axs[0].fill_between(np.arange(145)/6,np.r_[lo,lo[-1]],np.r_[hi,hi[-1]],step='post',color=C3,alpha=.12,label='滚动场景80%区间')
        for h in (6,12,18):axs[0].axvline(h,color=GRAY,lw=.6,ls=':')
        time_axis(axs[0]);axstyle(axs[0],'a  6月21日：日内更新','电价（元/kWh）');axs[0].legend(fontsize=6.8,loc='upper left')
        for mode,color,label in [('q42',C2,'零点预测'),('q43',C3,'滚动预测')]:
            error=np.abs(arrays[mode]['forecast_price']-prices[31:]).mean(axis=1)
            axs[1].plot(x,[error[months==m].mean() for m in x],marker='o' if mode=='q42' else 's',ms=3,color=color,label=label)
        axstyle(axs[1],'b  各月已执行时段预测误差','电价 MAE（元/kWh）');axs[1].set_xticks(x);axs[1].set_xlabel('月份');axs[1].legend()
        save(fig,'01_电价预测与信息更新','实线为实际价格与滚动场景均价，虚线为零点场景均价；淡绿色为历史残差场景的10%—90%区间，不能解释为已校准的概率保证。',pdf)
        # 2. Annual total bills, zero-baseline bars and exact stacked components.
        fig,axs=plt.subplots(1,2,figsize=(9.0,4.4));fig.subplots_adjust(left=.23,right=.975,bottom=.16,top=.79,wspace=1.0)
        header(fig,'全年实际购电费对照','评价期为2月1日至12月31日；1月已冻结主方案，全年对照结果不用于倒选参数。')
        for ax,mode,names in [(axs[0],'Q4-2',['q42_point','q42_pair','q42_independent','q42_gamma0']),
                            (axs[1],'Q4-3',['q43_point','q43_no_price_correction','q43_common','q43_lookahead_k2','q43_lookahead_k3','q43_gamma0'])]:
            normal=np.array([summaries[n]['planned_cost_yuan']+summaries[n]['adjustment_cost_yuan'] for n in names])/1e4
            emerg=np.array([summaries[n]['emergency_cost_yuan'] for n in names])/1e4;y=np.arange(len(names))
            ax.barh(y,normal,color=C2 if mode=='Q4-2' else C3,height=.6,label='正常结算')
            ax.barh(y,emerg,left=normal,color=RED,alpha=.8,height=.6,label='紧急购电')
            for j,v in enumerate(normal+emerg):ax.text(v+18,j,f'{v:,.2f}',va='center',fontsize=7)
            ax.set_yticks(y,[LABELS[n] for n in names],fontsize=7);ax.invert_yaxis();ax.set_xlim(0,1930)
            axstyle(ax,mode);ax.grid(False);ax.set_xlabel('实际总费用（万元）');ax.xaxis.set_major_locator(MaxNLocator(4));ax.legend(loc='lower left',bbox_to_anchor=(-.1,-.25),ncol=2,fontsize=7)
        save(fig,'02_全年实际费用对照','所有条形从零开始。正常结算包含相对原计划的上调费用与下调退款，紧急购电单列；预测目标中的终端惩罚不计入实际账单。',pdf)
        # 3. Seasonal behavior and emergency amount.
        fig,axs=plt.subplots(2,1,figsize=(7.5,5.6),sharex=True);fig.subplots_adjust(left=.12,right=.97,bottom=.13,top=.83,hspace=.4)
        header(fig,'月度费用与紧急购电','两条主方案均从1月1日6000 kWh开始热身，正式评价持续继承真实库存。')
        monthly=[]
        for mode,color,offset in [('q42',C2,-.18),('q43',C3,.18)]:
            name=selection[mode+'_main']
            for ax,key,scale in [(axs[0],'cash_cost_yuan',1e4),(axs[1],'emergency_kwh',1e3)]:
                val=values(name,key);totals=np.array([val[months==m].sum()/scale for m in x])
                ax.bar(x+offset,totals,width=.34,color=color,label='Q4-'+mode[-1])
            for m in x:monthly.append({'mode':mode,'month':m,'cash_cost_yuan':float(values(name,'cash_cost_yuan')[months==m].sum()),'emergency_kwh':float(values(name,'emergency_kwh')[months==m].sum())})
        axstyle(axs[0],'a  每月实际购电费','费用（万元）');axstyle(axs[1],'b  每月紧急购电量','电量（MWh）')
        axs[0].legend(ncol=2,loc='upper left');axs[1].set_xticks(x);axs[1].set_xlabel('月份')
        write_csv(out/'月度图源数据.csv',monthly)
        save(fig,'03_月度费用与紧急购电','费用与紧急电量分别使用独立纵轴；用于区分季节性购电规模和预测失配带来的补电需求。',pdf)
        # 4. Frozen historical choices versus later out-of-sample evidence.
        fig,axs=plt.subplots(2,2,figsize=(7.8,6.1));fig.subplots_adjust(left=.12,right=.97,bottom=.12,top=.82,hspace=.55,wspace=.35)
        header(fig,'参数选择与样本外检验','上排和左下仅使用1月21—31日账单；右下展示2—12月结果，不改变2月前已冻结的选择。')
        for mode,color in [('q42',C2),('q43',C3)]:
            rows=[r for r in selection['validation'] if r['name'].startswith('validate_'+mode+'_gamma')]
            g=[r['gamma'] for r in rows];v=np.array([r['cash_cost_yuan'] for r in rows])/1e4
            axs[0,0].plot(g,v,marker='o',ms=4,color=color,label='Q4-'+mode[-1])
        axstyle(axs[0,0],'a  终端软惩罚','1月验证费用（万元）');axs[0,0].set_xlabel('γ（元/kWh）');axs[0,0].legend()
        rr=[r for r in selection['validation'] if r['name'].startswith('validate_q42_rho')];v=np.array([r['cash_cost_yuan'] for r in rr]);xx=[r['rho'] for r in rr]
        axs[0,1].plot(xx,v-v.min(),color=C2,marker='o',ms=4);axs[0,1].set_xticks(xx);axs[0,1].set_xlabel('依赖收缩参数 ρ')
        axstyle(axs[0,1],'b  Q4-2：历史选择 ρ=0','相对验证最低费用（元）')
        rr=[r for r in selection['validation'] if r['name'].startswith('validate_q43_k')];v=np.array([r['cash_cost_yuan'] for r in rr]);xx=[r['k'] for r in rr]
        axs[1,0].bar(xx,v-v.min(),color=[C3,GRAY,GRAY],width=.55);axs[1,0].set_xticks(xx);axs[1,0].set_xlabel('候选信息组数 K')
        axstyle(axs[1,0],'c  Q4-3：历史选择 K=1','相对验证最低费用（元）')
        baseline=summaries['q43_common']['cash_cost_yuan'];diff=[summaries[n]['cash_cost_yuan']-baseline for n in ['q43_common','q43_lookahead_k2','q43_lookahead_k3']]
        axs[1,1].bar([1,2,3],diff,color=[C3,PURPLE,PURPLE],width=.55);axs[1,1].axhline(0,color=GRAY,lw=.7)
        axs[1,1].set_xticks([1,2,3]);axs[1,1].set_xlabel('候选信息组数 K');axstyle(axs[1,1],'d  全年分组前瞻对照','相对 K=1 的费用变化（元）')
        save(fig,'04_参数选择与样本外检验','主方案由1月历史验证冻结为γ=0.5、Q4-2的ρ=0及Q4-3的K=1。全年K=2、3略有改善，但不能据此回头更改年初参数。K表示请求组数，样本不足时按模型回退为1。',pdf)
        # 5. A recorded nonanticipative branch plan, distinct from actual execution.
        logs=json.loads((run/'q43_lookahead_k2/planning_log.json').read_text('utf-8'))
        eligible=[r for r in logs if r['day']=='2025-06-21' and len(r['branch_nodes'])>2]
        record=eligible[0] if eligible else next(r for r in logs if len(r['branch_nodes'])>2)
        day=[str(d) for d in ds.dates].index(record['day']);hour=record['hour'];group=record['grouping'];nodes=record['branch_nodes'];root=nodes[0]
        fig,axs=plt.subplots(1,2,figsize=(8,4.1));fig.subplots_adjust(left=.12,right=.97,bottom=.19,top=.79,wspace=.33)
        header(fig,'一次更新的分组前瞻结构',f"示例：{record['day']} {hour:02d}:00；实线为共同前缀，虚线为当时的条件尾部计划。")
        sig=np.array(group['signal_scenarios']);colors=[C2,PURPLE]
        for j,node in enumerate(nodes[1:]):
            ids=np.array(node['ids']);axs[0].scatter(sig[ids,0]/1000,sig[ids,1],s=22,color=colors[j],alpha=.8,label=f'组{j+1}，{len(ids)}个历史日')
            hours=hour+np.arange(node['start'],node['stop']+1)/6
            axs[1].plot(hours,np.array(node['S'])/1000,ls='--',color=colors[j],label=f'组{j+1}条件计划')
        axs[0].axhline(0,color=GRAY,lw=.5);axs[0].axvline(0,color=GRAY,lw=.5)
        axstyle(axs[0],'a  过去整日样本配对的信息信号','电价对数误差信号');axs[0].set_xlabel('下一次光伏预报修订量（MWh）');axs[0].legend(fontsize=7)
        axs[1].plot(hour+np.arange(len(root['S']))/6,np.array(root['S'])/1000,color='#242C33',lw=1.8,label='共同前缀')
        bridge=hour+6;axs[1].axvline(bridge,color=GRAY,lw=.8,ls=':');axs[1].scatter([bridge],[root['S'][-1]/1000],s=35,color='#242C33',zorder=5)
        for value in (1.2,10.8):axs[1].axhline(value,color=GRAY,lw=.6,ls=':')
        axs[1].set_ylim(.5,11.5);axs[1].set_xlim(hour,24);axs[1].set_xticks(np.arange(hour,25,6));axs[1].set_xlabel('时刻（h）')
        axstyle(axs[1],'b  各尾部共享同一桥接库存','电池内部储量（MWh）');axs[1].legend(fontsize=7,loc='best')
        write_json(out/'前瞻示例图源数据.json',record)
        save(fig,'05_分组信号与桥接库存','分组同时使用光伏预报修订和电价已结束时段误差信号。所有未来分支从同一共同前缀末库存出发；只执行实线前缀，下一次更新时重新求解，虚线不是实际轨迹。示例优先选6月21日首个有效双分组时刻。',pdf)
        # 6–7. Four required dates, exact intervals and separate energy/power units.
        for number,mode,color in [(6,'q42',C2),(7,'q43',C3)]:
            a=arrays[mode];fig,axs=plt.subplots(4,2,figsize=(7.8,9.5));fig.subplots_adjust(left=.10,right=.97,bottom=.08,top=.86,hspace=.60,wspace=.29)
            header(fig,'Q4-'+mode[-1]+'：题目指定四日调度','左列为交流母线侧平均功率，右列为电池内部储量；所有曲线保留原始10分钟分辨率。')
            handles=None
            for j,daytext in enumerate(DATES):
                day=[str(d) for d in ds.dates].index(daytext);ix=day-31;left,right=axs[j]
                net=(ds.load_kw[day]-ds.pv_kw[day])/1000
                step(left,net,color=GRAY,lw=1.,label='实际净负荷')
                if mode=='q43':step(left,a['Q'][ix]*6/1000,color=C2,ls='--',lw=.9,label='零点原计划')
                step(left,a['R'][ix]*6/1000,color=color,lw=1.2,label='最终正常购电')
                left.fill_between(np.arange(145)/6,np.r_[a['E'][ix],a['E'][ix,-1]]*6/1000,step='post',color=RED,alpha=.30,label='紧急购电')
                left.axhline(0,color=GRAY,lw=.5);axstyle(left,daytext+'  购电','平均功率（MW）')
                right.plot(np.arange(145)/6,a['S'][ix]/1000,color=PURPLE,lw=1.3)
                for value in (1.2,10.8):right.axhline(value,color=GRAY,lw=.6,ls=':')
                if mode=='q43':
                    for hour in (6,12,18):right.axvline(hour,color=GRAY,lw=.5,ls=':')
                right.set_ylim(.5,11.5);right.set_yticks([1.2,6.,10.8]);axstyle(right,'库存轨迹','储量（MWh）')
                cash=values(selection[mode+'_main'],'cash_cost_yuan')[ix]
                right.set_title(f'库存轨迹；日总费 {cash/1e4:.3f} 万元',loc='left',pad=10,fontsize=8.5)
                for ax in (left,right):time_axis(ax,label=j==3,ticks=6)
                if j==0:handles=left.get_legend_handles_labels()
            fig.legend(*handles,loc='upper left',bbox_to_anchor=(.09,.911),ncol=4,fontsize=7)
            save(fig,f'{number:02d}_{mode.upper()}指定日期购电与储能','题目指定3月20日、6月21日、9月23日、12月21日。左列电量除以1/6小时并换算成MW；负净负荷表示光伏超过负荷。紧急补电与多余能量由实际净负荷结算，不直接改变预先决定的电池路径。',pdf)
    write_json(out/'图表校验.json',qa)
    (out/'图注与使用顺序.md').write_text('# 第四问图表使用说明\n\n建议按照预测信息、实际账单、季节分解、参数选择、前瞻结构、指定日期的顺序组织论文。PNG为450 dpi，SVG保留可编辑文字，PDF为同版图集。\n\n'+'\n\n'.join(f'## {name}\n\n{caption}' for name,caption in captions)+'\n',encoding='utf-8')
    print('Created',len(qa),'figures',flush=True)

if __name__=='__main__':main()

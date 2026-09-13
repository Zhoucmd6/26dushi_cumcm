"""从已保存的真实结果生成科研图表。仅变更展示，不运行预测、求解或调参。"""
from pathlib import Path
from datetime import date
import argparse
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from forecasting import read_dataset
from scientific_style import (COLORS as C, METHOD_COLORS, STYLE_VERSION,
                              apply_style, step, time_axis, panel, note, save_figure)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['main_weighted28', 'comparison_mean7', 'comparison_point', 'comparison_no_storage']
METHOD_LABELS = ['加权历史 + 场景规划', '7日均值 + 场景规划',
                 '加权历史 + 单一预测', '同场景 + 无储能']


def plot_q1(ds, result):
    a = result['plan']
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.8), sharex=True,
                             gridspec_kw={'height_ratios': [1.5, 1, 1, .8]})
    fig.subplots_adjust(left=.12, right=.97, top=.94, bottom=.09, hspace=.48)
    panel(axes[0], 'a', '典型日负载、光伏与计划购电')
    for values, color, label, linestyle in (
        (ds.baseline_load_kw/1000, C['load'], '负载', '-'),
        (ds.baseline_pv_kw/1000, C['pv'], '光伏', '--'),
        (np.asarray(a['Q'])*.006, C['purchase'], '计划购电', '-.')):
        step(axes[0], values, color=color, label=label, ls=linestyle)
    axes[0].set_ylabel('功率 (MW)')
    axes[0].set_ylim(0, max(ds.baseline_load_kw.max()/1000,
                          ds.baseline_pv_kw.max()/1000, max(a['Q'])*.006)*1.28)
    axes[0].legend(ncol=3, loc='upper left')
    panel(axes[1], 'b', '储能充放电：正值充电，负值放电')
    t = np.arange(144)/6
    axes[1].bar(t, np.asarray(a['C'])*.006, width=1/6, align='edge',
                color=C['charge'], linewidth=0, label='充电')
    axes[1].bar(t, -np.asarray(a['D'])*.006, width=1/6, align='edge',
                color=C['discharge'], linewidth=0, label='放电')
    axes[1].axhline(0, color=C['ink'], lw=.6)
    axes[1].set(ylabel='功率 (MW)', ylim=(-6.1, 7.5), yticks=[-5, 0, 5])
    axes[1].legend(ncol=2, loc='upper left', borderaxespad=.1)
    panel(axes[2], 'c', '电池储量与运行边界 (初末均为6 MWh)')
    storage = np.asarray(a['S'])/1000
    axes[2].fill_between(np.arange(145)/6, 1.2, storage, color=C['state'], alpha=.13)
    axes[2].plot(np.arange(145)/6, storage, color=C['state'], lw=1.4)
    for value in [1.2, 10.8]:
        axes[2].axhline(value, color=C['rule'], lw=.75, ls=(0,(4,3)))
    axes[2].plot([0,24], storage[[0,-1]], 'o', ms=3.5, color=C['state'], clip_on=False)
    axes[2].set(ylabel='储量 (MWh)', ylim=(0,12), yticks=[1.2,6,10.8])
    panel(axes[3], 'd', '已知电价')
    step(axes[3], ds.prices, color=C['price'], lw=1.4)
    axes[3].fill_between(np.arange(145)/6, 0, np.r_[ds.prices,ds.prices[-1]],
                         step='post', color=C['price'], alpha=.09)
    axes[3].set(ylabel='电价 (元/kWh)', ylim=(0,float(ds.prices.max())*1.2))
    for ax in axes:
        time_axis(ax, label=ax is axes[-1])
    note(fig, f'全天电费 {result["cost_yuan"]:,.2f} 元；每步10 min；充放电在母线侧计量。')
    return fig


def plot_methods(by_name):
    planned = np.array([by_name[n]['total_planned_cost_yuan'] for n in METHODS])/1e4
    emergency = np.array([by_name[n]['total_emergency_cost_yuan'] for n in METHODS])/1e4
    energy = np.array([by_name[n]['total_emergency_purchase_kwh'] for n in METHODS])/1000
    totals = planned+emergency
    fig, axes = plt.subplots(2,1,figsize=(7.2,6.0))
    fig.subplots_adjust(left=.26,right=.96,top=.91,bottom=.17,hspace=.62)
    y=np.arange(4)
    panel(axes[0],'a','实际购电费及其构成')
    axes[0].axhspan(-.43,.43,color=C['purchase'],alpha=.07,lw=0)
    axes[0].barh(y,planned,height=.56,color=C['planned_cost'],label='计划购电费')
    axes[0].barh(y,emergency,left=planned,height=.56,color=C['emergency'],label='紧急购电费')
    for i,total in enumerate(totals):
        axes[0].text(total+28,i,f'{total:,.2f}',va='center',fontsize=8,
                     fontweight='bold' if i==0 else 'normal')
    axes[0].legend(loc='upper left',ncol=2,borderaxespad=.3)
    axes[0].set(xlim=(0,totals.max()*1.19),xlabel='334天实际购电费 (万元)')
    panel(axes[1],'b','实际紧急购电量')
    axes[1].axhspan(-.43,.43,color=C['purchase'],alpha=.07,lw=0)
    axes[1].barh(y,energy,height=.56,color=METHOD_COLORS)
    for i,total in enumerate(energy):
        axes[1].text(total+32,i,f'{total:,.2f}',va='center',fontsize=8)
    axes[1].set(xlim=(0,energy.max()*1.19),xlabel='334天紧急购电量 (MWh)')
    for ax in axes:
        ax.set_yticks(y,METHOD_LABELS);ax.invert_yaxis()
        ax.get_yticklabels()[0].set_fontweight('bold')
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    axes[0].set_ylim(3.65,-1.15)
    base = by_name['main_weighted28']['total_cash_cost_yuan']
    dp=(1-base/by_name['comparison_point']['total_cash_cost_yuan'])*100
    dn=(1-base/by_name['comparison_no_storage']['total_cash_cost_yuan'])*100
    fig.text(.26,.025,f'主方案费用较单一预测低 {dp:.2f}%，较无储能低 {dn:.2f}%。\n'
             '期间：2025年2-12月；所有对照使用相同日期和初末储量条件。',fontsize=7.2,color=C['muted'])
    return fig


def plot_selected(ds, arrays, payload):
    fig=plt.figure(figsize=(7.2,7.3))
    outer=fig.add_gridspec(2,2,left=.11,right=.97,top=.93,bottom=.105,wspace=.30,hspace=.49)
    dates=['2025-03-20','2025-06-21','2025-09-23','2025-12-21']
    indices=[ds.dates.index(date.fromisoformat(d))-31 for d in dates]
    max_q=float(arrays['Q'][indices].max())*.006
    max_e=float(arrays['E'][indices].max())*6
    totals={item['date']:item for item in payload['q2']['selected_dates']}
    for letter, day, idx, grid in zip('abcd',dates,indices,outer):
        inner=grid.subgridspec(2,1,height_ratios=[2.2,1],hspace=.12)
        top=fig.add_subplot(inner[0]);bottom=fig.add_subplot(inner[1],sharex=top)
        panel(top,letter,day)
        step(top,arrays['Q'][idx]*.006,color=C['purchase'])
        top.fill_between(np.arange(145)/6,0,np.r_[arrays['Q'][idx],arrays['Q'][idx,-1]]*.006,
                         step='post',color=C['purchase'],alpha=.10)
        top.set(ylabel='计划购电 (MW)',ylim=(0,max_q*1.23))
        top.tick_params(labelbottom=False)
        top.text(.03,.91,f'应急电量 {totals[day]["emergency_purchase_kwh"]:,.1f} kWh',
                 transform=top.transAxes,fontsize=7.5,va='top')
        step(bottom,arrays['E'][idx]*6,color=C['emergency'])
        bottom.fill_between(np.arange(145)/6,0,np.r_[arrays['E'][idx],arrays['E'][idx,-1]]*6,
                            step='post',color=C['emergency'],alpha=.24)
        bottom.set(ylabel='应急 (kW)',ylim=(0,max_e*1.18))
        if arrays['E'][idx].max()<1e-6:
            bottom.text(.5,.45,'无紧急购电',transform=bottom.transAxes,ha='center',fontsize=8)
        time_axis(top,label=False,ticks=6);time_axis(bottom,ticks=6)
    note(fig,'四天使用相同的计划功率刻度和相同的应急功率刻度；上下两类面板的单位不同。')
    return fig


def plot_sensitivity(by_name):
    base=by_name['main_weighted28']['total_cash_cost_yuan']
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.65),sharey=True)
    fig.subplots_adjust(left=.12,right=.965,top=.85,bottom=.23,wspace=.27)
    cases=[('terminal','终端价值系数',C['state'],'o'),('residual','历史残差幅度',C['pv'],'s')]
    for ax,letter,(key,title,color,marker) in zip(axes,'ab',cases):
        panel(ax,letter,title)
        vals=[by_name[f'sensitivity_{key}_0_8']['total_cash_cost_yuan'],base,
              by_name[f'sensitivity_{key}_1_2']['total_cash_cost_yuan']]
        change=(np.asarray(vals)/base-1)*100
        ax.axhline(0,color=C['rule'],ls=(0,(4,3)),lw=.8)
        ax.plot([.8,1,1.2],change,ls='--',marker=marker,color=color,mfc='white',mew=1.3,ms=5)
        for x,y in zip([.8,1,1.2],change):
            ax.annotate(f'{y:+.3f}%',(x,y),xytext=(0,9),textcoords='offset points',
                        ha='center',fontsize=8)
        ax.set(xlim=(.75,1.25),ylim=(-.3,2.2),xticks=[.8,1,1.2],xlabel='参数相对基准倍数')
    axes[0].set_ylabel('334天实际费用变化 (%)')
    note(fig,'每个点均重新回放334天；两面板使用相同纵轴。虚线只连接已计算的三个点。',y=.055)
    return fig


def plot_annual(ds, arrays):
    fig,axes=plt.subplots(2,1,figsize=(7.2,7.5))
    fig.subplots_adjust(left=.12,right=.90,top=.93,bottom=.11,hspace=.38)
    dates=ds.dates[31:]
    rows=[i for i,d in enumerate(dates) if d.day==1]
    labels=[f'{dates[i].month:02d}月' for i in rows]
    for ax,letter,key,title,cmap in zip(axes,'ab',['Q','E'],
            ['计划购电的日内与季节分布','紧急购电的日内与季节分布'],['viridis','magma_r']):
        panel(ax,letter,title)
        values=arrays[key]*.006
        image=ax.imshow(values,origin='upper',aspect='auto',interpolation='none',
                        extent=(0,24,len(dates)-.5,-.5),vmin=0,vmax=float(values.max()),cmap=cmap)
        ax.set(yticks=rows,yticklabels=labels,ylabel='2025年')
        time_axis(ax,ticks=4)
        bar=fig.colorbar(image,ax=ax,pad=.025,fraction=.035,aspect=25)
        bar.set_label('功率 (MW)');bar.outline.set_linewidth(.5)
    note(fig,'每格为一天的一个10 min区间；两幅色标独立，均从0起，无平滑、无极值截断。')
    return fig


def plot_data_overview(project):
    p=json.loads((project/'reports/plot_payload.json').read_text(encoding='utf-8'))
    s=json.loads((project/'reports/dataset_profile.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.9))
    fig.subplots_adjust(left=.12,right=.97,top=.92,bottom=.14,wspace=.33,hspace=.53)
    h=np.asarray(p['hours'])
    panel(axes[0,0],'a','日内负载与实际光伏')
    for key,label,color,ls in [('load','负载',C['load'],'-'),('pv','光伏',C['pv'],'--')]:
        q=p['profiles'][key]
        axes[0,0].fill_between(h,np.asarray(q['p10'])/1000,np.asarray(q['p90'])/1000,color=color,alpha=.17,lw=0)
        axes[0,0].plot(h,np.asarray(q['mean'])/1000,color=color,label=label,ls=ls)
    axes[0,0].set(ylabel='功率 (MW)',ylim=(0,12));axes[0,0].legend(loc='upper left',ncol=2)
    panel(axes[0,1],'b','附件4电价的日内分布')
    q=p['profiles']['price']
    axes[0,1].fill_between(h,q['p10'],q['p90'],color=C['state'],alpha=.20,lw=0)
    axes[0,1].plot(h,q['mean'],color=C['state'])
    axes[0,1].set(ylabel='电价 (元/kWh)',ylim=(0,float(max(q['p90']))*1.10))
    for ax in axes[0]:
        time_axis(ax,ticks=6);ax.set_xlabel('源时间标签 (h)')
    panel(axes[1,0],'c','月均负载与实际光伏')
    months=[r['month'] for r in p['monthly']]
    for field,label,color,marker in [('load_mean_kw','负载',C['load'],'o'),('pv_mean_kw','光伏',C['pv'],'s')]:
        axes[1,0].plot(months,np.array([r[field] for r in p['monthly']])/1000,
                       marker=marker,color=color,label=label,ms=3.5,mfc='white',mew=1)
    axes[1,0].set(xlabel='月份',ylabel='平均功率 (MW)',xticks=[1,3,5,7,9,12],ylim=(0,8))
    axes[1,0].legend(loc='upper left',ncol=2)
    panel(axes[1,1],'d','附件3预报误差与提前量')
    bands=s['forecast_alignment']['lead_bands']
    keys=['1-6','7-12','13-18','19-24'];values=np.array([bands[k]['mae_kw'] for k in keys])
    bars=axes[1,1].bar(keys,values,color=plt.colormaps['viridis']([.25,.43,.61,.79]),width=.62)
    for bar,value in zip(bars,values):
        axes[1,1].text(bar.get_x()+bar.get_width()/2,value+15,f'{value:.1f}',ha='center',fontsize=8)
    axes[1,1].set(xlabel='预报提前量 (h)',ylabel='MAE (kW)',ylim=(0,float(values.max())*1.20))
    note(fig,'来源：附件2、3、4；阴影为跨日P10-P90分位范围，不是置信区间。\n'
         '预报误差只统计实际光伏>0的整点；本图为全年描述统计，不作为第二问日前决策输入。')
    return fig


CAPTIONS={
    '第一问调度':'第一问典型日调度。a为负载、光伏和计划购电功率，b为母线侧充放电功率，c为电池储量及1.2/10.8 MWh边界，d为已知电价。区间功率采用真实10 min阶梯；电池储量为区间边界值的连线。',
    '第二问方法对照':'相同334天和初末储量条件下的方法对照。a分解计划与紧急购电费用，b比较紧急购电量。浅色带及加粗标签标识预先指定的主方案。主方案相对7日均值场景方案费用仅降低约0.184%，不支持宣称显著优势；费用更低也不意味着所有对照下应急电量均更少。',
    '第二问指定日期':'四个指定日期的购电轨迹。每组上图为计划功率(MW)，下图为应急功率(kW)；同类面板共用尺度，避免为突出微小波动而逐日改变纵轴。日累计应急电量由原轨迹求和，单位kWh。',
    '第二问灵敏度':'终端价值及历史残差幅度的单因素±20%灵敏度。每个结果均连续回放334天。两面板使用相同纵轴；虚线为阅读辅助，不意味着已计算连续参数空间或已证明普遍稳健。',
    '第二问全年调度热力图':'主方案334×144个时段的计划及应急功率。两图分别用各自完整数据范围映射颜色，色条均明确标出MW；没有分位数截断或平滑。可用于观察时段分布，不能据颜色差异直接比较两个独立色标下的绝对功率。',
    'C题数据概览':'全年附件描述统计。日内阴影为同一源时间标签跨日P10-P90范围，实线/虚线为均值。附件3预报MAE使用实际光伏为正的目标整点。附件4电价展示和全年描述统计不参与本版第二问的日前决策。',
}


def build_figures(run, project, output, *, include_overview=True, dpi=450):
    run,project,output=Path(run),Path(project),Path(output)
    ds=read_dataset(project/'data/processed')
    summaries=json.loads((run/'experiment_summaries.json').read_text(encoding='utf-8'))
    by_name={s['name']:s for s in summaries}
    if len(by_name)!=8:
        raise ValueError('All eight completed experiments are required')
    audit=json.loads((run/'independent_audit.json').read_text(encoding='utf-8'))
    if audit['status']!='passed' or len(audit['experiments'])!=8 or 'workbooks' not in audit:
        raise ValueError('Only independently audited results may be plotted')
    metadata=json.loads((run/'config.json').read_text(encoding='utf-8'))
    if ds.sources!=metadata['input_sha256']:
        raise ValueError('Processed input does not match the audited run')
    q1=json.loads((run/'q1.json').read_text(encoding='utf-8'))['milp']
    payload=json.loads((run/'workbook_payload.json').read_text(encoding='utf-8'))
    with np.load(run/'main_weighted28/trajectories.npz') as stored:
        arrays={k:stored[k].copy() for k in ['Q','C','D','S','E','U','day_indices']}
    if not np.array_equal(arrays['day_indices'],np.arange(31,365)):
        raise ValueError('Unexpected trajectory date alignment')
    np.testing.assert_allclose(arrays['E'].sum(),by_name['main_weighted28']['total_emergency_purchase_kwh'],rtol=1e-10)
    for s in summaries:
        np.testing.assert_allclose(s['total_planned_cost_yuan']+s['total_emergency_cost_yuan'],s['total_cash_cost_yuan'],rtol=1e-12)
    inputs=[run/n for n in ['experiment_summaries.json','q1.json','workbook_payload.json',
                              'independent_audit.json','config.json','main_weighted28/trajectories.npz']]
    if include_overview:
        inputs += [project/'reports'/n for n in ['plot_payload.json','dataset_profile.json']]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    apply_style();output.mkdir(parents=True,exist_ok=True)
    jobs=[('第一问调度',lambda:plot_q1(ds,q1)),
          ('第二问方法对照',lambda:plot_methods(by_name)),
          ('第二问指定日期',lambda:plot_selected(ds,arrays,payload)),
          ('第二问灵敏度',lambda:plot_sensitivity(by_name)),
          ('第二问全年调度热力图',lambda:plot_annual(ds,arrays))]
    if include_overview:
        jobs.append(('C题数据概览',lambda:plot_data_overview(project)))
    records=[]
    with PdfPages(output/'科研图表合集.pdf',metadata={'Title':'C题科研图表合集','Author':'C题建模团队',
                  'Subject':'已审计结果的可复现科研可视化','Keywords':STYLE_VERSION}) as pdf:
        for name,build in jobs:
            records.append(save_figure(build(),output,name,pdf=pdf,dpi=dpi))
    for path,digest in before.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise AssertionError('A figure input was modified')
    lines=['# 图表说明与使用','',f'视觉版本：{STYLE_VERSION}。所有图均直接使用已保存数据；本次未重新拟合、求解或调整模型参数。','',
           'PNG为450 dpi；SVG保留可编辑文字；PDF合集保留矢量坐标、文字与曲线，热力图数据层为嵌入图像。',
           '本组按约183 mm双栏宽度设计，缩小到单栏时请重排布局及字号，不宜直接缩小到一半。','']
    for i,(name,_) in enumerate(jobs,1):
        lines.extend([f'## 图{i} {name}','',CAPTIONS[name],''])
    (output/'图表说明.md').write_text('\n'.join(lines),encoding='utf-8')
    details={'style_version':STYLE_VERSION,'run_id':run.name,'figure_count':len(records),
             'inputs_unchanged':True,'input_sha256':before,'figures':records,
             'note':'程序边界与缺字检查通过；视觉审查结果另行记录。'}
    (output/'figure_manifest.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return details


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--skip-overview',action='store_true')
    args=parser.parse_args()
    output=args.output or args.run/'deliverables/figures'
    result=build_figures(args.run,args.project_root,output,include_overview=not args.skip_overview)
    print(json.dumps({'figure_count':result['figure_count'],'output':str(output),'inputs_unchanged':True},ensure_ascii=False))


if __name__=='__main__':
    main()

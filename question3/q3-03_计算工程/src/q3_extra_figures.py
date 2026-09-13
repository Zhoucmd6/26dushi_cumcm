"""保持现有科研图风格，增加PDF消融链、终端对照和R可解释性。"""
import numpy as np
import pandas as pd
from scientific_style import plt,COLORS,panel,note,save_figure

def extra_figures(run,records,out,pdf):
    qa=[]
    fig,axs=plt.subplots(1,2,figsize=(9,4.4));fig.subplots_adjust(left=.10,right=.97,top=.86,bottom=.30,wspace=.35)
    names=['info_mask0','point_official','scale_pooled','main','cvar95']
    labels=['B0\n仅0点预报','B1\n原始点预测','B2\n统一尺度','B3\n条件尺度','B4\nCVaR95']
    colors=[COLORS['muted'],COLORS['planned_cost'],COLORS['purchase'],COLORS['pv'],COLORS['state']]
    for ax,metric,ylabel,title in [(axs[0],'total_fee_yuan','2—12月总费用 (万元)','逐项模型对照'),
                                   (axs[1],'daily_emergency_cvar95_yuan','日紧急费用CVaR95 (万元)','实现尾部风险')]:
        values=[records[n][metric]/1e4 for n in names]
        ax.bar(range(5),values,color=colors,width=.62)
        ax.set_xticks(range(5),labels,fontsize=7);ax.set_ylabel(ylabel);ax.set_ylim(0,max(values)*1.18)
        for j,v in enumerate(values):ax.text(j,v+max(values)*.025,f'{v:.2f}',ha='center',fontsize=7)
        panel(ax,'a' if ax is axs[0] else 'b',title)
    note(fig,'B2→B3只改变误差尺度；B3→B4只增加风险项（α=0.95，λ=0.1）。B0仍每日优化4次，仅屏蔽日内新预报。')
    qa.append(save_figure(fig,out,'05_B0至B4模型对照',pdf=pdf))

    cache=dict(np.load(run/'forecast_cache.npz'));day=171;reference=cache['sref'][day]
    fig,axs=plt.subplots(1,2,figsize=(9,4.3));fig.subplots_adjust(left=.10,right=.97,top=.84,bottom=.28,wspace=.34)
    for g,color in enumerate([COLORS['muted'],COLORS['purchase'],COLORS['pv'],COLORS['state']]):
        # 图示式14函数；仅画当天剩余时域能出现的提前量支持区间。
        h=np.linspace(1/6,(g+1)*6,100);sigma=cache['sigma0'][day,g]*np.exp(cache['beta'][day]*h)
        axs[0].plot(h,sigma,color=color,label=f'目标时刻{6*g}—{6*g+6}h')
        axs[1].plot(h,1/(1+(sigma/reference)**2),color=color)
    for ax in axs:ax.set_xlabel('原始预报提前量 h (小时)');ax.set_xticks([0,6,12,18,24])
    axs[0].set_ylabel('指数条件尺度 σ (kW)');axs[0].legend(fontsize=7);panel(axs[0],'a','6月21日历史拟合的指数候选')
    axs[1].set_ylabel('可信度指标 R');axs[1].set_ylim(0,1.05);panel(axs[1],'b','同组内由尺度确定的R')
    note(fig,'R=1/[1+(σ/s_ref)²]，不是正确概率。本图解释指数候选；主方案按1月验证使用分箱尺度，其R不强制单调。')
    qa.append(save_figure(fig,out,'06_提前量尺度与可信度解释',pdf=pdf))

    fig,axs=plt.subplots(1,2,figsize=(9,4.3));fig.subplots_adjust(left=.11,right=.97,top=.84,bottom=.28,wspace=.39)
    names=['terminal_zero' if 'terminal_zero' in records else 'main','main' if records['main']['gamma'] else 'terminal_soft','terminal_hard']
    labels=['无软目标','正软目标','每日硬目标']
    for j,name in enumerate(names):
        row=records[name];axs[0].bar(j,row['total_fee_yuan']/1e4,color=[COLORS['muted'],COLORS['purchase'],COLORS['state']][j],width=.6)
    axs[0].set_xticks(range(3),labels);axs[0].set_ylabel('2—12月总费用 (万元)');panel(axs[0],'a','相同预测与信息条件的终端对照')
    for alpha,color,marker in [(.9,COLORS['purchase'],'o'),(.95,COLORS['state'],'s')]:
        names=['main',f'cvar{round(100*alpha)}_lambda005',f'cvar{round(100*alpha)}']
        x=[records[n]['total_fee_yuan']/1e4 for n in names];y=[records[n]['daily_emergency_cvar95_yuan']/1e4 for n in names]
        axs[1].plot(x,y,color=color,marker=marker,label=f'α={alpha:.2f}')
        for xx,yy,lam in zip(x[1:],y[1:],[.05,.1]):axs[1].annotate(f'λ={lam:g}',(xx,yy),xytext=(4,5),textcoords='offset points',fontsize=7)
    axs[1].set_xlabel('2—12月总费用 (万元)');axs[1].set_ylabel('日紧急费用CVaR95 (万元)');axs[1].legend(fontsize=7)
    panel(axs[1],'b','CVaR风险偏好网格');note(fig,'所有方案跨日连续并在年末统一6000 kWh。软目标仅进入优化目标，不计入实际电费；λ=0对应B3。')
    qa.append(save_figure(fig,out,'07_终端条件与风险参数',pdf=pdf))
    return qa

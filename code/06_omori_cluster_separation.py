#!/usr/bin/env python3
"""
论文C 压轴分析: 丛集分离 + Omori-Utsu 指数测量
─────────────────────────────────────────────
把每个目录分解为: 背景过程(Poisson) + 丛集过程(Omori律)
单独测量丛集内部的 Omori p 指数 — 不会坍缩的真正物理量
"""
import pandas as pd, numpy as np
from scipy import stats, optimize
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import json, warnings
warnings.filterwarnings('ignore')
UP='/mnt/user-data/uploads/'

ZONES={
 'Parkfield':{'file':'1781909251892_query_SAF.csv','group':'Strike-slip','mc_floor':2.0,'color':'#2166ac'},
 'NAF':{'file':'1781909251890_query_NAF.csv','group':'Strike-slip','mc_floor':4.5,'color':'#4393c3'},
 'Alpine':{'file':'1781909251891_query_NZ.csv','group':'Strike-slip','mc_floor':4.0,'color':'#74add1'},
 'Nankai':{'file':'1781909251891_query_Nankal_Trough.csv','group':'Subduction','mc_floor':4.5,'color':'#f4a582'},
 'Chile':{'file':'1781909251890_query_Chile_Subduction.csv','group':'Subduction','mc_floor':5.0,'color':'#d6604d'},
 'Cascadia':{'file':'1781909251889_query_Cascadia.csv','group':'Subduction','mc_floor':3.5,'color':'#b2182b'},
 'Charlevoix':{'file':'1781909251889_query_Charlevoix.csv','group':'Intraplate','mc_floor':2.5,'color':'#66c2a5'},
 'Longmenshan':{'file':'1781909251890_query_Longmenshan.csv','group':'Intraplate','mc_floor':3.5,'color':'#8da0cb'},
}

def to_mw(mag,mt):
    mt=str(mt).lower()
    if mt.startswith('mw'):return mag
    if mt=='mb':return 0.85*mag+1.03
    if mt=='ms':return 0.67*mag+2.07 if mag<=6.1 else 0.99*mag+0.08
    return mag
def estimate_mc(m,floor):
    bins=np.arange(np.floor(m.min()),np.ceil(m.max())+0.1,0.1)
    h,e=np.histogram(m,bins=bins);return max(e[np.argmax(h)]+0.2,floor)
def haversine(la1,lo1,la2,lo2):
    R=6371.0;p1,p2=np.radians(la1),np.radians(la2)
    dp=np.radians(la2-la1);dl=np.radians(lo2-lo1)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def gk_mainshock_mask(lat,lon,mag,t):
    N=len(t);is_main=np.ones(N,dtype=bool)
    aftershock_of=-np.ones(N,dtype=int)
    for i in range(N):
        if not is_main[i]:continue
        M=mag[i];L=10**(0.1238*M+0.983)
        T=10**(0.032*M+2.7389) if M>=6.5 else 10**(0.5409*M-0.547)
        for j in range(i+1,N):
            if t[j]-t[i]>T:break
            if is_main[j] and mag[j]<=M and haversine(lat[i],lon[i],lat[j],lon[j])<=L:
                is_main[j]=False
                aftershock_of[j]=i
    return is_main, aftershock_of

def omori_p(dt_days):
    """
    拟合 Omori-Utsu 律: n(t)=K/(t+c)^p
    用余震相对主震的时间差, MLE估计p
    """
    dt=dt_days[dt_days>0]
    if len(dt)<10: return np.nan,np.nan,len(dt)
    # MLE for modified Omori (Ogata 1983)
    def negll(params):
        p,c=params
        if p<=0 or c<=0: return 1e10
        T=dt.max()
        if abs(p-1)<1e-6:
            A=np.log(T+c)-np.log(c)
        else:
            A=((T+c)**(1-p)-c**(1-p))/(1-p)
        return -(np.sum(-p*np.log(dt+c)) - len(dt)*np.log(A))
    try:
        res=optimize.minimize(negll,[1.1,0.01],method='Nelder-Mead',
                              options={'xatol':1e-4,'fatol':1e-4})
        p,c=res.x
        return p,c,len(dt)
    except:
        return np.nan,np.nan,len(dt)

print("="*72)
print("  论文C压轴: 丛集分离 + Omori-Utsu 指数测量")
print("="*72)
print("""
  方法: GK去丛集同时记录'每个余震属于哪个主震'
        → 分离出纯丛集过程 → 测内部Omori p指数
  物理: p指数是断层流体/温度/岩性的探针, 不会坍缩为Poisson
""")

OUT={}
for k,cfg in ZONES.items():
    df=pd.read_csv(UP+cfg['file'])
    df=df[df['type']=='earthquake'].copy()
    df['time']=pd.to_datetime(df['time'],format='ISO8601')
    df['Mw']=df.apply(lambda r:to_mw(r['mag'],r['magType']),axis=1)
    df=df.sort_values('time').reset_index(drop=True)
    mc=estimate_mc(df['Mw'].values,cfg['mc_floor'])
    dfc=df[df['Mw']>=mc].sort_values('time').reset_index(drop=True)
    if len(dfc)<30: 
        print(f"  {k}: 样本不足跳过"); continue
    t=(dfc['time']-dfc['time'].min()).dt.total_seconds().values/86400.0
    lat,lon,mag=dfc['latitude'].values,dfc['longitude'].values,dfc['Mw'].values

    mask,aft_of=gk_mainshock_mask(lat,lon,mag,t)
    n_main=mask.sum(); n_aft=(~mask).sum()

    # 分离丛集: 每个余震相对其主震的时间差
    dt_list=[]
    for j in np.where(~mask)[0]:
        i=aft_of[j]
        if i>=0:
            dt_list.append(t[j]-t[i])
    dt_arr=np.array(dt_list)

    p,c,n_used=omori_p(dt_arr)
    aft_frac=n_aft/len(dfc)

    OUT[k]={'group':cfg['group'],'color':cfg['color'],'mc':mc,
            'n_total':len(dfc),'n_main':int(n_main),'n_aft':int(n_aft),
            'aft_frac':aft_frac,'omori_p':p,'omori_c':c,'n_aftershocks_fit':n_used}
    print(f"─── {k} [{cfg['group']}] ───")
    print(f"  总{len(dfc)} = 主震{n_main} + 余震{n_aft} (余震占比{aft_frac:.0%})")
    print(f"  Omori p = {p:.3f}  (c={c:.4f}, 拟合用{n_used}个余震)")
    print()

print("="*72)
print("  Omori p 指数 — 真正区分构造环境的物理量")
print("="*72)
print(f"\n  {'靶区':14s}{'类型':12s}{'余震占比':>9s}{'Omori_p':>9s}")
print("  "+"-"*48)
for k,v in OUT.items():
    print(f"  {k:14s}{v['group']:12s}{v['aft_frac']:9.0%}{v['omori_p']:9.3f}")

# 按构造分组看p
print(f"\n  按构造类型平均 Omori p:")
for g in ['Strike-slip','Subduction','Intraplate']:
    ps=[v['omori_p'] for v in OUT.values() if v['group']==g and not np.isnan(v['omori_p'])]
    fr=[v['aft_frac'] for v in OUT.values() if v['group']==g]
    if ps:
        print(f"    {g:12s}: <p>={np.mean(ps):.3f}  <余震占比>={np.mean(fr):.0%}")

with open('/home/claude/omori_results.json','w') as f:
    json.dump(OUT,f,indent=2,default=str)
print(f"\n  ✓ 已保存 omori_results.json")

#!/usr/bin/env python3
"""
方向B: 单段精钻 — 最简单孤立断层段能否守住互斥?
靶区: (1) Parkfield 特征地震序列(著名~22年周期)
      (2) NAF 东段(A阶段发现β̂=0.85的残余信号)
对照: 同时测背景(去丛集)与完整,看单段是否不同于全断层
"""
import pandas as pd, numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import json, warnings
warnings.filterwarnings('ignore')

UP='/mnt/user-data/uploads/'

def to_mw(mag,mt):
    mt=str(mt).lower()
    if mt.startswith('mw'): return mag
    if mt=='mb': return 0.85*mag+1.03
    if mt=='ms': return 0.67*mag+2.07 if mag<=6.1 else 0.99*mag+0.08
    return mag

def wigner_gue(s): return (32/np.pi**2)*s**2*np.exp(-4*s**2/np.pi)
def wigner_goe(s): return (np.pi/2)*s*np.exp(-np.pi*s**2/4)
def make_cdf(f,mx=10,n=10000):
    s=np.linspace(0,mx,n);c=cumulative_trapezoid(f(s),s,initial=0);c/=c[-1]
    return interp1d(s,c,bounds_error=False,fill_value=(0,1))
POI=lambda x:1-np.exp(-x); GUE=make_cdf(wigner_gue); GOE=make_cdf(wigner_goe)
def sr(sp):
    if len(sp)<3:return np.nan,np.nan
    r=np.minimum(sp[:-1],sp[1:])/np.maximum(sp[:-1],sp[1:])
    return r.mean(),r.std()/np.sqrt(len(r))
def beta_from_r(r):
    if np.isnan(r):return np.nan
    if r<=0.386:return 0.0
    elif r<=0.536:return (r-0.386)/0.15
    elif r<=0.603:return 1+(r-0.536)/0.067
    else:return min(2+(r-0.603)/0.1,3)
def analyze(t_yr,tag):
    iv=np.diff(np.sort(t_yr))
    iv=iv[iv>0]
    if len(iv)<8: return None
    s=iv/iv.mean()
    r,re=sr(s); 
    ksp,pp=stats.kstest(s,POI); kso,po=stats.kstest(s,GOE)
    cv=np.std(iv)/np.mean(iv)
    best='Poisson' if pp>=po else 'GOE'
    return {'tag':tag,'n':len(s),'r':r,'re':re,'bh':beta_from_r(r),
            'cv':cv,'pp':pp,'po':po,'best':best,'mean_iv':np.mean(iv)}

print("="*72)
print("  方向B: 单段精钻")
print("="*72)

# ════════ B1: Parkfield 特征地震序列 ════════
print("\n" + "="*72)
print("  B1: Parkfield — 提取 M≥5.5 特征地震 (著名~22年周期)")
print("="*72)
df=pd.read_csv(UP+'1781909251892_query_SAF.csv')
df=df[df['type']=='earthquake'].copy()
df['time']=pd.to_datetime(df['time'],format='ISO8601')
df['Mw']=df.apply(lambda r:to_mw(r['mag'],r['magType']),axis=1)
df=df.sort_values('time').reset_index(drop=True)

# Parkfield特征地震是M~6, 但仪器期内只有1966,2004两次M6
# 提取不同震级阈值看周期性如何随震级变化
print("\n  不同震级阈值下的序列特征:")
print(f"  {'Mth':>5s}{'n':>5s}{'⟨r⟩':>8s}{'CV':>6s}{'β̂':>6s}{'mean_dt':>9s}  best")
park_results={}
for mth in [4.0,4.5,5.0,5.5]:
    sub=df[df['Mw']>=mth]
    t=(sub['time']-df['time'].min()).dt.total_seconds().values/86400.0/365.25
    res=analyze(t,f'Parkfield_M{mth}')
    if res:
        park_results[mth]=res
        print(f"  {mth:5.1f}{res['n']:5d}{res['r']:8.3f}{res['cv']:6.2f}"
              f"{res['bh']:6.2f}{res['mean_iv']:9.2f}  {res['best']}")

# ════════ B2: NAF 东段 (>37°E) ════════
print("\n" + "="*72)
print("  B2: NAF 东段 (>37°E) — A阶段残余信号复核")
print("="*72)
naf=pd.read_csv(UP+'1781909251890_query_NAF.csv')
naf=naf[naf['type']=='earthquake'].copy()
naf['time']=pd.to_datetime(naf['time'],format='ISO8601')
naf['Mw']=naf.apply(lambda r:to_mw(r['mag'],r['magType']),axis=1)
naf=naf.sort_values('time').reset_index(drop=True)
naf_c=naf[naf['Mw']>=5.0]

print(f"\n  按经度分段 (Mc=5.0完整目录):")
print(f"  {'段':16s}{'n':>5s}{'⟨r⟩':>8s}{'CV':>6s}{'β̂':>6s}  best  Poi_p")
naf_results={}
segs={'东段 >37E':(37,99),'中段 33-37E':(33,37),'西段 <33E':(0,33),'全段':(0,99)}
for name,(lo,hi) in segs.items():
    seg=naf_c[(naf_c['longitude']>=lo)&(naf_c['longitude']<hi)]
    t=(seg['time']-naf['time'].min()).dt.total_seconds().values/86400.0/365.25
    res=analyze(t,name)
    if res:
        naf_results[name]=res
        print(f"  {name:16s}{res['n']:5d}{res['r']:8.3f}{res['cv']:6.2f}"
              f"{res['bh']:6.2f}  {res['best']:7s}{res['pp']:.3f}")

# ════════ 裁决 ════════
print("\n" + "="*72)
print("  方向B 裁决")
print("="*72)
print("""
  问题: 最简单孤立断层段能否守住 GOE 互斥?
""")
# Parkfield最高阈值
if park_results:
    hi_mth=max(park_results.keys())
    pr=park_results[hi_mth]
    print(f"  Parkfield M≥{hi_mth}: ⟨r⟩={pr['r']:.3f}, β̂={pr['bh']:.2f}, {pr['best']}")
    if pr['bh']>0.5:
        print(f"    → 单段在高震级阈值显现互斥! (n={pr['n']})")
    else:
        print(f"    → 仍接近Poisson (n={pr['n']}, 样本受限)")
# NAF东段
if '东段 >37E' in naf_results:
    er=naf_results['东段 >37E']
    print(f"\n  NAF东段: ⟨r⟩={er['r']:.3f}, β̂={er['bh']:.2f}, {er['best']}")
    if er['bh']>0.5:
        print(f"    → 东段确认残余互斥信号 (β̂={er['bh']:.2f})")
    else:
        print(f"    → 残余信号微弱")

out={'parkfield':{str(k):v for k,v in park_results.items()},'naf_seg':naf_results}
with open('/home/claude/segment_results.json','w') as f:
    json.dump(out,f,indent=2,default=str)
print(f"\n  ✓ 已保存 segment_results.json")

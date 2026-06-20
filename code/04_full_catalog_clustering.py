#!/usr/bin/env python3
"""
方向A: 反向验证 — 完整目录(含余震)的丛集结构
检验假说: 地震记忆藏在丛集里, 而非去丛集后的主震里 (Corral 2004)

关键: 丛集过程的间隔分布不是Poisson(指数), 而是幂律/Gamma,
      表现为 ⟨r⟩ < 0.386 (比泊松更聚集, 即"反互斥"/吸引)
"""
import pandas as pd, numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import json, warnings
warnings.filterwarnings('ignore')

ZONES = {
    'Parkfield':   {'file':'1781909251892_query_SAF.csv',          'group':'Strike-slip','mc_floor':2.0,'color':'#2166ac'},
    'NAF':         {'file':'1781909251890_query_NAF.csv',          'group':'Strike-slip','mc_floor':4.5,'color':'#4393c3'},
    'Alpine':      {'file':'1781909251891_query_NZ.csv',           'group':'Strike-slip','mc_floor':4.0,'color':'#74add1'},
    'Nankai':      {'file':'1781909251891_query_Nankal_Trough.csv','group':'Subduction', 'mc_floor':4.5,'color':'#f4a582'},
    'Chile':       {'file':'1781909251890_query_Chile_Subduction.csv','group':'Subduction','mc_floor':5.0,'color':'#d6604d'},
    'Cascadia':    {'file':'1781909251889_query_Cascadia.csv',     'group':'Subduction', 'mc_floor':3.5,'color':'#b2182b'},
    'Charlevoix':  {'file':'1781909251889_query_Charlevoix.csv',   'group':'Intraplate', 'mc_floor':2.5,'color':'#66c2a5'},
    'Longmenshan': {'file':'1781909251890_query_Longmenshan.csv',  'group':'Intraplate', 'mc_floor':3.5,'color':'#8da0cb'},
}
UP='/mnt/user-data/uploads/'

def to_mw(mag,mt):
    mt=str(mt).lower()
    if mt.startswith('mw'): return mag
    if mt=='mb': return 0.85*mag+1.03
    if mt=='ms': return 0.67*mag+2.07 if mag<=6.1 else 0.99*mag+0.08
    return mag
def estimate_mc(mags,floor):
    bins=np.arange(np.floor(mags.min()),np.ceil(mags.max())+0.1,0.1)
    h,e=np.histogram(mags,bins=bins)
    return max(e[np.argmax(h)]+0.2,floor)

def wigner_gue(s): return (32/np.pi**2)*s**2*np.exp(-4*s**2/np.pi)
def wigner_goe(s): return (np.pi/2)*s*np.exp(-np.pi*s**2/4)
def make_cdf(f,mx=12,n=12000):
    s=np.linspace(0,mx,n);c=cumulative_trapezoid(f(s),s,initial=0);c/=c[-1]
    return interp1d(s,c,bounds_error=False,fill_value=(0,1))
POI=lambda x:1-np.exp(-x); GUE=make_cdf(wigner_gue); GOE=make_cdf(wigner_goe)

def spacing_ratio(sp):
    if len(sp)<3: return np.nan,np.nan
    r=np.minimum(sp[:-1],sp[1:])/np.maximum(sp[:-1],sp[1:])
    return r.mean(),r.std()/np.sqrt(len(r))

def cv(iv):
    """变异系数: Poisson=1, 周期=0, 丛集>1"""
    return np.std(iv)/np.mean(iv)

print("="*72)
print("  方向A: 完整目录(含余震)丛集结构分析")
print("="*72)
print("""
  判读标尺:
    ⟨r⟩ ≈ 0.386, CV ≈ 1.0  → Poisson (随机)
    ⟨r⟩ > 0.50,  CV < 1.0  → 互斥/周期 (GOE/GUE)
    ⟨r⟩ < 0.386, CV > 1.0  → 丛集/吸引 (幂律, Corral律)
""")

ALL={}
for k,cfg in ZONES.items():
    df=pd.read_csv(UP+cfg['file'])
    df=df[df['type']=='earthquake'].copy()
    df['time']=pd.to_datetime(df['time'],format='ISO8601')
    df=df.sort_values('time').reset_index(drop=True)
    df['Mw']=df.apply(lambda r:to_mw(r['mag'],r['magType']),axis=1)
    mc=estimate_mc(df['Mw'].values,cfg['mc_floor'])
    dfc=df[df['Mw']>=mc].sort_values('time').reset_index(drop=True)
    if len(dfc)<20: continue

    t=(dfc['time']-dfc['time'].min()).dt.total_seconds().values/86400.0
    iv=np.diff(t)  # 天
    iv=iv[iv>0]
    if len(iv)<15: continue
    s=iv/iv.mean()

    r,re=spacing_ratio(s)
    cvv=cv(iv)
    ksp,pp=stats.kstest(s,POI)
    kso,po=stats.kstest(s,GOE)

    # 幂律/Gamma拟合: Corral的丛集间隔遵循 Gamma 分布 γ<1
    # 拟合 Gamma 形状参数 γ; γ=1 即指数(Poisson), γ<1 即丛集
    try:
        gshape,gloc,gscale=stats.gamma.fit(s,floc=0)
    except:
        gshape=np.nan

    # 判读
    if r<0.34 and cvv>1.3:
        verdict='CLUSTERED (丛集)'
    elif r>0.48:
        verdict='REPULSIVE (互斥)'
    else:
        verdict='Poisson-like'

    ALL[k]={'group':cfg['group'],'color':cfg['color'],'mc':mc,
            'n':len(s),'r':r,'re':re,'cv':cvv,'gamma':gshape,
            'pp':pp,'po':po,'verdict':verdict,'s':s.tolist()}

    print(f"─── {k} [{cfg['group']}] ───")
    print(f"  完整目录 n={len(s)} (Mc={mc:.1f}, 含全部余震)")
    print(f"  ⟨r⟩={r:.3f}  CV={cvv:.2f}  Gamma_γ={gshape:.2f}")
    print(f"  KS_Poisson p={pp:.4f}")
    print(f"  → {verdict}\n")

# 汇总
print("="*72)
print("  反向验证总裁决")
print("="*72)
print(f"\n  {'靶区':14s}{'n':>6s}{'⟨r⟩':>8s}{'CV':>7s}{'γ':>7s}  判读")
print("  "+"-"*60)
for k,v in ALL.items():
    print(f"  {k:14s}{v['n']:6d}{v['r']:8.3f}{v['cv']:7.2f}{v['gamma']:7.2f}  {v['verdict']}")

# 对比: 去丛集前后
print(f"\n  关键对比 (以 ⟨r⟩ 为例):")
print(f"  {'靶区':14s}{'去丛集后':>10s}{'完整目录':>10s}  变化")
declust=json.load(open('/home/claude/all_real_results.json'))
for k in ALL:
    if k in declust and declust[k]['etas']:
        r_dc=declust[k]['etas']['r']
        r_full=ALL[k]['r']
        arrow='↓丛集' if r_full<r_dc-0.03 else ('↑互斥' if r_full>r_dc+0.03 else '≈')
        print(f"  {k:14s}{r_dc:10.3f}{r_full:10.3f}  {arrow}")

with open('/home/claude/reverse_test_results.json','w') as f:
    json.dump(ALL,f,indent=2,default=str)
print(f"\n  ✓ 已保存 reverse_test_results.json")

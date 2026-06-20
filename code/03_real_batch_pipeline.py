#!/usr/bin/env python3
"""
全球断裂带真实数据 RMT 批量流水线
Real-Data Global Fault RMT Pipeline

4 工序: 震级统一(Scordilis) → Mc截断(MaxCurvature) → 双去丛集(GK+ETAS) → RMT分析
"""
import pandas as pd, numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import json, warnings
warnings.filterwarnings('ignore')

# ─── 靶区配置 (文件路径 + 物理先验) ───
ZONES = {
    'Parkfield':   {'file':'1781909251892_query_SAF.csv',          'group':'Strike-slip','comp':1.0,'fluid':1.0,'beta_pred':1.5,'mc_floor':2.0,'color':'#2166ac','marker':'o'},
    'NAF':         {'file':'1781909251890_query_NAF.csv',          'group':'Strike-slip','comp':1.5,'fluid':1.0,'beta_pred':1.0,'mc_floor':4.5,'color':'#4393c3','marker':'o'},
    'Alpine':      {'file':'1781909251891_query_NZ.csv',           'group':'Strike-slip','comp':1.0,'fluid':1.5,'beta_pred':1.3,'mc_floor':4.0,'color':'#74add1','marker':'o'},
    'Nankai':      {'file':'1781909251891_query_Nankal_Trough.csv','group':'Subduction', 'comp':5.0,'fluid':8.0,'beta_pred':0.5,'mc_floor':4.5,'color':'#f4a582','marker':'s'},
    'Chile':       {'file':'1781909251890_query_Chile_Subduction.csv','group':'Subduction','comp':6.0,'fluid':7.5,'beta_pred':0.6,'mc_floor':5.0,'color':'#d6604d','marker':'s'},
    'Cascadia':    {'file':'1781909251889_query_Cascadia.csv',     'group':'Subduction', 'comp':4.0,'fluid':9.0,'beta_pred':0.3,'mc_floor':3.5,'color':'#b2182b','marker':'s'},
    'Charlevoix':  {'file':'1781909251889_query_Charlevoix.csv',   'group':'Intraplate', 'comp':7.5,'fluid':3.0,'beta_pred':0.15,'mc_floor':2.5,'color':'#66c2a5','marker':'^'},
    'Longmenshan': {'file':'1781909251890_query_Longmenshan.csv',  'group':'Intraplate', 'comp':8.5,'fluid':4.0,'beta_pred':0.25,'mc_floor':3.5,'color':'#8da0cb','marker':'^'},
}
UP = '/mnt/user-data/uploads/'

# ─── 工序1: 震级统一 → Mw (Scordilis 2006) ───
def to_mw(mag, magType):
    mt = str(magType).lower()
    if mt.startswith('mw'): return mag
    if mt == 'mb':  return 0.85*mag + 1.03
    if mt == 'ms':  return 0.67*mag+2.07 if mag<=6.1 else 0.99*mag+0.08
    return mag  # ml, md ≈ Mw 近似

# ─── 工序2: Mc 估计 (Maximum Curvature) ───
def estimate_mc(mags, floor):
    bins = np.arange(np.floor(mags.min()), np.ceil(mags.max())+0.1, 0.1)
    hist, edges = np.histogram(mags, bins=bins)
    mc_mc = edges[np.argmax(hist)] + 0.2
    return max(mc_mc, floor)

# ─── 工序3a: Gardner-Knopoff 窗口去丛集 ───
def haversine(la1,lo1,la2,lo2):
    R=6371.0; p1,p2=np.radians(la1),np.radians(la2)
    dp=np.radians(la2-la1); dl=np.radians(lo2-lo1)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def decluster_gk(lat,lon,mag,t):
    N=len(t); is_main=np.ones(N,dtype=bool)
    for i in range(N):
        if not is_main[i]: continue
        M=mag[i]
        L=10**(0.1238*M+0.983)
        T=10**(0.032*M+2.7389) if M>=6.5 else 10**(0.5409*M-0.547)
        for j in range(i+1,N):
            if t[j]-t[i]>T: break
            if is_main[j] and mag[j]<=M and haversine(lat[i],lon[i],lat[j],lon[j])<=L:
                is_main[j]=False
    return is_main

# ─── 工序3b: ETAS 随机去丛集 ───
def decluster_etas(lat,lon,mag,t,mc,seed=2026):
    N=len(t); rng=np.random.default_rng(seed)
    K,alpha,c,p,d,q=0.008,1.5,0.01,1.1,0.5,1.8
    span=max(t.max()-t.min(),1)
    mu=N/(span*np.pi*100**2)*0.5
    probs=np.ones(N)
    for i in range(1,N):
        trig=0.0
        # 只看前面200个事件加速
        for j in range(max(0,i-200),i):
            dt=t[i]-t[j]
            if dt<=0: continue
            r=haversine(lat[i],lon[i],lat[j],lon[j])
            trig+=K*np.exp(alpha*(mag[j]-mc))/(dt+c)**p/(r**2+d**2)**q
        probs[i]=mu/(mu+trig) if (mu+trig)>0 else 1.0
    # 200次随机声明
    counts=np.zeros(N)
    for _ in range(200):
        counts+=(rng.random(N)<probs).astype(int)
    return counts/200>=0.5

# ─── 工序4: RMT 统计 ───
def wigner_gue(s): return (32/np.pi**2)*s**2*np.exp(-4*s**2/np.pi)
def wigner_goe(s): return (np.pi/2)*s*np.exp(-np.pi*s**2/4)
def make_cdf(f,mx=8,n=10000):
    s=np.linspace(0,mx,n); cc=cumulative_trapezoid(f(s),s,initial=0); cc/=cc[-1]
    return interp1d(s,cc,bounds_error=False,fill_value=(0,1))
POI=lambda x:1-np.exp(-x); GUE_CDF=make_cdf(wigner_gue); GOE_CDF=make_cdf(wigner_goe)

def spacing_ratio(sp):
    if len(sp)<3: return np.nan,np.nan
    r=np.minimum(sp[:-1],sp[1:])/np.maximum(sp[:-1],sp[1:])
    return r.mean(), r.std()/np.sqrt(len(r))
def beta_from_r(r):
    if np.isnan(r): return np.nan
    if r<=0.386: return 0.0
    elif r<=0.536: return (r-0.386)/0.15
    elif r<=0.603: return 1+(r-0.536)/0.067
    else: return min(2+(r-0.603)/0.1,3)

def rmt_analyze(t_days, label):
    tt=np.sort(t_days)
    iv=np.diff(tt)/365.25
    if len(iv)<10: return None
    s=iv/iv.mean()
    ksp,pp=stats.kstest(s,POI)
    kso,po=stats.kstest(s,GOE_CDF)
    ksu,pu=stats.kstest(s,GUE_CDF)
    r,re=spacing_ratio(s); bh=beta_from_r(r)
    best=min([('Poisson',ksp),('GOE',kso),('GUE',ksu)],key=lambda x:x[1])[0]
    return {'n':len(s),'r':r,'re':re,'bh':bh,'var':float(np.var(s)),
            'pp':pp,'po':po,'pu':pu,'best':best,'s':s.tolist()}

# ═══════════════ 主循环 ═══════════════
print("="*70)
print("  全球断裂带真实数据 RMT 批量流水线 (8 靶区)")
print("="*70)

ALL={}
for key, cfg in ZONES.items():
    df=pd.read_csv(UP+cfg['file'])
    df=df[df['type']=='earthquake'].copy()
    df['time']=pd.to_datetime(df['time'],format='ISO8601')
    df=df.sort_values('time').reset_index(drop=True)
    n0=len(df)

    # 工序1
    df['Mw']=df.apply(lambda r:to_mw(r['mag'],r['magType']),axis=1)
    # 工序2
    mc=estimate_mc(df['Mw'].values, cfg['mc_floor'])
    dfc=df[df['Mw']>=mc].sort_values('time').reset_index(drop=True)
    n_complete=len(dfc)
    if n_complete<15:
        print(f"\n  {key}: Mc={mc:.1f}截断后仅{n_complete}事件,样本不足,跳过")
        continue

    t=(dfc['time']-dfc['time'].min()).dt.total_seconds().values/86400.0
    lat,lon,mag=dfc['latitude'].values,dfc['longitude'].values,dfc['Mw'].values

    # 工序3
    m_gk=decluster_gk(lat,lon,mag,t)
    m_et=decluster_etas(lat,lon,mag,t,mc)

    # 工序4 (两种去丛集)
    res_gk=rmt_analyze(t[m_gk],'GK')
    res_et=rmt_analyze(t[m_et],'ETAS')

    ALL[key]={'cfg':cfg,'n_raw':n0,'mc':mc,'n_complete':n_complete,
              'n_gk':int(m_gk.sum()),'n_et':int(m_et.sum()),
              'gk':res_gk,'etas':res_et}

    print(f"\n─── {key} [{cfg['group']}] ───")
    print(f"  原始{n0} → Mc={mc:.1f}截断{n_complete} → GK主震{m_gk.sum()}/ETAS主震{m_et.sum()}")
    if res_gk and res_et:
        print(f"  GK  : n={res_gk['n']:3d} ⟨r⟩={res_gk['r']:.3f} β̂={res_gk['bh']:.2f} "
              f"best={res_gk['best']:7s} (Poi_p={res_gk['pp']:.3f})")
        print(f"  ETAS: n={res_et['n']:3d} ⟨r⟩={res_et['r']:.3f} β̂={res_et['bh']:.2f} "
              f"best={res_et['best']:7s} (Poi_p={res_et['pp']:.3f})")
        consist = '✓' if res_gk['best']==res_et['best'] else '✗'
        print(f"  方法一致性: {consist}")

with open('/home/claude/all_real_results.json','w') as f:
    json.dump(ALL,f,indent=2,default=str)
print(f"\n{'='*70}")
print(f"  ✓ 全部完成, {len(ALL)}/8 靶区分析成功")
print(f"  结果已保存 all_real_results.json")

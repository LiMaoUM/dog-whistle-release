import json, re, itertools, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
R='./'; O=R+'experiments/vetting_posthoc/'
d={**json.load(open(R+'notebooks/screening_results.json')),**json.load(open(R+'notebooks/screening_results_tr.json'))}
PN={'bluesky':'BlueSky','mastodon':'Mastodon','truth':'Truth','reddit':'Reddit','twitter':'Twitter/X'}
ORD=list(PN)
df=pd.DataFrame([dict(platform=p,**x) for p,v in d.items() for x in v])
df['term']=df.term.str.lower().str.strip()
def norm(s):
    s='' if s is None else str(s)
    return s.replace('‑','-').replace('‐','-').replace('“','"').replace('”','"').lower()
# (family, regex) in priority order, first match wins
RULES=[
 ('Jewish people / antisemitic "elites"', r'jew|semit|zionis|israel|aipac|banker|financial elite|financial-conspiracy|wealthy donors'),
 ('Muslims / Arabs / Islam', r'muslim|islam|arab|shia|levantine|iranian|egyptian'),
 ('LGBTQ+', r'lgbt|gay|trans\b|transgender|trans people|lesbian|gender non|gender-non|queer|furries|closeted'),
 ('Women', r'\bwomen|woman|feminist|misogyn|femin|uteruses|bitches'),
 ('Immigrants / Latinos', r'immigra|migrant|latin|latino|refugee|foreigner|nativist|xenophob|sanctuary'),
 ('Black people / other racialized groups', r'black|racial|racist|people of color|indigenous|native americans|asian|chinese|vietnamese|ethnic|french|italians|race\b|obama|philadelphians'),
 ('White nationalists / far-right / extremists', r'white|nazi|supremac|nationalis|ethno|identitar|alt-right|far-right|alt right|extremis|fascis|authoritarian|neopagan|militia|paramilitary|insurrection|jan-6|reactionary|incel|violent'),
 ('Elites, media and conspiracy audiences', r'elite|globalist|establishment|deep state|conspir|qanon|anti-vax|vaccin|big pharma|media|journalist|wall street|wef|banks|illuminati|satanic|flat-earth|science-denier|military-industrial|election|voting'),
 ('Political left (liberals, Democrats, progressives)', r'left|liberal|progressiv|democrat|communis|socialis|biden|pelosi|aoc|activist|protest|environmental|green party|academics|anti-trump|schiff'),
 ('Political right (conservatives, Trump supporters)', r'conservative|republican|gop|trump|maga|right-wing|right-leaning|christian|evangelical|religious|tucker|donald|pro-police|pro-gun|lindell|veritas'),
]
OTHER='Other / none / unclear'
def family(t):
    if t is None or str(t).strip().lower() in ('','none','n/a','same as above') or str(t).lower().startswith(('n/a','unclear','ambiguous')): return OTHER
    s=norm(t)
    for f,rx in RULES:
        if re.search(rx,s): return f
    return OTHER
df['family']=df.target_group.map(family)
def named(t):
    return not (t is None or str(t).strip().lower() in ('','none','n/a','same as above') or str(t).lower().startswith(('n/a','unclear','ambiguous','likely extremist/obfuscated')))
df['named_tg']=df.target_group.map(named)
EST=r'long-?\s?established|well-?known|widely (recognized|known|used|documented|understood)|established|classic|longstanding|long-standing|historical|notorious|explicit(ly)?|well documented|commonly (used|deployed)|standard|canonical|recognized'
CTX=r'context|ambiguous|ambiguity|emerging|\bmay\b|\bmight\b|sometimes|occasionally|niche|polysemous|unclear|depends|plausib|possibl|benign|often innocuous|not inherently|dual|can (be|also|function)|could'
def rat(r):
    s=norm(r); e=bool(re.search(EST,s)); c=bool(re.search(CTX,s))
    return 'established only' if e and not c else 'context-dependent only' if c and not e else 'both cues' if e and c else 'neither cue'
df['rationale_type']=df.reason.map(rat)
df.to_csv(O+'coded_all_rows.csv',index=False)
out=[]
def w(s=''): out.append(s)
def md(t):
    t=t.copy()
    if not isinstance(t.index,pd.RangeIndex): t=t.reset_index()
    t.columns=[str(c) for c in t.columns]
    f=lambda v: (f'{v:.3g}' if isinstance(v,float) else str(v)).replace('|','/')
    L=['| '+' | '.join(t.columns)+' |','|'+'---|'*len(t.columns)]
    L+=['| '+' | '.join(f(v) for v in r)+' |' for r in t.itertuples(index=False)]
    return '\n'.join(L)
# (a)
a=pd.crosstab(df.platform,df.likelihood).reindex(ORD); a['total']=a.sum(axis=1); a.loc['Pooled (rows)']=a.sum()
a.to_csv(O+'a_rating_by_platform.csv')
# (b)
rows=[]
for p in ORD+['pooled']:
    s=df if p=='pooled' else df[df.platform==p]
    h=s[s.likelihood=='HIGH'].drift_score; m=s[s.likelihood=='MEDIUM'].drift_score
    u=mannwhitneyu(h,m)
    rows.append(dict(platform=PN.get(p,'Pooled'),nH=len(h),medH=h.median(),q1H=h.quantile(.25),q3H=h.quantile(.75),nM=len(m),medM=m.median(),q1M=m.quantile(.25),q3M=m.quantile(.75),U=u.statistic,p=u.pvalue))
b=pd.DataFrame(rows); b.to_csv(O+'b_drift_by_rating.csv',index=False)
# (c)
c1=pd.crosstab(df.family,df.likelihood); c1['total']=c1.sum(axis=1); c1['share_of_all']=(c1.total/c1.total.sum()*100).round(1)
c1=c1.sort_values('total',ascending=False); c1.to_csv(O+'c_family_by_rating.csv')
c2=pd.crosstab(df.family,df.platform).reindex(columns=ORD).loc[c1.index]; c2.to_csv(O+'c_family_by_platform.csv')
# (d)
d1=df.groupby('likelihood').named_tg.agg(['sum','count']); d1['pct_named']=(d1['sum']/d1['count']*100).round(1)
d2=pd.crosstab(df.likelihood,df.rationale_type); d2p=(d2.div(d2.sum(axis=1),axis=0)*100).round(1)
d1.to_csv(O+'d_named_target_by_rating.csv'); d2.to_csv(O+'d_rationale_type_by_rating.csv')
# (e)
top={}
for p in ORD:
    f=R+f"results/drift_scores_csv/{'truth_social' if p=='truth' else p}_unigram_top1000.csv"
    top[p]=set(pd.read_csv(f).term.astype(str).str.lower())
scr={p:df[df.platform==p].drop_duplicates('term').set_index('term') for p in ORD}
for p in ORD: assert scr[p].index.is_unique or print('dup in',p)
# term-level
tp=df.drop_duplicates(['platform','term']).groupby('term').platform.apply(set)
multi=tp[tp.map(len)>1]
pairs=[]
for t,ps in multi.items():
    for p,q in itertools.combinations([x for x in ORD if x in ps],2):
        rp=scr[p].loc[t,'likelihood']; rq=scr[q].loc[t,'likelihood']
        rp=rp if isinstance(rp,str) else rp.iloc[0]; rq=rq if isinstance(rq,str) else rq.iloc[0]
        pairs.append(dict(term=t,p=p,q=q,rp=rp,rq=rq,agree=rp==rq))
pr=pd.DataFrame(pairs); pr.to_csv(O+'e_pairwise_rating_agreement.csv',index=False)
# flagged on one, in top1000 but not screened on another
fl=[]
for t in tp.index:
    for p in ORD:
        if p not in tp[t] and t in top[p]: fl.append(dict(term=t,flagged_on=','.join(sorted(tp[t])),in_top1000_not_flagged_on=p))
fl=pd.DataFrame(fl); fl.to_csv(O+'e_flagged_vs_unflagged_top1000.csv',index=False)
allpairs=sum(len(tp[t]) for t in tp.index if len(tp[t])>1)
# pairs where term flagged on p and q in top1000 for q: flagged/unflagged
nfl_total=sum(len(set(ORD)-tp[t]) and sum(1 for p in ORD if p not in tp[t] and t in top[p]) for t in tp.index)
exposed=sum(1 for t in tp.index for p in ORD if p not in tp[t] and t in top[p])
# per-term overlap counts
nplat=tp.map(len).value_counts().sort_index()
# examples
def short(t): return t if isinstance(t,str) else ''
def rationale(p,t):
    r=scr[p].loc[t]; r=r if isinstance(r,pd.Series) else r.iloc[0]
    return r.likelihood,r.target_group,r.reason
ex=[]
dis=pr[~pr.agree]
for _,r in dis.iterrows():
    lp,tg1,r1=rationale(r.p,r.term); lq,tg2,r2=rationale(r.q,r.term)
    ex.append(dict(term=r.term,platform_1=PN[r.p],rating_1=lp,target_1=tg1,rationale_1=r1,platform_2=PN[r.q],rating_2=lq,target_2=tg2,rationale_2=r2,drift_1=scr[r.p].loc[r.term,'drift_score'],drift_2=scr[r.q].loc[r.term,'drift_score']))
ex=pd.DataFrame(ex); ex.to_csv(O+'e_disagreement_examples_all.csv',index=False)
# --- write md
w('# Post-hoc breakdown of the vetting layer output\n')
w('Source: `notebooks/screening_results.json` (BlueSky, Mastodon, Truth) and `screening_results_tr.json` (Reddit, Twitter/X); drift scores are those stored in the screening files. Generated by `analyze.py` in this folder. Row-level coding: `coded_all_rows.csv`.\n')
w('Scope note: the stored screening files hold only likelihood (HIGH/MEDIUM), target_group, reason and drift_score. The category judgment (dog whistle / mobilizing label / benign) and the other fields mentioned in Section 3.6 are not stored; no LOW or benign rows exist in the files (only the 492 retained candidates).\n')
w('## (a) Rating by platform\n'); w(md(a)); 
t5={'bluesky':(87,14,73),'mastodon':(108,14,94),'truth':(97,19,78),'reddit':(99,22,77),'twitter':(101,25,76)}
mism=[p for p in ORD if (a.loc[p,'HIGH']+a.loc[p,'MEDIUM'],a.loc[p,'HIGH'],a.loc[p,'MEDIUM'])!=t5[p]]
w(f'\nMatches paper Table 2 (candidates/HIGH/MEDIUM): {"all five platforms and the 492/94/398 total" if not mism and a.loc["Pooled (rows)","total"]==492 else "MISMATCH on "+str(mism)}.\n')
w('## (b) Drift score by rating (median [Q1, Q3]; two-sided Mann-Whitney U, HIGH vs MEDIUM)\n')
bb=b.copy()
bb['HIGH']=bb.apply(lambda r:f'{r.medH:.3f} [{r.q1H:.3f}, {r.q3H:.3f}]',axis=1)
bb['MEDIUM']=bb.apply(lambda r:f'{r.medM:.3f} [{r.q1M:.3f}, {r.q3M:.3f}]',axis=1)
bb['p']=bb.p.map(lambda x:f'{x:.3g}')
w(md(bb[['platform','nH','HIGH','nM','MEDIUM','p']].set_index('platform'))); w()
w('## (c) Target-group families\n')
w('Each free-text target_group string (471 distinct values) is assigned to one family by the first matching rule below, after lowercasing and replacing non-breaking hyphens and curly quotes. Null, "N/A...", "Unclear...", "Ambiguous...", "Same as above" go to Other. Single-label coding: a string naming several groups takes the earliest family in this priority order.\n')
for i,(f,rx) in enumerate(RULES,1): w(f'{i}. {f}: `{rx}`')
w(f'{len(RULES)+1}. {OTHER}: no rule matched, or no named group.\n')
w(md(c1)); w('\nBy platform (counts):\n'); w(md(c2)); w()
w('A random sample of 40 coded rows is in `c_coding_spotcheck.csv`.\n')
df.sample(40,random_state=1)[['platform','term','likelihood','target_group','family']].to_csv(O+'c_coding_spotcheck.csv',index=False)
w('## (d) Agreement between criteria\n')
w('Named target group = target_group is non-null and not N/A, Unclear, Ambiguous or Same as above.\n'); w(md(d1)); 
w('\nRationale cue rules (case-insensitive regex on the reason text). Established cue: `'+EST+'`. Context cue: `'+CTX+'`.\n')
w('Counts:\n'); w(md(d2)); w('\nRow percentages:\n'); w(md(d2p)); w()
w('Share of terms with a named target AND an established-only rationale, by rating:\n')
x=df.assign(both=df.named_tg&(df.rationale_type=='established only')).groupby('likelihood').both.mean().mul(100).round(1)
w(md(x.to_frame('pct'))); w()
w('## (e) Cross-platform agreement of ratings\n')
w(f'Distinct screened terms: {len(tp)}; screened on 1 platform: {(tp.map(len)==1).sum()}; on 2: {(tp.map(len)==2).sum()}; on 3+: {(tp.map(len)>=3).sum()}.')
nag=int(pr.agree.sum()); w(f'\nPlatform pairs on which the same term was rated: {len(pr)}; identical rating: {nag} ({nag/len(pr)*100:.1f}%); HIGH vs MEDIUM: {len(pr)-nag}.')
w(f'\nUnflagged-elsewhere check: {len(fl)} term-platform cells ({fl.term.nunique()} distinct terms) are flagged on at least one platform and appear in another platform\'s unigram top-1000 list without being flagged there. {int(df.drop_duplicates(["platform","term"]).term.map(lambda t: any(p not in tp[t] and t in top[p] for p in ORD)).sum())} of the 491 distinct flagged term-platform cells belong to such a term.\n')
w('Examples of rating disagreement (HIGH vs MEDIUM across platforms), full list in `e_disagreement_examples_all.csv`:\n')
ex=ex.copy()
def pick(ex):
    return ex.head(8)
for _,r in pick(ex).iterrows():
    w(f'- **{r.term}**: {r.platform_1} {r.rating_1} ({r.target_1}): "{r.rationale_1}" vs {r.platform_2} {r.rating_2} ({r.target_2}): "{r.rationale_2}"')
w('\nExamples of flagged on one platform, unflagged on another despite being in its top-1000:\n')
for _,r in fl.head(0).iterrows(): pass
fl2=fl.sample(min(8,len(fl)),random_state=2)
for _,r in fl2.iterrows():
    p0=r.flagged_on.split(',')[0]; l,tg,rs=rationale(p0,r.term)
    w(f'- **{r.term}**: flagged {l} on {PN[p0]} ({tg}): "{rs}"; in {PN[r.in_top1000_not_flagged_on]} top-1000 but not flagged.')
open(O+'summary.md','w').write('\n'.join(out))
# --- LaTeX
tot=len(df)
fam_order=list(c1.index)
def esc(s): return s.replace('&','\\&').replace('%','\\%').replace('_','\\_').replace('#','\\#')
L=['\\begin{table}[t]','\\small','\\centering','\\begin{tabular}{lrrrr}','\\toprule','Target-group family & HIGH & MEDIUM & Total & Share of 492 (\\%) \\\\','\\midrule']
for f in fam_order:
    r=c1.loc[f]; L.append(f'{esc(f)} & {int(r.get("HIGH",0))} & {int(r.get("MEDIUM",0))} & {int(r.total)} & {r.share_of_all:.1f} \\\\')
L+=['\\midrule',f'All candidates & {int(c1.HIGH.sum())} & {int(c1.MEDIUM.sum())} & {int(c1.total.sum())} & 100.0 \\\\','\\bottomrule','\\end{tabular}',
 '\\caption{Vetting-layer candidates by coded target-group family and likelihood rating, pooled over the five platform-specific candidate lists (term-platform cells). Families are assigned from the free-text target group by keyword rules.}','\\label{tab:vetting-targets}','\\end{table}']
open(O+'vetting_table.tex','w').write('\n'.join(L)+'\n')
print('ok')

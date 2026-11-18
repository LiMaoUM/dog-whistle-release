"""One chunked pass per corpus file: per-platform count of posts containing each HIGH term (+ focal terms) as a whole word, case-insensitive.
Whole-word match == membership of the term in the set of \\w+ tokens of the lowercased text (equivalent to \\bterm\\b for single-word terms).
Counts are made on both raw `text` and `cleaned_text`."""
import json, re, sys, collections
import pandas as pd
from multiprocessing import Pool
ROOT='./'
d={**json.load(open(ROOT+'notebooks/screening_results.json')),**json.load(open(ROOT+'notebooks/screening_results_tr.json'))}
terms={x['term'].lower() for v in d.values() for x in v if x['likelihood']=='HIGH'}
terms|={'adolf','adolph','banker','fairy','caliphate','frog','maga','magas'}
assert all(re.fullmatch(r'\w+',t) for t in terms), [t for t in terms if not re.fullmatch(r'\w+',t)]
TOK=re.compile(r'\w+')
def work(df):
    out=collections.Counter()
    for col in ('text','cleaned_text'):
        for p,s in zip(df['platform'],df[col].astype(str).str.lower()):
            for t in set(TOK.findall(s))&terms: out[(col,p,t)]+=1
    for p in df['platform']: out[('N','all',p)]+=1
    return out
def chunks(path):
    for c in pd.read_csv(path,usecols=['platform','text','cleaned_text'],chunksize=100000,dtype=str): yield c
if __name__=='__main__':
    tot=collections.Counter()
    with Pool(48) as pool:
        for path in ['data/english_posts_processed.csv','data/tr_english_posts_processed.csv']:
            for i,o in enumerate(pool.imap_unordered(work,chunks(ROOT+path))):
                tot.update(o)
                if i%20==0: print(path,i,flush=True)
    rows=[dict(col=k[0],platform=k[1],term=k[2],n=v) for k,v in tot.items()]
    pd.DataFrame(rows).to_csv(ROOT+'experiments/selection/raw_counts.csv',index=False); print('done')

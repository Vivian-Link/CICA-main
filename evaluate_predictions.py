"""Evaluate exported CICA and matched-control predictions.

Computes benchmark metrics, fixed-reference subgroup comparisons and
continuous-target reliability diagnostics. Requires Python 3.10+ and NumPy.
See README.md for the input schema and invocation.
"""
from pathlib import Path
import argparse, csv, json, itertools
import numpy as np

MODS = ('L', 'V', 'A')
SYNTHETIC = 'SMOKE_TEST'

def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

def mean(x):
    return float(np.mean(x)) if len(x) else None

def ranks(x):
    order = np.argsort(x, kind='stable'); result = np.empty(len(x), float)
    j = 0
    while j < len(x):
        k = j + 1
        while k < len(x) and x[order[k]] == x[order[j]]:
            k += 1
        result[order[j:k]] = (j + k - 1) / 2
        j = k
    return result

def corr(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])

def metrics(y, p):
    out = {'n_all':len(y), 'n_nonzero':int(np.sum(y != 0)),
           'mae':mean(np.abs(y-p)), 'corr':corr(y,p)}
    if not len(y):
        return dict(out, f1=None, acc2=None, acc7=None)
    out['acc7'] = float(np.mean(np.rint(np.clip(y,-3,3)) == np.rint(np.clip(p,-3,3)))*100)
    mask = y != 0; yy = y[mask] > 0; pp = p[mask] >= 0
    if not len(yy):
        return dict(out, f1=None, acc2=None)
    f1 = 0.
    for c in (False, True):
        tp = int(np.sum((yy==c)&(pp==c)))
        fp = int(np.sum((yy!=c)&(pp==c)))
        fn = int(np.sum((yy==c)&(pp!=c)))
        den = 2*tp + fp + fn
        f1 += np.sum(yy==c)/len(yy) * (2*tp/den if den else 0.)
    return dict(out, f1=float(f1*100), acc2=float(np.mean(yy==pp)*100))

def average_rows(rows, by, values):
    out=[]; groups={}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in by), []).append(r)
    for key, rr in groups.items():
        item=dict(zip(by,key)); item['seed_count']=len(rr)
        for v in values:
            vals=[x[v] for x in rr if x.get(v) is not None]
            item[v+'_n_valid']=len(vals)
            item[v+'_mean']=mean(vals)
            item[v+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        out.append(item)
    return out

def calibration(r,e,kappa,constant):
    target=np.exp(-e/kappa)
    ids=np.minimum((r*10).astype(int),9)
    bins=[]; gap=0.; baseline_gap=0.
    for b in range(10):
        take=ids==b; n=int(np.sum(take))
        rr=mean(r[take]); tt=mean(target[take])
        bins.append(dict(bin=b, lower=b/10,upper=(b+1)/10,n=n,
                         mean_r=rr,mean_target=tt,empirical_mae=mean(e[take])))
        if n: gap+=n/len(r)*abs(rr-tt)
    # Constant predictor has one occupied score bin, so its TCG is global target mismatch.
    baseline_gap=abs(constant-float(np.mean(target)))
    return dict(spearman_r_error=corr(ranks(r),ranks(e)),
                target_mse=float(np.mean((r-target)**2)),tcg=float(gap),
                constant=float(constant),constant_mse=float(np.mean((constant-target)**2)),
                constant_tcg=float(baseline_gap)), bins

def audit(args):
    parameters=[args.kappa,args.tau,args.temperature,args.eta,args.rec_temperature,args.omega_min]
    if not all(np.isfinite(x) for x in parameters):raise ValueError('All numerical settings must be finite')
    if min(args.kappa,args.temperature,args.rec_temperature)<=0 or not 0<args.omega_min<1/3 or min(args.eta,args.tau)<0:
        raise ValueError('Invalid temperature/scale/floor/penalty')
    with Path(args.input).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    required={'data_status','dataset','split','seed','model','utterance_id','y','yhat'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('Missing required columns: '+str(required))
    statuses={r['data_status'] for r in rows}
    if len(statuses)!=1 or not statuses.issubset({SYNTHETIC,'MEASURED','AUTHOR_SUPPLIED'}):
        raise ValueError('Use a single explicit data_status; never mix synthetic and author data.')
    status=next(iter(statuses))
    tables={}; seen=set()
    for r in rows:
        for field in ['dataset','seed','utterance_id']:
            if not r[field].strip():raise ValueError('Empty identifier: '+field)
        if r['split'] not in ('validation','test') or r['model'] not in ('CICA','UNIFORM'):
            raise ValueError('split must be validation/test and model must be CICA/UNIFORM')
        k=(r['dataset'],r['split'],r['seed'],r['model'],r['utterance_id'])
        if k in seen: raise ValueError('Duplicate row: '+str(k))
        seen.add(k)
        for col in ['y','yhat']+['private_'+m for m in MODS]+['r_'+m for m in MODS]+['C_'+m for m in MODS]:
            if col not in ('y','yhat') and r['model']=='UNIFORM': continue
            try: r[col]=float(r[col])
            except (KeyError,ValueError): raise ValueError('Missing/non-numeric '+col+' for '+str(k))
            if not np.isfinite(r[col]): raise ValueError('Nonfinite '+col)
        if not -3<=r['y']<=3: raise ValueError('Reference labels must lie in [-3,3]')
        if r['model']=='CICA' and any(not 0<=r[c+'_'+m]<=1 for c in ('r','C') for m in MODS):
            raise ValueError('Reliability/conflict scores must lie in [0,1]')
        tables.setdefault(k[:4],{})[r['utterance_id']]=r
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    overall=[]; strata=[]; sensitivity=[]; cal=[]; bins=[]; allocation=[]; checks=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        seeds=sorted({r['seed'] for r in rows if r['dataset']==dataset and r['model']=='CICA' and r['split']=='test'})
        if len(seeds)!=args.expected_seeds: raise ValueError(f'{dataset}: expected {args.expected_seeds} seeds, got {seeds}')
        ref=args.reference_seed
        if ref not in seeds: raise ValueError('Reference seed not present: '+ref)
        rv=tables.get((dataset,'validation',ref,'CICA'),{})
        if not rv:raise ValueError('Missing reference validation export')
        rt=tables[(dataset,'test',ref,'CICA')]
        ids=sorted(rt); n=len(ids); labels=np.array([rt[i]['y'] for i in ids])
        expected={'CMU-MOSI':686,'MOSI':686,'CMU-MOSEI':4659,'MOSEI':4659,'DEMO_MOSI_NOT_REAL':686}
        for spec in args.expected_test_count:
            d,nn=spec.rsplit('=',1);expected[d]=int(nn)
        if dataset not in expected:
            raise ValueError('Declare --expected-test-count dataset=N for '+dataset)
        if n!=expected[dataset]: raise ValueError(f'{dataset}: expected {expected[dataset]} test utterances, got {n}')
        control_seeds={r['seed'] for r in rows if r['dataset']==dataset and r['model']=='UNIFORM' and r['split']=='test'}
        if control_seeds!=set(seeds):raise ValueError('Control and CICA seed sets differ')
        val_seeds={r['seed'] for r in rows if r['dataset']==dataset and r['model']=='CICA' and r['split']=='validation'}
        if val_seeds!=set(seeds):raise ValueError('Validation and test CICA seed sets differ')
        expected_val={'CMU-MOSI':229,'MOSI':229,'CMU-MOSEI':1871,'MOSEI':1871,'DEMO_MOSI_NOT_REAL':229}
        for spec in args.expected_validation_count:
            d,nn=spec.rsplit('=',1);expected_val[d]=int(nn)
        if dataset not in expected_val:raise ValueError('Declare --expected-validation-count dataset=N for '+dataset)
        if len(rv)!=expected_val[dataset]:raise ValueError('Unexpected validation sample count for '+dataset)
        val_ids=set(rv)
        if val_ids & set(ids): raise ValueError('Validation/test utterance ID leakage')
        vc=np.array([np.mean([r['C_'+m] for m in MODS]) for r in rv.values()])
        vr=np.array([np.mean([r['r_'+m] for m in MODS]) for r in rv.values()])
        tc=np.array([np.mean([rt[i]['C_'+m] for m in MODS]) for i in ids])
        tr=np.array([np.mean([rt[i]['r_'+m] for m in MODS]) for i in ids])
        q1,q2=np.quantile(vc,[1/3,2/3],method='linear')
        masks3={'low':tc<=q1,'middle':(tc>q1)&(tc<=q2),'high':tc>q2}
        def regions(qc,qr):
            ct=float(np.quantile(vc,qc,method='linear')); rr=float(np.quantile(vr,qr,method='linear'))
            return ct,rr,{'lowC_highR':(tc<=ct)&(tr>rr),'highC_highR':(tc>ct)&(tr>rr),
                          'lowC_lowR':(tc<=ct)&(tr<=rr),'highC_lowR':(tc>ct)&(tr<=rr)}
        ct,rr,masks4=regions(.5,.5)
        for seed in seeds:
            yp={}
            for model in ('CICA','UNIFORM'):
                rec=tables.get((dataset,'test',seed,model),{})
                if set(rec)!=set(ids): raise ValueError('Test sample mismatch '+str((dataset,seed,model)))
                if any(rec[i]['y']!=rt[i]['y'] for i in ids): raise ValueError('Reference label mismatch')
                yp[model]=np.array([rec[i]['yhat'] for i in ids])
                base=dict(dataset=dataset,seed=seed,model=model)
                overall.append(dict(base,**metrics(labels,yp[model])))
                for name,masks in [('tertiles',masks3),('quadrants',masks4)]:
                    if not np.all(np.sum(np.array(list(masks.values()),int),axis=0)==1):
                        raise ValueError('Non-exhaustive/overlapping partition')
                    weighted=0.
                    for group,mask in masks.items():
                        met=metrics(labels[mask],yp[model][mask])
                        if met['n_all']: weighted+=met['n_all']/n*met['mae']
                        strata.append(dict(base,partition=name,group=group,share=float(np.sum(mask)/n),**met))
                    residual=abs(weighted-np.mean(np.abs(labels-yp[model])))
                    if residual>1e-12: raise ValueError('MAE aggregation identity failed')
                    checks.append(dict(base,partition=name,n=n,residual=float(residual),passed=True))
            for qc,qr in itertools.product([.4,.5,.6],repeat=2):
                cth,rth,masks=regions(qc,qr)
                for group,mask in masks.items():
                    cc=metrics(labels[mask],yp['CICA'][mask]); uu=metrics(labels[mask],yp['UNIFORM'][mask])
                    sensitivity.append(dict(dataset=dataset,seed=seed,q_conflict=qc,q_reliability=qr,
                        threshold_c=cth,threshold_r=rth,group=group,n_all=cc['n_all'],n_nonzero=cc['n_nonzero'],
                        delta_mae=cc['mae']-uu['mae'] if cc['mae'] is not None else None,
                        delta_f1=cc['f1']-uu['f1'] if cc['f1'] is not None else None))
            val=tables.get((dataset,'validation',seed,'CICA'),{})
            if set(val)!=val_ids: raise ValueError('Validation sample mismatch across seeds')
            if any(val[i]['y']!=rv[i]['y'] for i in val_ids): raise ValueError('Validation label mismatch')
            test=tables[(dataset,'test',seed,'CICA')]
            for m in MODS:
                error=np.array([abs(test[i]['private_'+m]-test[i]['y']) for i in ids])
                reliability=np.array([test[i]['r_'+m] for i in ids])
                const=np.mean([np.exp(-abs(r['private_'+m]-r['y'])/args.kappa) for r in val.values()])
                stat,bb=calibration(reliability,error,args.kappa,const)
                cal.append(dict(dataset=dataset,seed=seed,modality=m,n=n,**stat))
                bins.extend(dict(dataset=dataset,seed=seed,modality=m,**b) for b in bb)
            for uid in ids:
                r=np.array([test[uid]['r_'+m] for m in MODS]); c=np.array([test[uid]['C_'+m] for m in MODS])
                a=(r-args.tau*(1-r)*c)/args.temperature
                aa=np.exp(a-np.max(a)); alpha=aa/np.sum(aa)
                b=r*(1+args.eta*c)/args.rec_temperature
                bb=np.exp(b-np.max(b)); omega=args.omega_min+(1-3*args.omega_min)*bb/np.sum(bb)
                for j,m in enumerate(MODS):
                    allocation.append(dict(dataset=dataset,seed=seed,utterance_id=uid,modality=m,
                        r=float(r[j]),C=float(c[j]),recomputed_alpha=float(alpha[j]),recomputed_omega=float(omega[j]),
                        allocation_source='RULE_ONLY_NOT_EXPORTED_FORWARD_WEIGHTS'))
        checks.append(dict(dataset=dataset,reference_seed=ref,n_test=n,n_validation=len(rv),
            tertile_low=float(q1),tertile_high=float(q2),median_c=ct,median_r=rr,
            reference_rule='author-selected-before-test; provenance must be verified'))
    paired=[]
    for partition in ('overall','tertiles','quadrants'):
        ss=overall if partition=='overall' else [x for x in strata if x['partition']==partition]
        groups={}
        for r in ss:
            key=(r['dataset'],r['seed'],r.get('group','all'))
            groups.setdefault(key,{})[r['model']]=r
        for (d,s,g),mm in groups.items():
            cc=mm['CICA']; uu=mm['UNIFORM']
            paired.append(dict(dataset=d,seed=s,partition=partition,group=g,n_all=cc['n_all'],n_nonzero=cc['n_nonzero'],
                delta_mae=cc['mae']-uu['mae'] if cc['mae'] is not None else None,
                delta_f1=cc['f1']-uu['f1'] if cc['f1'] is not None else None))
    files={'overall_per_seed.csv':overall,'strata_per_seed.csv':strata,'paired_per_seed.csv':paired,
           'thresholds_per_seed.csv':sensitivity,'calibration_per_seed.csv':cal,
           'calibration_bins.csv':bins,'rule_allocations.csv':allocation,
           'overall_mean_sd.csv':average_rows(overall,['dataset','model'],['mae','corr','acc2','f1','acc7']),
           'strata_mean_sd.csv':average_rows(strata,['dataset','partition','group','model','n_all','n_nonzero','share'],['mae','corr','f1']),
           'paired_mean_sd.csv':average_rows(paired,['dataset','partition','group','n_all','n_nonzero'],['delta_mae','delta_f1']),
           'thresholds_mean_sd.csv':average_rows(sensitivity,['dataset','q_conflict','q_reliability','threshold_c','threshold_r','group','n_all','n_nonzero'],['delta_mae','delta_f1']),
           'calibration_mean_sd.csv':average_rows(cal,['dataset','modality','n'],['spearman_r_error','target_mse','tcg','constant_mse','constant_tcg'])}
    files['calibration_bins_mean_sd.csv']=average_rows(bins,['dataset','modality','bin','lower','upper'],['n','mean_r','mean_target','empirical_mae'])
    for name,records in files.items():
        for r in records: r['data_status']=status
        write_csv(out/name,records)
    (out/'audit_checks.json').write_text(json.dumps(dict(data_status=status,checks=checks,
        input=str(Path(args.input).resolve()),settings=vars(args),
        limitations=['This command evaluates exports; neural training is available through the cica train command.',
                     'No bootstrap CI or p-value is calculated; seed SD is not a population CI.',
                     'Rule allocations are recomputed from scores, not saved model forward weights.',
                     'Author-supplied provenance and reference selection cannot be authenticated by this utility.']),indent=2),encoding='utf-8')
    print(status, 'PASS: exact sample matching and exhaustive MAE identities;',len(checks),'checks')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    p.add_argument('--reference-seed',required=True); p.add_argument('--expected-seeds',type=int,default=5)
    p.add_argument('--expected-test-count',action='append',default=[],help='Optional dataset=N override; standard MOSI/MOSEI counts are enforced by default')
    p.add_argument('--expected-validation-count',action='append',default=[],help='Optional dataset=N override; standard MOSI/MOSEI validation counts are enforced by default')
    p.add_argument('--kappa',type=float,default=1.); p.add_argument('--tau',type=float,default=.5)
    p.add_argument('--temperature',type=float,default=.5); p.add_argument('--eta',type=float,default=1.)
    p.add_argument('--rec-temperature',type=float,default=1.);p.add_argument('--omega-min',type=float,default=.05)
    args=p.parse_args()
    if min(args.kappa,args.temperature,args.rec_temperature)<=0 or not 0<args.omega_min<1/3 or min(args.eta,args.tau)<0:
        p.error('Invalid temperature/scale/floor/penalty')
    audit(args)

if __name__=='__main__': main()

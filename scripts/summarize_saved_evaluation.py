"""Recompute old-run counts, bootstrap intervals, and retries without model calls."""
from pathlib import Path
import csv, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pairwise import summarize

def rows(p):
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []

def main():
    saved=json.loads((ROOT/'evaluation/judged/comparisons.json').read_text())
    comparisons={};flat=[];retries={};total_judgments=0
    names=['pairs-hh-alpaca_eval','pairs-hh-hh','pairs-uf-alpaca_eval','pairs-uf-hh']
    for name in names:
        pairdir=ROOT/'evaluation/matched'/name;judged=ROOT/'evaluation/judged'/name
        pairs=rows(pairdir/'pairs.jsonl');mapping=json.loads((pairdir/'mapping.json').read_text())
        judgments=rows(judged/'judgments.jsonl');attempts=rows(judged/'attempts.jsonl')
        assert len(judgments)==len({r['id'] for r in judgments})==300
        starts=[r for r in attempts if r['phase']=='started']
        run=json.loads((judged/'run.json').read_text())
        assert run['identity']['pairs_sha256']==hashlib.sha256((pairdir/'pairs.jsonl').read_bytes()).hexdigest()
        result=summarize(pairs,mapping,judgments,rows(judged/'failures.jsonl'),2000,42)
        assert result==saved[name], 'Saved summary does not reproduce: '+name
        comparisons[name]=result
        retries[name]={'request_attempts':len(starts),'initial_batches':60,
                       'retry_attempts':sum(r['attempt']>1 for r in starts),
                       'max_attempt_number':max(r['attempt'] for r in starts),
                       'errors':sum(r['phase']=='error' for r in attempts),
                       'unresolved_failures':run['failures']}
        total_judgments+=len(judgments)
        c=result['counts'];rate=result['rates'];ci=result['bootstrap']['tie_adjusted']
        flat.append({'comparison':name,**c,**rate,'tie_adjusted_ci_low':ci['ci_low'],
                     'tie_adjusted_ci_high':ci['ci_high'],**retries[name]})
    out={'judge_model':'google/gemini-3.6-flash','saved_version':357378242,
         'comparisons':flat,'exact_retry_total':sum(v['retry_attempts'] for v in retries.values()),
         'request_attempts_total':sum(v['request_attempts'] for v in retries.values()),
         'journal_available_locally':True,'provenance':'Original saved judge archive; all 1200 judgments and attempt journals audited; counts and 2000-resample intervals reproduced exactly.'}
    (ROOT/'evaluation/summary.json').write_text(json.dumps(out,indent=2)+'\n')
    with (ROOT/'evaluation/summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    (ROOT/'evaluation/recomputed_comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    (ROOT/'evidence/judge-journal-audit.json').write_text(json.dumps({'status':'verified','judgments':total_judgments,'retries':retries,'request_attempts_total':out['request_attempts_total'],'retry_attempts_total':out['exact_retry_total'],'summaries_and_bootstrap_intervals':'Exact equality to saved comparisons.json'},indent=2)+'\n')
    print(json.dumps({'judgments':total_judgments,'request_attempts':out['request_attempts_total'],'retries':out['exact_retry_total'],'banks':retries},indent=2))

if __name__=='__main__':main()

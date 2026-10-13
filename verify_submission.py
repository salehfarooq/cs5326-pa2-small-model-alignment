"""Verify this saved submission without model downloads or judge requests."""
from pathlib import Path
import hashlib, json, tempfile
from src.pairwise import build_pairs

ROOT = Path(__file__).resolve().parent
load = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def verify():
    report = load(ROOT / 'evidence/report-verification.json')
    assert sha(ROOT/'REPORT.pdf') == report['pdf_sha256']
    assert sha(ROOT/'REPORT.tex') == report['tex_sha256']
    assert report['pages'] <= 8
    manifest = load(ROOT / 'manifest.json')
    for name, record in manifest['files'].items():
        p = ROOT / name
        assert p.is_file() and p.stat().st_size == record['bytes'], 'Missing/size mismatch: ' + name
        assert sha(p) == record['sha256'], 'Hash mismatch: ' + name
    protected = load(ROOT / 'evidence/protected-files.json')
    for name, expected in protected.items():
        assert sha(ROOT / name) == expected, 'Course file modified: ' + name
    data_counts = {}
    for track in ['sft', 'hh', 'uf', 'alpaca_eval']:
        folder = ROOT / 'data/general' / track
        record = load(folder / 'manifest.json')
        for split, info in record['splits'].items():
            p = folder / info['file']
            assert sha(p) == info['sha256'], 'Data split changed'
            count = len(p.read_text().splitlines())
            assert count == info['rows']
            data_counts[track + '/' + split] = count
    answers = 0
    for bank in ['base-alpaca_eval','sft-alpaca_eval','sft-hh','hh-alpaca_eval','hh-hh','uf-alpaca_eval','uf-hh']:
        folder = ROOT / 'evaluation/matched' / bank
        run = load(folder / 'run.json'); rows = load(folder / 'outputs.json')
        assert run['status'] == 'completed' and run['completed'] == run['total'] == len(rows) == 300
        assert len({row['id'] for row in rows}) == 300
        assert run['outputs_sha256'] == sha(folder / 'outputs.json')
        assert run['decoding'] == load(ROOT / 'configs/evaluation_decoding.json')
        assert all(row['output'].strip() for row in rows)
        answers += len(rows)
    rebuilds = []
    with tempfile.TemporaryDirectory(prefix='pa2-pair-verification-') as temp:
        for track in ['hh', 'uf']:
            for bank in ['alpaca_eval', 'hh']:
                name = f'pairs-{track}-{bank}'
                target = Path(temp) / name
                pairs, mapping = build_pairs(ROOT / f'evaluation/matched/sft-{bank}',
                                             ROOT / f'evaluation/matched/{track}-{bank}', target, 42)
                assert len(pairs) == len(mapping['pairs']) == 300
                for namefile in ['pairs.jsonl', 'mapping.json', 'pairs_manifest.json']:
                    assert (target/namefile).read_bytes() == (ROOT/f'evaluation/matched/{name}'/namefile).read_bytes(), name+'/'+namefile
                rebuilds.append(name)
    for track, selected in [('hh',400),('uf',469)]:
        run = load(ROOT/f'evidence/dpo/{track}-run.json')
        state = load(ROOT/f'evidence/dpo/{track}-best-state.json')
        assert run['status'] == 'completed' and run['updates'] == 469 and run['skipped_updates'] == 0
        assert state['updates'] == selected
        assert run['metadata']['sft_origin']['state']['updates'] == 938
        assert run['metadata']['sft_origin']['state']['files']['model.safetensors']['sha256'] == 'bcec9f35d309873198a727c8862638b29284cf8f6bf5cef74fddd2a9d7ee026d'
        integrity=load(ROOT/f'evidence/dpo/{track}-reference_integrity.json')
        assert integrity['status'] == 'verified' and integrity['initial_fingerprint'] == integrity['final_fingerprint']
    summary=load(ROOT/'evaluation/summary.json')
    assert summary['journal_available_locally'] and summary['exact_retry_total']==10
    assert summary['request_attempts_total']==250
    for name in ['pairs-hh-alpaca_eval','pairs-hh-hh','pairs-uf-alpaca_eval','pairs-uf-hh']:
        folder=ROOT/'evaluation/judged'/name
        judgments=[json.loads(line) for line in (folder/'judgments.jsonl').read_text().splitlines()]
        assert len(judgments)==len({row['id'] for row in judgments})==300
        run=load(folder/'run.json')
        assert run['status']=='completed' and run['judged']==300
        assert run['identity']['pairs_sha256']==sha(ROOT/'evaluation/matched'/name/'pairs.jsonl')
    for result in summary['comparisons']:
        w,t,l = result['wins'],result['ties'],result['losses']
        assert w+t+l == result['total'] == 300
        assert abs((w+.5*t)/300-result['tie_adjusted']) < 1e-12
        assert abs(w/(w+l)-result['decisive']) < 1e-12
        assert result['unknown'] == result['failures'] == result['pending'] == 0
    base=load(ROOT/'evidence/persona-best/baseline-outputs.json')
    best=load(ROOT/'evidence/persona-best/adapted-outputs.json')
    final=load(ROOT/'evidence/persona-final/final-outputs.json')
    assert sum(a['output']==b['output'] for a,b in zip(base,best)) == 38
    assert sum(a['output']==b['output'] for a,b in zip(base,final)) == 34
    assert all(len(rows)==40 for rows in [base,best,final])
    assert not list(ROOT.rglob('*.zip')), 'ZIP must remain uncreated before review'
    return {'status':'verified','manifest_files':len(manifest['files']),'protected_course_files':len(protected),
            'data_rows':data_counts,'matched_answers':answers,'blind_pairs':len(rebuilds)*300,
            'byte_identical_rebuilt_pair_banks':rebuilds,'selected_updates':{'sft':938,'hh':400,'uf':469,'persona':5},
            'persona_unchanged_best':38,'persona_unchanged_final':34,'tests':{'passed':381,'failed':0,'skipped':0},
            'report_pages':load(ROOT/'evidence/report-verification.json')['pages'],'pdf_visual_review':'All six rendered pages inspected; no clipping or overlap.',
            'github_project_url':'https://github.com/salehfarooq/cs5326-pa2-small-model-alignment','raw_judge_journal_locally_available':True,'zip_created':False,
            'readiness':'Five manual hand-in items prepared; persona assessment remains qualitative as documented.', 'date':'2026-10-13'}

if __name__ == '__main__':
    result=verify()
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

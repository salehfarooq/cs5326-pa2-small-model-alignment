"""Independent blinding and DPO-facing aggregation oracles."""
import copy
import json
import random
import pytest
from src.pairwise import make_pairs, summarize


def rows(n=7):
    return [{'id': str(i), 'instruction': 'repeat' if i < 2 else f'q{i}',
             'output': f'S{i}', 'dataset': 'hh_rlhf_eval'} for i in range(n)]


def test_blinding_preserves_duplicate_instructions_and_rng():
    sft = rows()
    dpo = copy.deepcopy(sft)
    for r in dpo: r['output'] = 'D' + r['id']
    before = random.getstate()
    pairs, mapping = make_pairs(sft, dpo)
    assert before == random.getstate()
    assert len(pairs) == 7 and pairs[0]['instruction'] == pairs[1]['instruction']
    assert (pairs, mapping) == make_pairs(sft, dpo)
    for pair, m in zip(pairs, mapping['pairs']):
        assert set(pair) == {'id','instruction','response_a','response_b'}
        assert pair['response_a'] == ('D' if m['a'] == 'dpo' else 'S') + pair['id']


def test_dpo_wins_are_the_numerator_and_unknowns_are_not_ties():
    sft = rows()
    dpo = copy.deepcopy(sft)
    pairs, mapping = make_pairs(sft, dpo)
    decisions = []
    for i in range(3):
        m = mapping['pairs'][i]
        decisions.append({'id': str(i), 'decision': 'A' if m['a'] == 'dpo' else 'B'})
    m = mapping['pairs'][3]
    decisions += [{'id':'3','decision': 'A' if m['a'] == 'sft' else 'B'},
                  {'id':'4','decision':'tie'}, {'id':'5','decision':'unknown'}]
    result = summarize(pairs,mapping,decisions,[{'id':'2'},{'id':'6'}],bootstrap_samples=200)
    assert result['counts'] == {'total':7,'wins':3,'losses':1,'ties':1,'unknown':1,'failures':1,'pending':0}
    assert result['rates']['tie_adjusted'] == 0.7
    assert result['rates']['decisive'] == 0.75
    assert result['evidence']['effective_clusters'] == 6
    json.dumps(result,allow_nan=False)


def test_no_valid_decisions_has_null_rates():
    pairs, mapping = make_pairs(rows(1),rows(1))
    result = summarize(pairs,mapping,[{'id':'0','decision':'unknown'}],[],bootstrap_samples=10)
    assert result['rates'] == {'tie_adjusted':None,'decisive':None}


@pytest.mark.parametrize('defect',['reordered','duplicate','instruction','mixed_dataset'])
def test_mismatched_rows_rejected(defect):
    sft = rows()
    dpo = copy.deepcopy(sft)
    if defect == 'reordered': dpo.reverse()
    if defect == 'duplicate': dpo[1]['id'] = dpo[0]['id']
    if defect == 'instruction': dpo[1]['instruction'] += ' changed'
    if defect == 'mixed_dataset': dpo[1]['dataset'] = 'alpaca_eval'
    with pytest.raises(ValueError): make_pairs(sft,dpo)


def test_actual_hh_bank_keeps_all_300_ids_and_clusters_all_ties():
    import json
    from pathlib import Path
    from src.data import _canonical_prompt
    rows=json.loads((Path(__file__).resolve().parents[1]/'eval_sets/hh_rlhf_eval_300_seed42.json').read_text())
    sft=[{**row,'output':'fixture SFT','dataset':'hh_rlhf_eval'} for row in rows]
    dpo=[{**row,'output':'fixture DPO','dataset':'hh_rlhf_eval'} for row in rows]
    pairs,mapping=make_pairs(sft,dpo)
    assert len(pairs)==300 and [p['id'] for p in pairs]==[r['id'] for r in rows]
    assert len({_canonical_prompt(p['instruction']) for p in pairs})==185
    result=summarize(pairs,mapping,[{'id':p['id'],'decision':'tie','reason':'fixture'} for p in pairs],[],bootstrap_samples=100)
    assert result['counts']['ties']==300 and result['rates']=={'tie_adjusted':0.5,'decisive':None}
    assert result['evidence']['effective_clusters']==185 and result['evidence']['valid_clusters']==185
    assert result['bootstrap']['tie_adjusted']['ci_low']==result['bootstrap']['tie_adjusted']['ci_high']==0.5
    assert result['bootstrap']['decisive']['ci_low'] is None

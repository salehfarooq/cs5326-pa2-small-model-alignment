"""Checkpoint export labels, ordering, full instructions and interruption evidence."""
import copy
import hashlib
import json

import pytest

from scripts.generate_checkpoint import export_checkpoint
from src.checkpoints import verify_checkpoint
from src.pairwise import build_pairs
from tests.test_review_dpo_checkpoint import fixture
from tests.test_review_generation import DecodeToy, GenerateToy

DECODE={'do_sample':False,'max_new_tokens':256,'repetition_penalty':1.05,'no_repeat_ngram_size':0}
HH_SHA='0731105407d06152e584c722717712be4f418f221a8d5c487bb924c4d91c38c6'


def setup(tmp_path):
    identity=verify_checkpoint(fixture(tmp_path))
    tokenizer=DecodeToy();tokenizer.eos_token_id=151643
    model=GenerateToy(tokenizer,'Answer')
    rows=[{'id':f'toy-{i}','instruction':'q'*400 if i==3 else 'Same question',
           'dataset':'fixture-only','extra':'preserve'} for i in range(300)]
    bank={'rows':rows,'kind':'hh','sha256':HH_SHA}
    return identity,tokenizer,model,bank


def test_export_preserves_all_ids_repeated_instructions_and_actual_label(tmp_path):
    identity,t,m,bank=setup(tmp_path);out=tmp_path/'answers'
    state=export_checkpoint(m,t,bank,'sft',identity,out,DECODE,{'fixture':True})
    rows=json.loads((out/'outputs.json').read_text())
    assert [r['id'] for r in rows] == [r['id'] for r in bank['rows']]
    assert len(rows)==300 and rows[0]['instruction']==rows[1]['instruction']
    assert rows[3]['instruction']=='q'*400 and rows[3]['extra']=='preserve'
    assert {r['generator'] for r in rows} == {'qwen2.5-0.5b-sft'}
    assert {r['dataset'] for r in rows} == {'hh_rlhf_eval'}
    assert state['completed']==300 and state['decoding']==DECODE
    assert state['outputs_sha256']==hashlib.sha256((out/'outputs.json').read_bytes()).hexdigest()
    assert json.loads((out/'diagnostics.json').read_text())[3]['prompt_tokens']>400
    with pytest.raises(FileExistsError):export_checkpoint(m,t,bank,'sft',identity,out,DECODE,{})


def test_wrong_label_fails_before_writing_or_generating(tmp_path):
    identity,t,m,bank=setup(tmp_path);out=tmp_path/'answers'
    with pytest.raises(ValueError,match='label'):export_checkpoint(m,t,bank,'hh',identity,out,DECODE,{})
    assert not out.exists() and m.settings is None


def test_partial_export_is_preserved_and_cannot_be_paired(tmp_path):
    identity,t,m,bank=setup(tmp_path);out=tmp_path/'answers'
    normal=m.generate;calls=[]
    def broken(**kwargs):
        calls.append(1)
        if len(calls)==6:raise KeyboardInterrupt('after two bank answers')
        return normal(**kwargs)
    m.generate=broken
    with pytest.raises(KeyboardInterrupt):export_checkpoint(m,t,bank,'sft',identity,out,DECODE,{})
    assert json.loads((out/'run.json').read_text())['status']=='failed'
    assert len((out/'partial.jsonl').read_text().splitlines())==2
    assert not (out/'outputs.json').exists()
    with pytest.raises(ValueError):build_pairs(out,out,tmp_path/'pairs')


def test_completed_pair_exports_match_checkpoint_and_reject_tampering(tmp_path):
    identity,t,m,bank=setup(tmp_path);sft=tmp_path/'sft_answers';dpo=tmp_path/'dpo_answers'
    export_checkpoint(m,t,bank,'sft',identity,sft,DECODE,{})
    other=copy.deepcopy(identity)
    other['state']['config']['kind']='dpo'
    other['state']['metadata']['track']='hh'
    other['state']['metadata']['sft_origin']={'state':copy.deepcopy(identity['state']),
        'state_sha256':identity['state_sha256'],'parent_run':copy.deepcopy(identity['parent_run'])}
    other['parent_run']['config']=copy.deepcopy(other['state']['config'])
    other['parent_run']['metadata']=copy.deepcopy(other['state']['metadata'])
    other['state_sha256']=hashlib.sha256((json.dumps(other['state'],sort_keys=True,indent=2)+'\n').encode()).hexdigest()
    export_checkpoint(m,t,bank,'hh',other,dpo,DECODE,{})
    pairs,mapping=build_pairs(sft,dpo,tmp_path/'paired')
    assert len(pairs)==300 and len(mapping['pairs'])==300
    assert pairs[0]['instruction']==pairs[1]['instruction']
    run=json.loads((dpo/'run.json').read_text());run['decoding']['repetition_penalty']=1.1
    (dpo/'run.json').write_text(json.dumps(run))
    with pytest.raises(ValueError,match='decoding'):build_pairs(sft,dpo,tmp_path/'mismatched')
    assert not (tmp_path/'mismatched').exists()

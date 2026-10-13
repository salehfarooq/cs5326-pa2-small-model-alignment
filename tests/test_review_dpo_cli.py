"""Reject malformed diagnostics before allocation; terminal positions stay valid."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import train_qwen as cli
from src.training import default_config


def inputs(tmp_path,monkeypatch,updates='0,50,100,200',terminal=200):
    repo=Path(__file__).resolve().parents[1]
    data={'train':[{'prompt':'Unrelated prepared question'}]*15000,'validation':[{'prompt':'Other prepared question'}]}
    config=default_config('dpo')
    decode=tmp_path/'decoding.json'
    decode.write_text(json.dumps({'do_sample':False,'max_new_tokens':256,'repetition_penalty':1.05,'no_repeat_ngram_size':0}))
    exclusion=[];banks=[]
    for name in cli.EXCLUSION_BANK_ROWS:
        path=tmp_path/name
        row={'instruction':'Independent final question'}
        path.write_text(json.dumps(row)+'\n' if path.suffix=='.jsonl' else json.dumps([row]))
        exclusion.append(path);banks.append({'name':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    monkeypatch.setattr(cli,'EXCLUSION_BANK_ROWS',{name:1 for name in cli.EXCLUSION_BANK_ROWS})
    monkeypatch.setattr(cli,'load_prepared',lambda path,track:(data,{}))
    for track in ('hh','sft','uf'):
        folder=tmp_path/track;folder.mkdir();(folder/'manifest.json').write_text('{}')
    args=SimpleNamespace(track='hh',data_dir=tmp_path/'hh',development_prompts=repo/'data/development/dpo_quality_40.json',generation_updates=updates,decoding_config=decode,exclusion_bank=exclusion,max_updates=terminal)
    identity={'config':{'exclusions':{'banks':banks}}}
    return args,data,identity,config


@pytest.mark.parametrize('change',[{'repetition_penalty':True},{'no_repeat_ngram_size':-1},{'max_new_tokens':257},{'do_sample':True},{'repetition_penalty':float('nan')}])
def test_invalid_decoder_fails_before_prepared_sibling_reads(tmp_path,monkeypatch,change):
    args,data,identity,config=inputs(tmp_path,monkeypatch)
    decode=json.loads(args.decoding_config.read_text());decode.update(change);args.decoding_config.write_text(json.dumps(decode))
    monkeypatch.setattr(cli,'load_prepared',lambda *a:pytest.fail('invalid decoder must fail first'))
    with pytest.raises(ValueError):cli.validate_development_inputs(args,data,identity,config)


def test_nonmultiple_terminal_and_keyword_are_validation_positions(tmp_path,monkeypatch):
    args,data,identity,config=inputs(tmp_path,monkeypatch,'0,50,197,terminal',197)
    result=cli.validate_development_inputs(args,data,identity,config)
    assert result['updates']=={0,50,197} and result['terminal']==197


def test_missing_exclusions_and_nonvalidation_generation_fail(tmp_path,monkeypatch):
    args,data,identity,config=inputs(tmp_path,monkeypatch)
    args.exclusion_bank=args.exclusion_bank[:1]
    with pytest.raises(ValueError,match='missing exclusion banks'):cli.validate_development_inputs(args,data,identity,config)
    args.exclusion_bank=list(tmp_path.glob('*.jsonl'))+list(tmp_path.glob('hh_rlhf_eval_300_seed42.json'))
    args.generation_updates='49'
    with pytest.raises(ValueError,match='validation position'):cli.validate_development_inputs(args,data,identity,config)

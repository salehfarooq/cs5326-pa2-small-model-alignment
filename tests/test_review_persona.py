import importlib.util
import json
from pathlib import Path
import pytest
from src.persona_data import prepare_data

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('persona_train',ROOT/'scripts/train_persona_lora.py')
training=importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def test_complete_supplied_split_and_roles_preserve_original_text():
    rows=json.loads((ROOT/'data/persona_lora/ramsay_style_preferences.json').read_text())
    train,validation,ids=prepare_data(rows)
    assert (len(train),len(validation))==(198,22)
    assert set(ids['train_ids']).isdisjoint(ids['validation_ids'])
    assert sorted(ids['train_ids']+ids['validation_ids'])==list(range(220))
    for records, indices in [(train,ids['train_ids']),(validation,ids['validation_ids'])]:
        for record,idx in zip(records,indices):
            assert set(record)=={'prompt','chosen','rejected'}
            assert record['prompt']==[{'role':'user','content':rows[idx]['prompt']}]
            assert record['chosen']==[{'role':'assistant','content':rows[idx]['chosen']}]
            assert record['rejected']==[{'role':'assistant','content':rows[idx]['rejected']}]
    assert prepare_data(rows)==prepare_data(rows)


def test_canonical_duplicate_must_stop_not_filter():
    rows=[{'prompt':' What IS this? ','chosen':'a','rejected':'b'}, {'prompt':'what is   this?','chosen':'c','rejected':'d'}, {'prompt':'other','chosen':'a','rejected':'b'}]
    with pytest.raises(ValueError):prepare_data(rows)


@pytest.mark.parametrize('bad',[[],[None,None],[{'prompt':'x','chosen':'','rejected':'b'}]*2,[{'prompt':'x','chosen':'a','rejected':'a'}]*2])
def test_invalid_preferences_stop(bad):
    with pytest.raises(ValueError):prepare_data(bad)


def test_minimal_split_keeps_text_and_global_rng_unchanged():
    import random
    rows=[{'prompt':'  A\n','chosen':' X ','rejected':'y'}, {'prompt':'b','chosen':'c','rejected':'d'}]
    before=random.getstate()
    train,evaluation,ids=prepare_data(rows)
    assert len(train)==len(evaluation)==1
    assert random.getstate()==before
    with pytest.raises(ValueError):prepare_data(rows,True)


def test_atomic_progress_failure_keeps_previous_file(tmp_path,monkeypatch):
    path=tmp_path/'run.json'
    training.write_json(path,{'status':'running'})
    def fail(*args):raise OSError('disk failure')
    monkeypatch.setattr(training.os,'replace',fail)
    with pytest.raises(OSError):training.write_json(path,{'status':'completed'})
    assert json.loads(path.read_text())=={'status':'running'}
    assert list(tmp_path.iterdir())==[path]


def test_nonfinite_progress_and_incomplete_adapter_rejected(tmp_path):
    with pytest.raises(ValueError):training.write_json(tmp_path/'run.json',{'loss':float('nan')})
    with pytest.raises(ValueError):training.checkpoint_manifest(tmp_path)


def test_reference_hash_excludes_adapter_but_detects_base_change():
    import torch
    model=torch.nn.Module()
    model.register_parameter('base',torch.nn.Parameter(torch.tensor([1.,2.]),requires_grad=False))
    model.register_parameter('lora_A',torch.nn.Parameter(torch.tensor([0.])))
    before=training.frozen_hash(model)
    with torch.no_grad():model.lora_A.add_(1)
    assert training.frozen_hash(model)==before
    with torch.no_grad():model.base.add_(1)
    assert training.frozen_hash(model)!=before
    model.base.requires_grad_(True)
    with pytest.raises(ValueError):training.frozen_hash(model)

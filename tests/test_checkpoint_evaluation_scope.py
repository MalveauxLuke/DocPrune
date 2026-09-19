import pytest
from experiments.training_pilot.evaluate_checkpoints import evaluation_rows_for_contract


def test_evaluation_uses_trained_questions_and_all_dev_and_records_empty_exclusions():
    train=dict(qid='train',split='train',pairs=2)
    empty=dict(qid='empty',split='train',pairs=0)
    dev=dict(qid='dev',split='dev',pairs=0)
    audit={'rows':[train,empty,dev]}
    contract={'train':['train'],'dev':['dev']}
    rows,excluded=evaluation_rows_for_contract(audit,contract)
    assert rows==[train,dev]
    assert [r['qid'] for r in excluded]==['empty']
    empty['pairs']=1
    with pytest.raises(ValueError,match='without preference pairs'):
        evaluation_rows_for_contract(audit,contract)

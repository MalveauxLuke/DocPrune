import importlib.util,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/m3doc600'))
spec=importlib.util.spec_from_file_location('baseline471',ROOT/'experiments/m3doc600/baseline.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_keeps_supplement_and_frozen_order():
 q={'pages':[{'page_id':str(i)} for i in range(5)],'presentation_order':[4,2,0,3,1]}
 assert [p['page_id'] for p in m.admitted(q)]==['4','2','0','3','1']
 q['presentation_order']=[0,1,2,3,3]
 with pytest.raises(AssertionError):m.admitted(q)

def test_smoke_strata_no_answers():
 qs=[];pool={}
 for kind in ['TextQ','TableQ','ImageQ']:
  for covered in [False,True]:
   for n in [4,5]:
    key=f'{kind}-{covered}-{n}';qs.append({'question_key':key,'question':'q','pages':[{}]*n})
    pool[key]={'metadata':{'type':kind},'operational_coverage':{'within_original_top4':covered}}
 selected=m.smoke_questions(qs,pool)
 assert len(selected)==6 and all(len(q['pages'])==5 for q in selected)

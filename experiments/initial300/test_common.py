import tempfile,unittest
from pathlib import Path
from common import publish,read,pages_for,query_for,fingerprint
class Contracts(unittest.TestCase):
 def test_shards_partition_once_and_smoke_is_reusable_subset(self):
  c={'pages':[{'key':str(i),'width':i+1,'height':3} for i in range(41)],'smoke_keys':['0','7','29'],'questions':[{'question_key':str(i)} for i in range(19)],'smoke_questions':['3']}
  for count in (1,4,8):
   ids=[p['key'] for n in range(count) for p in pages_for(c,n,count)]
   self.assertEqual(sorted(ids),sorted(p['key'] for p in c['pages']));self.assertEqual(len(set(ids)),41)
   qs=[q['question_key'] for n in range(count) for q in query_for(c,n,count)]
   self.assertEqual(len(set(qs)),19);self.assertEqual(len(qs),19)
  self.assertEqual({p['key'] for p in pages_for(c,0,1,True)},set(c['smoke_keys']))
 def test_publish_is_immutable_and_idempotent(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'record.json';publish(p,{'a':1});publish(p,{'a':1})
   with self.assertRaises(ValueError):publish(p,{'a':2})
   self.assertEqual(read(p),{'a':1})
 def test_identity_order_invariance(self):
  self.assertEqual(fingerprint({'a':1,'b':2}),fingerprint({'b':2,'a':1}))
if __name__=='__main__':unittest.main()

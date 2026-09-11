"""CPU-only tests for portable correction inputs; no model or retrieval imports."""
import json
import tempfile
import unittest
from pathlib import Path
from docprune.correction_corpus import complete_gold, resolve_asset, sha256, check_assets
from docprune.correction_scoring import score_answer


class CorrectionCorpusTests(unittest.TestCase):
    def test_recipe_inventory_distinguishes_available_and_blocked_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'page.pdf'
            source.write_bytes(b'existing page')
            asset = {'kind':'pdf','doc_id':'a','original_path':'/source/page.pdf',
                     'path':'assets/a.pdf','sha256':sha256(source)}
            missing = dict(asset, doc_id='b', sha256=None)
            corpus = {'assets':[asset, missing], 'cases':[
                {'case_id':'ready','pages':[{'doc_id':'a'}]},
                {'case_id':'blocked','pages':[{'doc_id':'b'}]}]}
            (root/'corpus.json').write_text(json.dumps(corpus))
            (root/'MANIFEST.sha256').write_text(sha256(root/'corpus.json')+'  corpus.json\n')
            result = check_assets(root, [root])
            self.assertEqual(result['available_case_ids'], ['ready'])
            self.assertEqual(result['blocked_cases'][0]['case_id'], 'blocked')
            self.assertIsNone(result['missing_assets'][0]['sha256'])

    def test_complete_set_is_one_target(self):
        contract = {'kind':'set','items':[{'canonical':'1998','aliases':[]},{'canonical':'1999','aliases':[]}]}
        self.assertEqual(complete_gold(contract), ['1998, 1999'])
        self.assertNotEqual(score_answer('1998', contract)['status'], 'correct')
        contract['accepted_complete'] = complete_gold(contract)
        self.assertEqual(score_answer('1998, 1999', contract)['status'], 'correct')

    def test_reuse_requires_matching_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root/'package'
            reuse = root/'existing600'
            package.mkdir()
            reused = reuse/'raw/scratch/source/page.pdf'
            reused.parent.mkdir(parents=True)
            reused.write_bytes(b'authenticated source')
            asset = {'kind':'pdf','doc_id':'doc','original_path':'/scratch/source/page.pdf','path':'assets/pdf/doc.pdf','sha256':sha256(reused)}
            self.assertEqual(resolve_asset(asset, package, [reuse]), reused)
            reused.write_bytes(b'changed')
            with self.assertRaises(FileNotFoundError):
                resolve_asset(asset, package, [reuse])



if __name__ == '__main__':
    unittest.main()

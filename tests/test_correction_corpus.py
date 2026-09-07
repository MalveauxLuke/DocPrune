"""CPU-only tests for portable correction inputs; no model or retrieval imports."""
import json
import tempfile
import unittest
from pathlib import Path
from docprune.correction_corpus import complete_gold, resolve_asset, sha256
from docprune.correction_scoring import score_answer


class CorrectionCorpusTests(unittest.TestCase):
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

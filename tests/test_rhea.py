"""Regressions for the Rhea library included in the repository."""
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import restore

DATA = Path(os.environ.get('BIORXNRESTORE_DATA_DIR', ROOT / 'data'))
REACTION = 'CC(=O)OCCC(C)C>>CC(=O)[O-]'
REFERENCE = 'CC(=O)OCCC(C)C.O>>CC(=O)[O-].CC(C)CCO.[H+]'


class RheaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.restorer = restore.Restorer(DATA)
        cls.addClassCleanup(cls.restorer.close)

    def assert_frozen(self, result, expected_digest, supports):
        self.assertEqual(result['candidate_count'], 3)
        self.assertEqual(sum(len(c['supports']) for c in result['candidates']), supports)
        core = {key: result[key] for key in ['candidates', 'sources', 'selected', 'status']}
        digest = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(digest, expected_digest, 'Candidate order, scores, support records or selection differs from the audited Rhea example.')

    def test_default(self):
        self.assert_frozen(self.restorer.complete(REACTION),
            '75e21de331b416a37dc2b48e4e8b1237e89895cba6b19a810e23a00be7d98b48', 213)

    def test_strict_holdout(self):
        result = self.restorer.complete(REACTION, strict_holdout=True,
            references=[REFERENCE], source_masters=[('rhea', '60436'), ('kegg', 'R12516')])
        self.assert_frozen(result, 'b3d6222253052eec7a868b9bd640acd1acd33058be21169183e985451ab2e89c', 211)
        self.assertEqual(result['mode'], 'strict_holdout')
        self.assertTrue(result['excluded_groups'])

if __name__ == '__main__':
    unittest.main()

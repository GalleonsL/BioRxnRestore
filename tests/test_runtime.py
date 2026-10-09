"""Code-only regression checks using synthetic reactions; no database or research package."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import restore as runner
from biorxnrestore import restoration as e
import numpy as np

class RestorationTests(unittest.TestCase):

    def fake_restorer(self, fulls=(), new=(), positive=0, error=None):
        obj = runner.Restorer.__new__(runner.Restorer)
        obj.manifest = {'method': 'test'}
        obj.lookup = e.FullSourceLookup([dict(standardized_reaction_smiles=f, source_database='rhea', source_master_id=str(i), source_reaction_id=str(i), direction='LR') for i, f in enumerate(fulls)])
        obj.forms = SimpleNamespace(prepare=lambda q: q)
        obj.encoder = e.RDKitDiffEncoder()
        obj.scorer = SimpleNamespace(batches=lambda *a: iter([(0, np.array([[1.0]]))]))

        def predict(*args):
            if error:
                raise error
            return dict(new=new, positive_groups=positive, excluded={})
        obj.engine = SimpleNamespace(predict=predict, source_catalog={})
        return obj

    def test_ambiguity_retains_all_direct_matches_without_selection(self):
        result = self.fake_restorer(['C>>C', 'C.O>>C.O']).complete('C>>C')
        self.assertEqual(result['status'], 'ambiguous_match')
        self.assertIsNone(result['selected'])
        self.assertEqual(result['candidate_count'], 2)

    def test_unique_direct_match_and_lookup_only_similarity_null(self):
        result = self.fake_restorer(['C>>C']).complete('C>>C')
        self.assertEqual(result['selected'], 'C>>C')
        self.assertIsNone(result['candidates'][0]['similarity'])

    def test_empty_status_distinguishes_no_support_from_failed_conservation(self):
        self.assertEqual(self.fake_restorer().complete('C>>C')['status'], 'no_positive_similarity_support')
        self.assertEqual(self.fake_restorer(positive=1).complete('C>>C')['status'], 'no_conservation_candidate')

    def test_template_failure_is_not_hidden_by_direct_match(self):
        with self.assertRaisesRegex(RuntimeError, 'timeout'):
            self.fake_restorer(['C>>C'], error=RuntimeError('timeout')).complete('C>>C')

    def test_invalid_inputs(self):
        for value in ['invalid', 'C>>C.O', '*C>>C']:
            with self.assertRaises(ValueError):
                runner.validate_input(value)

    def test_msp_empty_sides_do_not_inherit_reaction_validation(self):
        self.assertEqual(e.MissingSpeciesPattern().key, 'L:>>R:')
        with self.assertRaises(e.ReactionValidationError):
            e.Reaction((), ('C',))

    def test_missing_or_corrupted_asset_rejected_before_engine_load(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                runner.Restorer(d)
            p = Path(d) / runner.read_manifest()['files'][0]['file']
            p.write_text('corrupt')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                runner.Restorer(d)

    def test_mode_flags_dispatch_and_metadata_validation(self):
        import contextlib, io
        mock = MagicMock()
        mock.__enter__.return_value.complete.return_value = {'status': 'completed', 'mode': 'default', 'candidates': []}
        base = ['restore.py', '--data-dir', 'unused', '--reaction', 'C>>C']
        for flags, strict in [([], False), (['--strict-holdout'], True)]:
            with patch.object(sys, 'argv', base + flags), patch.object(runner, 'Restorer', return_value=mock), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 0)
            mock.__enter__.return_value.complete.assert_called_with('C>>C', strict_holdout=strict, references=[], source_masters=[])
        flags = ['--strict-holdout', '--reference', 'C>>C', '--reference', 'C.O>>C.O', '--exclude-source', 'rhea:123', '--exclude-source', 'kegg:R456']
        with patch.object(sys, 'argv', base + flags), patch.object(runner, 'Restorer', return_value=mock), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 0)
        mock.__enter__.return_value.complete.assert_called_with('C>>C', strict_holdout=True, references=['C>>C', 'C.O>>C.O'], source_masters=[('rhea', '123'), ('kegg', 'R456')])
        for flags in [['--reference', 'C>>C'], ['--exclude-source', 'rhea:123'], ['--strict-holdout', '--exclude-source', 'bad']]:
            with patch.object(sys, 'argv', base + flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                runner.main()
            self.assertEqual(error.exception.code, 2)

    def test_strict_mode_excludes_all_supplied_labels_and_skips_lookup(self):
        obj = self.fake_restorer(['C>>C'])
        obj.engine.index = SimpleNamespace(groups_by_source_master={('rhea', '123'): np.array([0])}, group_by_full_reaction={'C>>C': 1, 'C.O>>C.O': 2})
        obj.lookup.match = MagicMock(side_effect=AssertionError('Strict mode must not perform lookup'))
        result = obj.complete('C>>C', strict_holdout=True, references=['C>>C', 'C.O>>C.O'], source_masters=[('rhea', '123')])
        self.assertEqual(result['mode'], 'strict_holdout')
        self.assertEqual(result['candidate_count'], 0)
        np.testing.assert_array_equal(obj.engine.index.groups_by_source_master['strict_holdout', 'query'], [0, 1, 2])
        obj.lookup.match.assert_not_called()
        with self.assertRaises(ValueError):
            obj.complete('C>>C', references=['C>>C'])

    def test_terminal_top_three_and_complete_json(self):
        import contextlib, io, tempfile, copy
        output = dict(status='completed', mode='default', candidate_count=3, candidates=[dict(full_reaction=f'{m}>>{m}', similarity=1.0, origins=['exact_lookup']) for m in ['C', 'CC', 'CCC']])
        output['candidates'].append(dict(output['candidates'][0], full_reaction='fourth-candidate'))
        output['candidate_count'] = 4
        before = copy.deepcopy(output)
        mock = MagicMock()
        mock.__enter__.return_value.complete.return_value = output
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'result.json'
            with patch.object(sys, 'argv', ['restore.py', '--data-dir', 'unused', '--reaction', 'C>>C', '--output', str(target)]), patch.object(runner, 'Restorer', return_value=mock), contextlib.redirect_stdout(io.StringIO()) as console:
                self.assertEqual(runner.main(), 0)
            text = console.getvalue()
            self.assertIn('Top 1', text)
            self.assertIn('Top 3', text)
            self.assertNotIn('fourth-candidate', text)
            self.assertLess(text.index('Saved:'), text.index('Rank'))
            for i, candidate in enumerate(output['candidates'][:3], 1):
                line = next((line for line in text.splitlines() if line.startswith(f'Top {i}')))
                self.assertIn(candidate['full_reaction'].replace('>>', ' -> '), line)
            self.assertEqual(json.loads(target.read_text()), before)
        self.assertEqual(output, before)
        for status, candidates in [('no_conservation_candidate', []), ('ambiguous_match', output['candidates'][:2])]:
            with contextlib.redirect_stdout(io.StringIO()) as console:
                runner.print_summary(dict(mode='default', status=status, candidates=candidates))
            self.assertNotIn('Top 3', console.getvalue())
            self.assertIn('No completion candidates' if not candidates else 'no automatic selection', console.getvalue())

    def test_warning_filter_keeps_other_warnings_and_errors(self):
        import logging
        filt = runner._HydrogenWarningFilter()

        def record(level, message):
            return logging.LogRecord('rdkit', level, '', 0, message, (), None)
        self.assertFalse(filt.filter(record(logging.WARNING, 'WARNING: not removing hydrogen atom without neighbors')))
        self.assertFalse(filt.filter(record(logging.WARNING, 'WARNING: not removing hydrogen atom with dummy atom neighbors')))
        self.assertTrue(filt.filter(record(logging.WARNING, 'another warning')))
        self.assertTrue(filt.filter(record(logging.ERROR, 'not removing hydrogen atom without neighbors')))

    def test_cosine_and_mirror_numerics(self):
        matrix = np.array([[1, 2, 0], [-1, -2, 0], [0, 0, 0]], dtype=np.int64)
        scorer = e.StableCosine(matrix, integer=True)
        values = next(scorer.batches(matrix[:1], 1, 'numpy'))[1][0]
        np.testing.assert_allclose(values, [1, -1, 0], atol=0, rtol=1e-15)
        with self.assertRaises(ValueError):
            list(scorer.batches(matrix[:1], 1, 'torch_cuda'))
if __name__ == '__main__':
    unittest.main(verbosity=2)

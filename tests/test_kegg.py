"""Two-column KEGG imports validate chemistry and retain user-input provenance."""
import csv
import tempfile
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from biorxnrestore import preprocessing as p
import build_database as build

FULL = 'CC(=O)OCC.O>>CC(=O)O.CCO'


class KeggTests(unittest.TestCase):
    def load_rows(self, rows, fields=('reaction_id', 'reaction_smiles')):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.tsv'
            with path.open('w') as handle:
                writer = csv.writer(handle, delimiter='\t')
                writer.writerow(fields)
                writer.writerows(rows)
            return p.build_kegg_reactions(path, expected_sha256=p.sha256_file(path),
                source_release='user-supplied', direction_policy='both')

    def test_example_and_reverse(self):
        path = ROOT / 'examples/kegg_reactions.tsv'
        records, stats = p.build_kegg_reactions(path, expected_sha256=p.sha256_file(path),
            source_release='custom-date', direction_policy='both')
        self.assertEqual(len(records), 2)
        self.assertEqual(stats['input_rows'], 1)
        self.assertEqual(records[0].source_release, 'custom-date')
        self.assertEqual(records[0].equation, '')
        self.assertEqual(records[0].reaction_smiles, FULL)
        self.assertEqual(records[1].reaction_smiles, 'CC(=O)O.CCO>>CC(=O)OCC.O')

    def test_reject_invalid_chemistry_and_schema(self):
        cases = [('', FULL), ('X', ''), ('X', 'bad'), ('X', '*C>>*C'),
                 ('X', 'CCO>>CC=O'), ('X', '[Na+]>>[Na]')]
        for row in cases:
            with self.subTest(row=row), self.assertRaises(p.TableSchemaError):
                self.load_rows([row])
        with self.assertRaises(p.TableSchemaError):
            self.load_rows([('X', FULL), ('X', FULL)])
        with self.assertRaises(p.TableSchemaError):
            self.load_rows([('X', FULL)], fields=('reaction_id', 'wrong_column'))

    def test_optional_equation_and_legacy_flags_do_not_replace_audit(self):
        fields = ('reaction_id', 'reaction_smiles', 'equation', 'element_balanced',
                  'charge_balanced', 'balanced', 'status')
        records, _ = self.load_rows([('X', FULL, 'source equation', 'true', 'true', 'true', 'complete_balanced')], fields)
        self.assertEqual(records[0].equation, 'source equation')
        with self.assertRaises(p.TableSchemaError):
            self.load_rows([('X', 'CCO>>CC=O', '', 'true', 'true', 'true', 'complete_balanced')], fields)

    def test_custom_input_hash_and_rhea_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            rhea = raw / 'rhea.tsv'; rhea.write_text('frozen')
            kegg = raw / 'custom.tsv'; kegg.write_text('reaction_id\treaction_smiles\nX\t' + FULL + '\n')
            settings = {'inputs': [dict(role='rhea_smiles', file=rhea.name, sha256=p.sha256_file(rhea)),
                                  dict(role='kegg', file='historical.tsv', sha256='0'*64)]}
            paths, inputs = build.source_inputs(raw, settings, 'rhea-kegg', kegg)
            self.assertEqual(paths['kegg'], kegg)
            self.assertEqual(inputs[1]['sha256'], p.sha256_file(kegg))
            self.assertEqual(inputs[1]['file'], 'custom.tsv')
            self.assertEqual(settings['inputs'][1]['sha256'], '0'*64)
            kegg.rename(raw/'kegg_reactions.tsv')
            self.assertEqual(build.source_inputs(raw, settings, 'rhea-kegg')[0]['kegg'], raw/'kegg_reactions.tsv')
            self.assertNotIn('kegg', build.source_inputs(raw, settings)[0])
            rhea.write_text('changed')
            with self.assertRaises(ValueError):
                build.source_inputs(raw, settings, 'rhea-kegg')


if __name__ == '__main__':
    unittest.main()

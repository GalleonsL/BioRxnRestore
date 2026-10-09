import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import restore

class ManifestTests(unittest.TestCase):
    def test_custom_hashes_and_pinned_fallback(self):
        original=restore.read_manifest()
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(restore.read_manifest(directory),original)
            custom=copy.deepcopy(original);custom['files'][0]['sha256']='a'*64
            path=Path(directory)/'assets.json';path.write_text(json.dumps(custom))
            self.assertEqual(restore.read_manifest(directory),custom)
            for mutate in [lambda x:x.update(method='another'),lambda x:x['files'][0].update(file='../escape'),lambda x:x['files'].pop(0),lambda x:x['files'][0].update(sha256='invalid'),lambda x:x['query_forms'].update(policy='another'),lambda x:x['files'].append(x['files'][0]),lambda x:x.update(databases=['rhea','kegg'])]:
                broken=copy.deepcopy(custom);mutate(broken);path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):restore.read_manifest(directory)

    def test_rhea_and_local_pooled_layouts(self):
        manifest=restore.read_manifest()
        self.assertEqual(restore.manifest_databases(manifest),['rhea'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'assets.json'
            manifest['files']=[a for a in manifest['files'] if a['role']!='data_license']
            path.write_text(json.dumps(manifest))
            self.assertEqual(restore.read_manifest(directory),manifest)
            manifest['files'].append(dict(role='lookup_kegg',file='kegg_reactions.tsv',sha256='a'*64))
            manifest['databases']=['rhea','kegg']
            path.write_text(json.dumps(manifest))
            self.assertEqual(restore.manifest_databases(restore.read_manifest(directory)),['rhea','kegg'])

if __name__=='__main__':unittest.main(verbosity=2)

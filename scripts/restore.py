"""Single-query CPU entrypoint for the fixed, article-adopted method."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biorxnrestore import restoration as e

class _HydrogenWarningFilter(logging.Filter):
    def filter(self, record):
        known = ('not removing hydrogen atom without neighbors',
                 'not removing hydrogen atom with dummy atom neighbors')
        return record.levelno >= logging.ERROR or not any(text in record.getMessage() for text in known)


# Install in the CLI process and spawned workers; retain all other diagnostics.
from rdkit import rdBase
rdBase.LogToPythonLogger()
logging.getLogger('rdkit').addFilter(_HydrogenWarningFilter())


DATA = Path(__file__).resolve().parents[1] / 'data'


def read_manifest(data_dir=None):
    pinned = json.loads((DATA / 'manifest.json').read_text())
    local = Path(data_dir) / 'assets.json' if data_dir is not None else None
    if local is None or not local.exists():
        return pinned
    manifest = json.loads(local.read_text())
    required = {a['role']: a['file'] for a in pinned['files'] if a['role'] != 'data_license'}
    allowed = dict(required, lookup_kegg='kegg_reactions.tsv', data_license='LICENSE_DATA.txt')
    layout = {a['role']: a['file'] for a in manifest['files']}
    if (manifest['method'] != pinned['method'] or len(layout) != len(manifest['files'])
            or not required.keys() <= layout.keys()
            or any(allowed.get(role) != name for role, name in layout.items())
            or manifest['query_forms'] != pinned['query_forms']):
        raise ValueError('Data manifest changes the fixed method, file layout or chemical forms.')
    if manifest.get('databases', manifest_databases(manifest)) != manifest_databases(manifest):
        raise ValueError('Declared databases do not match the source tables.')
    for asset in manifest['files']:
        digest = asset['sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid data checksum in manifest.')
    return manifest


def manifest_databases(manifest):
    roles = {a['role'] for a in manifest['files']}
    return ['rhea', 'kegg'] if 'lookup_kegg' in roles else ['rhea']


def validate_input(reaction):
    query = e.normalize_reaction(reaction)
    if any(a.GetAtomicNum() == 0 for s in query.reactants + query.products
           for a in e.Chem.MolFromSmiles(s).GetAtoms()):
        raise ValueError('Use concrete molecular structures; unresolved wildcards are unsupported.')
    if len(query.products) != 1:
        raise ValueError('This entrypoint requires one observed product instance.')
    return query


class Restorer:
    """Load the pinned library once; close the template worker after use."""

    def __init__(self, data_dir):
        self.engine = None
        data_dir = Path(data_dir).expanduser().resolve()
        self.manifest = read_manifest(data_dir)
        paths = {}
        for asset in self.manifest['files']:
            path = data_dir / asset['file']
            if not path.is_file():
                raise FileNotFoundError(f'Missing data file: {path}')
            if e.sha256_file(path) != asset['sha256']:
                raise ValueError(f'Data hash mismatch: {path.name}')
            paths[asset['role']] = path
        native = e.read_clean_evidence(paths['evidence'])
        databases = manifest_databases(self.manifest)
        if {r.source_database for r in native} != set(databases):
            raise ValueError('Evidence databases do not match the source tables.')
        fingerprints = e.read_fingerprint_index(paths['fingerprints'],
            [r.evidence_id for r in native], 2048, 'RDKitDiff')
        matrix = fingerprints.fingerprints.matrix.copy()
        matrix[~fingerprints.fingerprints.valid] = 0
        self.scorer = e.StableCosine(matrix, integer=True)
        self.encoder = e.RDKitDiffEncoder()
        self.lookup = e.FullSourceLookup.from_config({
            'evidence_contract': {'sources': databases},
            'full_reaction_lookup': {'method': 'side_count_containment_v1', 'sources': [
                {'database': db, 'path': str(paths['lookup_' + db]),
                 'sha256': next(a['sha256'] for a in self.manifest['files'] if a['role'] == 'lookup_' + db)}
                for db in databases]}})
        forms = dict(self.manifest['query_forms'],
            mapping_path=str(paths['chemical_forms_mapping']),
            structures_path=str(paths['chemical_forms_structures']))
        self.forms = e.configured_query_forms({'query_chemical_forms': forms})
        extended = native + [e.mirror_evidence(r, len(native) + i) for i, r in enumerate(native)]
        self.engine = e.FixedPartitionEngine(data_dir, extended)

    def close(self):
        if self.engine is not None:
            self.engine.worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def complete(self, reaction, *, strict_holdout=False, references=(), source_masters=()):
        if (references or source_masters) and not strict_holdout:
            raise ValueError('Reference/source exclusions require --strict-holdout.')
        references = [e.normalize_reaction(value).key for value in references]
        if any(len(pair) != 2 or pair[0] not in {'rhea', 'kegg'} or not pair[1] for pair in source_masters):
            raise ValueError('Source exclusions require a rhea or kegg database and nonempty master ID.')
        skeleton = validate_input(reaction)
        query = e.QueryRecord(0, 'query', '', '', '', '', 'LR', skeleton, skeleton.key)
        query = self.forms.prepare(query)
        strict = strict_holdout
        if strict:
            # Benchmark-only labels alter exclusions, never candidate scoring/selection.
            groups = set()
            index = self.engine.index
            for database, master in source_masters:
                groups.update(index.groups_by_source_master.get((database, master), []))
            for value in references:
                full = e.normalize_reaction(value)
                for key in (full.key, e.Reaction(full.products, full.reactants).key):
                    if key in index.group_by_full_reaction:
                        groups.add(index.group_by_full_reaction[key])
            index.groups_by_source_master['strict_holdout', 'query'] = np.array(sorted(groups), dtype=np.int64)
            query = e.replace(query, label_database='strict_holdout', label_master_id='query')
        encoded = self.encoder.encode([query.skeleton.key])
        if not encoded.valid[0]:
            raise ValueError('RDKitDiff could not encode the query.')
        scores = next(self.scorer.batches(encoded.matrix, 1, 'numpy'))[1][0]
        scores = np.concatenate((scores, -scores))
        # Similarity always runs, even when direct matches exist. A timeout fails
        # the query rather than silently returning a partial candidate set.
        prediction = self.engine.predict(query, scores, {'query_template_timeout_seconds': 10}, strict)
        matches = [] if strict else self.lookup.match(query.skeleton)
        candidates, sources = self._merge(query, matches, prediction)
        if len(matches) > 1:
            status, selected = 'ambiguous_match', None
        elif candidates:
            status, selected = 'completed', candidates[0]['full_reaction']
        else:
            status = 'no_conservation_candidate' if prediction['positive_groups'] else 'no_positive_similarity_support'
            selected = None
        result = dict(input=reaction, prepared_input=query.skeleton.key,
            mode='strict_holdout' if strict else 'default',
            method=self.manifest['method'], library=self.manifest.get('library', 'custom'),
            status=status, selected=selected,
            candidate_count=len(candidates), candidates=candidates, sources=sources,
            query_form_changes=query.query_form_audit)
        if strict:
            result['excluded_groups'] = prediction['excluded']
            result['holdout'] = dict(full_references=references, source_masters=list(source_masters),
                policy='same skeleton and bidirectional full-source containment; supplied references and masters also excluded')
        return result

    def _merge(self, query, matches, prediction):
        candidates = {}
        sources = {}
        for i in matches:
            full = self.lookup.reactions[i]
            gid = 'group_' + e.sha256(('full-reaction-group:v2:' + full.key).encode()).hexdigest()[:20]
            supports = []
            for source in self.lookup.sources[i]:
                eid = 'LOOKUP:' + source['source_database'] + ':' + source['source_reaction_id']
                sources[eid] = dict(evidence_id=eid, source_group_id=gid,
                    source_full_reaction=full.key,
                    direction=self.lookup.source_directions[source['source_database'], source['source_reaction_id']], **source)
                supports.append(dict(evidence_id=eid, source_group_id=gid, similarity=None,
                    path='exact_lookup', restored_parent=None))
            candidates[full.key] = dict(full_reaction=full.key,
                msp=e.extract_msp(full, query.skeleton).key, similarity=None,
                similarity_rank=None, origins=['exact_lookup'], supports=supports)
        for rank, row in enumerate(prediction['new'], 1):
            full = e.normalize_reaction(row['completed'])
            if not e.reaction_delta(full).is_zero or e.extract_msp(full, query.skeleton).key != row['msp']:
                raise ValueError('Generated candidate failed conservation or query reconstruction.')
            current = candidates.setdefault(full.key, dict(full_reaction=full.key,
                msp=row['msp'], origins=[], supports=[]))
            current.update(similarity=row['similarity'], similarity_rank=rank)
            current['origins'].append('similarity_transfer')
            current['supports'].extend(row['branch_supports'])
            for support in row['branch_supports']:
                eid = support['evidence_id']
                sources[eid] = self.engine.source_catalog[eid]
                if eid.startswith('REV:'):
                    sources[eid[4:]] = self.engine.source_catalog[eid[4:]]
        return [dict(rank=i, **row) for i, row in enumerate(candidates.values(), 1)], dict(sorted(sources.items()))


def print_summary(result, output=None):
    mode = 'strict holdout' if result['mode'] == 'strict_holdout' else 'default'
    candidates = result['candidates']
    count = len(candidates)
    print(f"BioRxnRestore | {mode} | {count} candidate{'s' if count != 1 else ''}")
    if output is not None:
        print(f'Saved: {output}')
    if result['status'] == 'ambiguous_match':
        print('Ambiguous matches; no automatic selection.')
    if not candidates:
        print(f"No completion candidates ({result['status']}).")
        return
    rows = [('Rank', 'Method', 'Score', 'Completed reaction')]
    for position, candidate in enumerate(candidates[:3], 1):
        score = candidate['similarity']
        rows.append((f'Top {position}',
            'Match' if 'exact_lookup' in candidate['origins'] else 'Similarity',
            '—' if score is None else f'{score:.4f}',
            candidate['full_reaction'].replace('>>', ' -> ')))
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    print()
    for i, row in enumerate(rows):
        print(' | '.join(value.ljust(width) for value, width in zip(row, widths)).rstrip())
        if i == 0:
            print('-+-'.join('-' * width for width in widths))


def main():
    parser = argparse.ArgumentParser(description='Complete one biochemical reaction skeleton with the fixed paper method.')
    parser.add_argument('--data-dir', required=True, type=Path, help='Installed Rhea library or a locally built database')
    parser.add_argument('--reaction', required=True, help='Skeleton reaction SMILES')
    parser.add_argument('--strict-holdout', action='store_true',
        help='Exclude same-skeleton and containing sources in both directions; run similarity only')
    parser.add_argument('--reference', action='append', default=[], metavar='FULL_REACTION',
        help='Known full reference to exclude in both directions; repeat for all references (strict mode only)')
    parser.add_argument('--exclude-source', action='append', default=[], metavar='DATABASE:MASTER_ID',
        help='Source master to exclude, e.g. rhea:60436; repeat as needed (strict mode only)')
    parser.add_argument('--output', type=Path, help='Save all candidates and sources as JSON; the terminal still shows the top three')
    args = parser.parse_args()
    if (args.reference or args.exclude_source) and not args.strict_holdout:
        parser.error('--reference and --exclude-source require --strict-holdout')
    masters = []
    for value in args.exclude_source:
        database, separator, master = value.partition(':')
        if not separator or database not in {'rhea', 'kegg'} or not master:
            parser.error('--exclude-source must be rhea:MASTER_ID or kegg:MASTER_ID')
        masters.append((database, master))
    reaction = args.reaction
    try:
        validate_input(reaction)
        with Restorer(args.data_dir) as restorer:
            result = restorer.complete(reaction, strict_holdout=args.strict_holdout,
                references=args.reference, source_masters=masters)
        if args.output:
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print_summary(result, args.output)
    except Exception as exc:
        print(json.dumps(dict(status='error', error=str(exc)), ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

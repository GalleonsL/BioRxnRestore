"""Build the runtime library from pinned source snapshots; never edit research data."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import gzip
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biorxnrestore import restoration as e, preprocessing as p

DATA = Path(__file__).resolve().parents[1] / 'data'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def table(path):
    with Path(path).open() as f:
        return list(p.csv.DictReader(f, delimiter='\t'))


def write_table(path, rows):
    if not rows:
        raise ValueError(f'No rows to write: {path}')
    with Path(path).open('x', newline='') as f:
        writer = p.csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def audit(out, stage, rows):
    with gzip.open(out/'build_audit.jsonl.gz', 'at') as f:
        for row in rows:
            f.write(json.dumps(dict(stage=stage, record=row))+'\n')


def prepare_sources(raw, out, settings, database='rhea'):
    if out.exists():
        raise FileExistsError(f'Use a new output directory: {out}')
    paths = {}
    for item in settings['inputs']:
        if item['role'] == 'kegg' and database == 'rhea':
            continue
        path = raw/item['file']
        if not path.is_file() or e.sha256_file(path) != item['sha256']:
            raise ValueError(f'Missing or mismatched source snapshot: {path}')
        paths[item['role']] = path
    out.mkdir(parents=True)
    report = dict(status='preparing_sources', input_sha256={str(path):e.sha256_file(path) for path in paths.values()},
                  settings_sha256=e.sha256_file(DATA/'sources.json'),
                  databases=['rhea'] if database == 'rhea' else ['rhea', 'kegg'], stages={})
    write(out/'build_report.json', report)
    rhea, stats = p.build_rhea_reactions(p.RheaInputPaths(
        paths['rhea_smiles'],paths['rhea_directions'],paths['rhea_metadata'],paths['rhea_ec']),settings['rhea_release'])
    report['stages']['rhea_filter'] = dict(stats)
    auxiliary, stats = p.build_auxiliary_species(paths['cofactors'],rhea)
    report['stages']['auxiliary'] = dict(stats)
    p.write_auxiliary_species(out/'rhea_auxiliary.tsv',auxiliary)
    sources = [('rhea', rhea, 'net_reactions.tsv')]
    if database == 'rhea-kegg':
        forms = e.OfficialChemicalForms(paths['mapping'],paths['structures'],
            mapping_sha256=e.sha256_file(paths['mapping']),structures_sha256=e.sha256_file(paths['structures']))
        mapped_aux = []
        for row in auxiliary:
            form = forms.resolve(row.smiles)
            mapped_aux.append(replace(row,smiles=form.prepared_smiles,smiles_source=row.smiles_source+';'+form.status))
        p.write_auxiliary_species(out/'kegg_auxiliary.tsv',mapped_aux)
        kegg,stats = p.build_kegg_reactions(paths['kegg'],expected_sha256=e.sha256_file(paths['kegg']),
            source_release=settings['kegg_release'],direction_policy='recorded_only')
        report['stages']['kegg_snapshot'] = dict(stats)
        kegg,molecules,reactions,stats = p.prepare_kegg_chemical_forms(kegg,forms,direction_policy='both')
        audit(out,'kegg_molecular_forms',molecules)
        audit(out,'kegg_reference_forms',reactions)
        report['stages']['kegg_forms'] = dict(stats)
        sources.append(('kegg', kegg, 'kegg_reactions.tsv'))
    for name,records,filename in sources:
        rows=[{k:str(v) for k,v in r.to_tsv_row().items()} for r in records]
        prepared,records_audit,stats=p.prepare_net_source_rows(rows)
        write_table(out/filename,prepared)
        audit(out,name+'_net_reactions',records_audit)
        report['stages'][name+'_net_reactions']=dict(stats)
        print(f'{name.upper():<5} | {len(prepared):>6} directed net reactions',flush=True)
    for role in ['mapping','structures']:
        shutil.copyfile(paths[role],out/paths[role].name)
    names=['net_reactions.tsv','rhea_auxiliary.tsv',paths['mapping'].name,paths['structures'].name]
    if database == 'rhea-kegg':
        names.extend(['kegg_reactions.tsv', 'kegg_auxiliary.tsv'])
    report.update(status='sources_complete', source_outputs={n:e.sha256_file(out/n) for n in names})
    write(out/'build_report.json',report)
    return report


def take_slots(values, selected):
    remaining=Counter(selected);slots=[]
    for i,value in enumerate(values):
        if remaining[value]:
            slots.append(i);remaining[value]-=1
    if +remaining:
        raise ValueError('Selected participants are absent from the full source reaction')
    return slots


def build_parents(rows, decisions, evidence, offset=0):
    """Capture the complete parent before emitting its single-product children."""
    plans=defaultdict(list);by_source=defaultdict(list)
    for record in decisions:plans[record.source_reaction_id].append(record)
    for record in evidence:by_source[record.source_reaction_id].append(record)
    parents,children=[],[]
    for source in rows:
        sid=source['source_reaction_id']
        if sid not in plans:continue
        choices=sorted(plans[sid],key=lambda x:x.product_rank)
        if [r.product_rank for r in choices]!=list(range(1,choices[0].valid_product_count+1)):
            raise ValueError(f'Incomplete parent product partition: {sid}')
        full=e.normalize_reaction(source['standardized_reaction_smiles'])
        left=Counter();right=[]
        for choice in choices:
            branch=e.normalize_reaction(choice.skeleton_reaction_smiles)
            if any(n!=Counter(full.reactants)[v] for v,n in Counter(branch.reactants).items()):
                raise ValueError(f'Partial identical-participant selection: {sid}')
            left |= Counter(branch.reactants);right.extend(branch.products)
        ls=take_slots(full.reactants,left.elements());rs=take_slots(full.products,right)
        parent=e.Reaction(tuple(full.reactants[i] for i in ls),tuple(full.products[i] for i in rs))
        pid=offset+len(parents)
        parents.append(dict(parent_id=pid,source_reaction_id=sid,source_master_id=source['source_master_id'],
            full=full.key,parent=parent.key,auxiliary=e.extract_msp(full,parent).key,full_left_slots=ls,full_right_slots=rs))
        used=set();matched=set()
        for choice in choices:
            branch=e.normalize_reaction(choice.skeleton_reaction_smiles)
            lslots=take_slots(parent.reactants,branch.reactants)
            rslot=next(i for i,value in enumerate(parent.products) if value==branch.products[0] and i not in used)
            used.add(rslot)
            record=next(r for r in by_source[sid] if r.evidence_id not in matched and r.skeleton==branch and r.full_reaction==full)
            if record.msp!=e.extract_msp(full,branch):raise ValueError('Source MSP partition mismatch')
            matched.add(record.evidence_id)
            children.append(dict(evidence_id=record.evidence_id,parent_id=pid,query_id=choice.case_id,
                skeleton=branch.key,msp=record.msp.key,parent_left_slots=lslots,parent_product_slot=rslot))
        if len(matched)!=len(by_source[sid]):raise ValueError('Unmatched source branches')
    if len(children)!=len(evidence):raise ValueError('Not every evidence branch has a parent')
    return parents,children


def map_sources(parents, out, settings, cache=None):
    fulls=sorted({row['full'] for row in parents if len(e.normalize_reaction(row['parent']).products)>1})
    if cache is not None:
        saved=read(cache)
        identity=saved['mapper_identity'];rows=saved['mappings']
        if any(identity[k]!=settings['mapper_identity'][k] for k in ['version','max_tokens','model_files']):
            raise ValueError('Mapping cache has a different mapper identity')
        mapping={r['full']:r for r in rows}
        if len(mapping)!=len(rows) or not set(fulls)<=set(mapping):
            raise ValueError('Mapping cache must uniquely cover every required full reaction')
        selected=[mapping[f] for f in fulls]
        for row in selected:
            if row['status']=='mapped' and e.normalize_reaction(row['mapped_rxn']).key!=row['full']:
                raise ValueError('Cached atom mapping changes the full reaction')
        audit(out,'mapping_cache',[dict(path=str(cache),sha256=e.sha256_file(cache))])
    else:
        import torch
        from rxnmapper import RXNMapper
        if importlib.metadata.version('rxnmapper')!=settings['mapper_identity']['version']:
            raise ValueError('Install the pinned RXNMapper version')
        torch.set_num_threads(4)
        mapper=RXNMapper()
        identity=dict(version=importlib.metadata.version('rxnmapper'),device=str(mapper.device),
            max_tokens=mapper.model.config.max_position_embeddings,
            model_files={f.name:e.sha256_file(f) for f in sorted(Path(mapper.model_path).iterdir()) if f.is_file()})
        if any(identity[k]!=settings['mapper_identity'][k] for k in ['version','max_tokens','model_files']):
            raise ValueError('Installed RXNMapper assets differ from the pinned model')
        selected=[]
        for start in range(0,len(fulls),8):
            active=[];batch=[]
            for full in fulls[start:start+8]:
                tokens=len(mapper.tokenizer.encode(full))
                if tokens>identity['max_tokens']:batch.append(dict(full=full,status='model_length_limit',tokens=tokens))
                else:active.append(full)
            if active:
                try:
                    values=mapper.get_attention_guided_atom_maps(active)
                    if len(values)!=len(active):raise ValueError('Mapping result count mismatch')
                    batch.extend(dict(full=f,status='mapped',**v) for f,v in zip(active,values))
                except Exception as error:
                    batch.extend(dict(full=f,status='mapping_error',reason=f'{type(error).__name__}: {error}') for f in active)
            selected.extend(batch);audit(out,'mapping',batch)
            if start%64==0:print(f'Mapping | {min(start+8,len(fulls))}/{len(fulls)}',flush=True)
    write(out/'mapper_identity.json',identity)
    write(out/'mappings.json',dict(mapper_identity=identity,mappings=selected))
    return {r['full']:r for r in selected}


def template_worker(connection):
    e.RDLogger.DisableLog('rdApp.warning')
    while True:
        parent,mapped=connection.recv()
        try:result=p.make_template(parent,mapped)
        except Exception as error:result=dict(status='error',reason=f'{type(error).__name__}: {error}',candidates=[])
        connection.send(result)


class TemplateWorker:
    def __init__(self):self.process=None;self.connection=None
    def close(self):
        if self.process is not None:
            self.process.terminate();self.process.join();self.connection.close();self.process=None
    def make(self,parent,mapped):
        if self.process is None or not self.process.is_alive():
            self.close();ctx=mp.get_context('spawn');self.connection,child=ctx.Pipe()
            self.process=ctx.Process(target=template_worker,args=(child,),daemon=True);self.process.start();child.close()
        self.connection.send((parent,mapped))
        if not self.connection.poll(30):
            self.close();return dict(status='timeout',candidates=[])
        try:return self.connection.recv()
        except EOFError:
            self.close();return dict(status='worker_failed',candidates=[])


def build_library(out, settings, cache=None):
    report=read(out/'build_report.json')
    if report['status']!='sources_complete' or report['settings_sha256']!=e.sha256_file(DATA/'sources.json'):
        raise ValueError('Library building requires the unchanged completed sources stage')
    for name,h in report['source_outputs'].items():
        if e.sha256_file(out/name)!=h:raise ValueError(f'Prepared source changed: {name}')
    report['status']='building_library';write(out/'build_report.json',report)
    evidence=[];parents=[];children=[]
    config=p.ControlledBenchmarkConfig(**settings['skeleton'])
    databases=report.get('databases', ['rhea', 'kegg'])
    for database,filename in [('rhea','net_reactions.tsv'),('kegg','kegg_reactions.tsv')]:
        if database not in databases:
            continue
        rows=table(out/filename);cofactors,coa=p.read_auxiliary_species(out/(database+'_auxiliary.tsv'))
        decisions,stats=p.build_controlled_benchmark(rows,cofactors,coa,config)
        audit(out,database+'_associations',[r.to_tsv_row() for r in decisions])
        report['stages'][database+'_associations']=dict(stats)
        native=p.build_evidence_library(decisions)
        ps,cs=build_parents(rows,decisions,native,offset=len(parents))
        evidence_offset=len(evidence)
        evidence.extend(replace(r,index=evidence_offset+i) for i,r in enumerate(native))
        parents.extend(ps);children.extend(cs)
        print(f'{database.upper():<5} | {len(native):>6} skeleton branches',flush=True)
    write(out/'parents.json',parents);write(out/'children.json',children)
    p.write_clean_evidence(out/'evidence_library.tsv',evidence)
    encoded=e.RDKitDiffEncoder().encode([r.raw_retrieval_smiles for r in evidence])
    p.write_fingerprint_index(out/'fingerprints.npz',e.FingerprintIndex(tuple(r.evidence_id for r in evidence),encoded,'RDKitDiff'))
    mappings=map_sources(parents,out,settings,cache)
    counts=Counter();worker=TemplateWorker()
    try:
        with (out/'templates.jsonl').open('x') as f:
            for number,parent in enumerate(parents):
                if len(e.normalize_reaction(parent['parent']).products)==1:
                    result=dict(status='single_parent_direct_msp')
                else:
                    mapping=mappings[parent['full']]
                    result=worker.make(parent,mapping['mapped_rxn']) if mapping['status']=='mapped' else dict(status=mapping['status'])
                row=dict(parent_id=parent['parent_id'],**result);f.write(json.dumps(row)+'\n');f.flush()
                counts[result['status']]+=1
                if number%256==0:print(f'Templates | {number+1}/{len(parents)}',flush=True)
    finally:worker.close()
    # Only a fully finished build receives the manifest consumed by restore.py.
    manifest=read(DATA/'manifest.json')
    manifest['files']=[a for a in manifest['files'] if a['role']!='data_license']
    if 'kegg' in databases:
        manifest['files'].append(dict(role='lookup_kegg',file='kegg_reactions.tsv'))
    for item in manifest['files']:item['sha256']=e.sha256_file(out/item['file'])
    manifest['databases']=databases
    manifest['inputs']=[a for a in settings['inputs'] if a['role']!='kegg' or 'kegg' in databases]
    manifest['library']='-'.join(databases)+'-local'
    manifest.pop('license', None)
    manifest.pop('counts', None)
    manifest['build']=dict(rhea_release=settings['rhea_release'],
        source_input_sha256=report['input_sha256'],settings_sha256=report['settings_sha256'],
        mapper='live' if cache is None else 'saved_mapping_cache',paper_byte_identity=False)
    if 'kegg' in databases:
        manifest['build']['kegg_release']=settings['kegg_release']
    report.update(status='complete',evidence_rows=len(evidence),parents=len(parents),template_statuses=dict(counts),
        runtime_files={a['file']:a['sha256'] for a in manifest['files']},
        limitation='Fresh MCS timeouts/mapping outcomes can vary; a completed build is not proof of identity with the paper library.')
    write(out/'build_report.json',report)
    write(out/'assets.json',manifest)
    print(f'Complete | {len(evidence)} branches | {len(parents)} parents | {out}',flush=True)


def main():
    parser=argparse.ArgumentParser(description='Prepare the fixed BioRxnRestore source snapshots and runtime library.')
    parser.add_argument('--input-dir',type=Path,help='Folder containing the selected source files listed in data/sources.json')
    parser.add_argument('--database',choices=['rhea','rhea-kegg'],
        help='Default: rhea; rhea-kegg also requires the locally supplied KEGG snapshot')
    parser.add_argument('--output-dir',required=True,type=Path,help='New output directory; never overwrite a previous library')
    parser.add_argument('--stage',choices=['all','sources','library'],default='all',help='Default: run both source cleaning and library construction')
    parser.add_argument('--mapping-cache',type=Path,help='Optional exhaustive mappings.json from an audited prior build')
    args=parser.parse_args();settings=read(DATA/'sources.json');out=args.output_dir.resolve()
    if args.stage!='library' and args.input_dir is None:parser.error('--input-dir is required for sources/all')
    try:
        if args.stage=='library' and args.database is not None:
            expected=['rhea'] if args.database=='rhea' else ['rhea','kegg']
            if read(out/'build_report.json').get('databases',['rhea','kegg'])!=expected:
                raise ValueError('--database differs from the completed sources stage')
        if args.stage!='library':prepare_sources(args.input_dir.resolve(),out,settings,args.database or 'rhea')
        if args.stage!='sources':build_library(out,settings,args.mapping_cache)
        return 0
    except Exception as error:
        print(f'Error: {error}',file=sys.stderr)
        return 1


if __name__=='__main__':sys.exit(main())

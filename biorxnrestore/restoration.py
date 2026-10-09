"""Prepared-library CPU inference extracted from the adopted method.

Only runtime definitions are retained; no training, asset preparation or benchmarks.
"""
from __future__ import annotations

from collections import Counter
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
from itertools import permutations
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdChemReactions
from rdkit.Chem import rdqueries
from typing import ClassVar
from typing import Iterator
from typing import Literal
import csv
import hashlib
import json
import multiprocessing as mp
import numpy as np
import re


# biorxnrestore.models.reaction.ReactionValidationError
class ReactionValidationError(ValueError):
    """Raised when a normalized reaction violates its data contract."""


# biorxnrestore.models.reaction._validate_side
def _validate_side(name: str, values: tuple[str, ...]) -> None:
    if not values:
        raise ReactionValidationError(f'Reaction {name} side must contain at least one molecule')
    if any((not isinstance(value, str) or not value for value in values)):
        raise ReactionValidationError(f'Reaction {name} side contains an empty molecule')
    if values != tuple(sorted(values)):
        raise ReactionValidationError(f'Reaction {name} side must be canonically sorted')


# biorxnrestore.models.reaction.Reaction
@dataclass(frozen=True, slots=True)
class Reaction:
    """A directed reaction whose two sides are sorted canonical-SMILES multisets."""
    reactants: tuple[str, ...]
    products: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_side('reactant', self.reactants)
        _validate_side('product', self.products)

    def to_smiles(self) -> str:
        return f"{'.'.join(self.reactants)}>>{'.'.join(self.products)}"

    @property
    def key(self) -> str:
        return self.to_smiles()

    @property
    def stable_id(self) -> str:
        digest = sha256(f'reaction:v1:{self.key}'.encode('utf-8')).hexdigest()[:20]
        return f'rxn_{digest}'

    @property
    def reactant_counts(self) -> Counter[str]:
        return Counter(self.reactants)

    @property
    def product_counts(self) -> Counter[str]:
        return Counter(self.products)


# biorxnrestore.restoration.parent_engine.contains
def contains(query, full):
    return not Counter(query.reactants) - Counter(full.reactants) and (not Counter(query.products) - Counter(full.products))


# biorxnrestore.restoration.parent_engine.ExactContainmentIndex
class ExactContainmentIndex:
    """An exact, count-aware acceleration of the unchanged two-direction check."""

    def __init__(self, full_reactions: list[Reaction]):
        self.full_reactions = full_reactions
        self.postings = {side: defaultdict(set) for side in ('reactants', 'products')}
        for number, full in enumerate(full_reactions):
            for side in self.postings:
                for molecule in set(getattr(full, side)):
                    self.postings[side][molecule].add(number)
        self.cache = {}

    def find(self, query: Reaction) -> list[int]:
        if query.key in self.cache:
            return self.cache[query.key]
        found = set()
        for reverse in (False, True):
            postings = []
            for side, other in (('reactants', 'products'), ('products', 'reactants')):
                for molecule in set(getattr(query, side)):
                    postings.append(self.postings[other if reverse else side].get(molecule, set()))
            possible = set.intersection(*postings) if postings else set(range(len(self.full_reactions)))
            for number in possible:
                full = self.full_reactions[number]
                if contains(query, Reaction(full.products, full.reactants) if reverse else full):
                    found.add(number)
        result = sorted(found)
        self.cache[query.key] = result
        return result


# biorxnrestore.restoration.parent_templates.mol
@lru_cache(maxsize=16384)
def mol(smiles):
    return Chem.MolFromSmiles(smiles)


# biorxnrestore.restoration.parent_templates.canonical
def canonical(mol):
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


# biorxnrestore.restoration.parent_templates.reaction_template
@lru_cache(maxsize=8192)
def reaction_template(smarts):
    rxn = rdChemReactions.ReactionFromSmarts(smarts)
    if rxn is None:
        raise ValueError('invalid_template')
    rxn.GetSubstructParams().useChirality = True
    rxn.GetSubstructParams().maxMatches = 4294967295
    rxn.Initialize()
    return rxn


# biorxnrestore.restoration.parent_templates.apply_observed
@lru_cache(maxsize=512)
def apply_observed(smarts, inputs, product_slot, observed):
    rxn = reaction_template(smarts)
    observed_heavy_atoms = mol(observed).GetNumHeavyAtoms()
    generated, invalid = (set(), 0)
    for products in rxn.RunReactants(tuple((mol(s) for s in inputs)), maxProducts=0):
        retained = products[product_slot]
        if retained.GetNumHeavyAtoms() != observed_heavy_atoms:
            continue
        try:
            known_value = canonical(retained)
            if known_value != observed:
                continue
            values = tuple((known_value if i == product_slot else canonical(p) for i, p in enumerate(products)))
            if any(('.' in value for value in values)):
                invalid += 1
            else:
                generated.add(values)
        except (ValueError, RuntimeError):
            invalid += 1
    return (sorted(generated), invalid)


# biorxnrestore.models.msp.MSPValidationError
class MSPValidationError(ValueError):
    """Raised when an MSP violates its normalized data contract."""


# biorxnrestore.models.msp._validate_side
def _validate_msp_side(name: str, values: tuple[str, ...]) -> None:
    if any((not isinstance(value, str) or not value for value in values)):
        raise MSPValidationError(f'MSP {name} side contains an empty molecule')
    if values != tuple(sorted(values)):
        raise MSPValidationError(f'MSP {name} side must be canonically sorted')


# biorxnrestore.models.msp.MissingSpeciesPattern
@dataclass(frozen=True, slots=True)
class MissingSpeciesPattern:
    """Auxiliary species missing from each side of a skeleton reaction.

    Empty sides represent evidence that the skeleton is already complete.
    """
    reactants: tuple[str, ...] = ()
    products: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_msp_side('reactant', self.reactants)
        _validate_msp_side('product', self.products)

    @property
    def key(self) -> str:
        """Paper-parity key used by the frozen deterministic tie-break."""
        return f"L:{'|'.join(self.reactants)}>>R:{'|'.join(self.products)}"

    @property
    def structured_json(self) -> str:
        return json.dumps({'reactants': self.reactants, 'products': self.products}, ensure_ascii=False, separators=(',', ':'), sort_keys=True)

    @property
    def stable_id(self) -> str:
        digest = sha256(f'msp:v1:{self.structured_json}'.encode('utf-8')).hexdigest()[:20]
        return f'msp_{digest}'

    @property
    def species_count(self) -> int:
        return len(self.reactants) + len(self.products)


# biorxnrestore.representation.msp.MSPExtractionError
class MSPExtractionError(ValueError):
    """Raised when a skeleton is not a side-specific submultiset of a full reaction."""


# biorxnrestore.representation.msp._multiset_subtract
def _multiset_subtract(full: tuple[str, ...], retained: tuple[str, ...], side: str) -> tuple[str, ...]:
    remaining = Counter(full)
    for molecule in retained:
        if remaining[molecule] <= 0:
            raise MSPExtractionError(f'Skeleton {side} molecule is absent from the full reaction: {molecule}')
        remaining[molecule] -= 1
    values: list[str] = []
    for molecule, count in remaining.items():
        values.extend([molecule] * count)
    return tuple(sorted(values))


# biorxnrestore.representation.msp.extract_msp
def extract_msp(full_reaction: Reaction, skeleton: Reaction) -> MissingSpeciesPattern:
    return MissingSpeciesPattern(reactants=_multiset_subtract(full_reaction.reactants, skeleton.reactants, 'reactant'), products=_multiset_subtract(full_reaction.products, skeleton.products, 'product'))


# biorxnrestore.representation.reaction.MoleculeParseError
class MoleculeParseError(ValueError):
    """Raised when RDKit cannot parse a required molecule."""


# biorxnrestore.representation.reaction.canonicalize_molecule
@lru_cache(maxsize=None)
def canonicalize_molecule(smiles: str) -> str:
    if not smiles:
        raise MoleculeParseError('Molecule SMILES must not be empty')
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise MoleculeParseError(f'RDKit failed to parse molecule SMILES: {smiles}')
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


# biorxnrestore.representation.reaction.normalize_molecules
def normalize_molecules(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(sorted((canonicalize_molecule(value) for value in values)))


# biorxnrestore.representation.reaction.ReactionParseError
class ReactionParseError(ValueError):
    """Raised when a reaction does not satisfy the explicit input protocol."""


# biorxnrestore.representation.reaction.split_reaction_smiles
def split_reaction_smiles(reaction_smiles: str) -> tuple[str, str]:
    parts = reaction_smiles.split('>>')
    if len(parts) != 2 or '>' in parts[0] or '>' in parts[1]:
        raise ReactionParseError(f"Reaction SMILES must contain exactly one '>>' delimiter: {reaction_smiles}")
    if not parts[0] or not parts[1]:
        raise ReactionParseError(f'Reaction must contain molecules on both sides: {reaction_smiles}')
    return (parts[0], parts[1])


# biorxnrestore.representation.reaction.split_side_smiles
def split_side_smiles(side_smiles: str) -> tuple[str, ...]:
    return tuple((value for value in side_smiles.split('.') if value))


# biorxnrestore.representation.reaction.normalize_reaction
def normalize_reaction(reaction_smiles: str) -> Reaction:
    reactant_text, product_text = split_reaction_smiles(reaction_smiles)
    reactants = normalize_molecules(split_side_smiles(reactant_text))
    products = normalize_molecules(split_side_smiles(product_text))
    return Reaction(reactants=reactants, products=products)


# biorxnrestore.restoration.balance.CompositionDelta
@dataclass(frozen=True, slots=True)
class CompositionDelta:
    """Product-minus-reactant composition delta used in the manuscript."""
    elements: tuple[tuple[str, int], ...]
    charge: int

    @property
    def is_zero(self) -> bool:
        return not self.elements and self.charge == 0

    def __add__(self, other: 'CompositionDelta') -> 'CompositionDelta':
        values: defaultdict[str, int] = defaultdict(int)
        for element, count in self.elements + other.elements:
            values[element] += count
        return CompositionDelta(elements=tuple(sorted(((element, count) for element, count in values.items() if count))), charge=self.charge + other.charge)


# biorxnrestore.restoration.balance.CompositionError
class CompositionError(ValueError):
    """Raised when molecular composition cannot be computed."""


# biorxnrestore.restoration.balance.MoleculeComposition
@dataclass(frozen=True, slots=True)
class MoleculeComposition:
    elements: tuple[tuple[str, int], ...]
    charge: int


# biorxnrestore.restoration.balance.molecule_composition
@lru_cache(maxsize=None)
def molecule_composition(smiles: str) -> MoleculeComposition:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise CompositionError(f'RDKit failed to parse molecule for composition: {smiles}')
    molecule_with_hydrogen = Chem.AddHs(molecule)
    elements: defaultdict[str, int] = defaultdict(int)
    charge = 0
    for atom in molecule_with_hydrogen.GetAtoms():
        elements[atom.GetSymbol()] += 1
        charge += atom.GetFormalCharge()
    return MoleculeComposition(elements=tuple(sorted(elements.items())), charge=charge)


# biorxnrestore.restoration.balance.delta_for_sides
def delta_for_sides(reactants: tuple[str, ...], products: tuple[str, ...]) -> CompositionDelta:
    values: defaultdict[str, int] = defaultdict(int)
    charge = 0
    for sign, side in ((-1, reactants), (1, products)):
        for smiles in side:
            composition = molecule_composition(smiles)
            for element, count in composition.elements:
                values[element] += sign * count
            charge += sign * composition.charge
    return CompositionDelta(elements=tuple(sorted(((element, count) for element, count in values.items() if count))), charge=charge)


# biorxnrestore.restoration.balance.reaction_delta
def reaction_delta(reaction: Reaction) -> CompositionDelta:
    return delta_for_sides(reaction.reactants, reaction.products)


# biorxnrestore.representation.msp.reconstruct_reaction
def reconstruct_reaction(skeleton: Reaction, msp: MissingSpeciesPattern) -> Reaction:
    return Reaction(reactants=tuple(sorted(skeleton.reactants + msp.reactants)), products=tuple(sorted(skeleton.products + msp.products)))


# biorxnrestore.restoration.parent_templates.restore
def restore(query_key, parent, child, template):
    q, p = (normalize_reaction(query_key), normalize_reaction(parent['parent']))
    slots = child['parent_left_slots']
    if len(q.reactants) != len(slots) or len(q.products) != 1:
        return dict(status='role_count_mismatch', candidates=[])
    rxn = reaction_template(template['smarts'])
    if not mol(q.products[0]).HasSubstructMatch(rxn.GetProductTemplate(child['parent_product_slot']), useChirality=True):
        return dict(status='product_pattern_mismatch', candidates=[])
    candidates, assignments, invalid, outputs = ({}, 0, 0, 0)
    for values in sorted(set(permutations(q.reactants))):
        if not all((mol(value).HasSubstructMatch(rxn.GetReactantTemplate(slot), useChirality=True) for slot, value in zip(slots, values))):
            continue
        assignments += 1
        inputs = list(template['inputs'])
        for slot, value in zip(slots, values):
            inputs[slot] = value
        products, bad = apply_observed(template['smarts'], tuple(inputs), child['parent_product_slot'], q.products[0])
        invalid += bad
        outputs += len(products)
        for right in products:
            if right[child['parent_product_slot']] != q.products[0]:
                continue
            parent_left = list(p.reactants)
            for slot, value in zip(slots, values):
                parent_left[slot] = value
            restored = Reaction(tuple(sorted(parent_left)), tuple(sorted(right)))
            auxiliary = extract_msp(normalize_reaction(parent['full']), p)
            completed = reconstruct_reaction(restored, auxiliary)
            msp = extract_msp(completed, q)
            assert reconstruct_reaction(q, msp) == completed
            if reaction_delta(completed).is_zero:
                candidates[msp.key] = dict(msp=msp.key, completed=completed.key, restored_parent=restored.key)
    return dict(status='completed' if candidates else 'no_conserved_restoration', candidates=[candidates[k] for k in sorted(candidates)], assignments=assignments, generated_outcomes=outputs, invalid_outcomes=invalid)


# biorxnrestore.restoration.parent_templates._worker
def _worker(connection):
    RDLogger.DisableLog('rdApp.*')
    while True:
        name, args = connection.recv()
        try:
            if name != 'restore':
                raise ValueError('Only prepared-template restoration is supported')
            result = restore(*args)
        except Exception as e:
            result = dict(status='error', reason=f'{type(e).__name__}: {e}', candidates=[])
        connection.send(result)


# biorxnrestore.restoration.parent_templates.SupervisedWorker
class SupervisedWorker:
    """Persistent C++ worker, killed on timeout; incomplete products are never used."""

    def __init__(self):
        self.process = None
        self.connection = None

    def close(self):
        if self.process is not None:
            self.process.terminate()
            self.process.join()
            self.connection.close()
            self.process = None

    def call(self, name, args, seconds):
        if self.process is None or not self.process.is_alive():
            self.close()
            ctx = mp.get_context('spawn')
            self.connection, child = ctx.Pipe()
            self.process = ctx.Process(target=_worker, args=(child,), daemon=True)
            self.process.start()
            child.close()
        self.connection.send((name, args))
        if not self.connection.poll(seconds):
            self.close()
            return dict(status='timeout', candidates=[])
        try:
            return self.connection.recv()
        except EOFError:
            self.close()
            return dict(status='worker_failed', candidates=[])


# biorxnrestore.restoration.full_library.EvidenceGroupIndex
@dataclass(frozen=True, slots=True)
class EvidenceGroupIndex:
    """Stable complete-reaction groups and their MSP support relationships."""
    group_keys: tuple[str, ...]
    group_ids: tuple[str, ...]
    record_sort_order: np.ndarray
    group_starts: np.ndarray
    group_patterns: tuple[tuple[MissingSpeciesPattern, ...], ...]
    group_record_ids: tuple[tuple[str, ...], ...]
    group_source_databases: tuple[tuple[str, ...], ...]
    group_source_record_ids: tuple[tuple[str, ...], ...]
    pattern_support_groups: dict[str, np.ndarray]
    patterns_by_delta: dict[CompositionDelta, tuple[MissingSpeciesPattern, ...]]
    groups_by_source_master: dict[tuple[str, str], np.ndarray]
    groups_by_source_reaction: dict[tuple[str, str], np.ndarray]
    groups_by_skeleton: dict[str, np.ndarray]
    group_by_full_reaction: dict[str, int]

    @property
    def group_count(self) -> int:
        return len(self.group_keys)


# biorxnrestore.models.evidence.EvidenceRecord
@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    index: int
    evidence_id: str
    source_case_id: str
    source_database: str
    source_release: str
    source_master_id: str
    source_reaction_id: str
    direction: str
    skeleton: Reaction
    raw_retrieval_smiles: str
    full_reaction: Reaction
    msp: MissingSpeciesPattern
    ec_numbers: tuple[str, ...] = ()


# biorxnrestore.restoration.full_library._index_arrays
def _index_arrays(values: dict[object, set[int]]) -> dict[object, np.ndarray]:
    return {key: np.asarray(sorted(group_indexes), dtype=np.int64) for key, group_indexes in values.items()}


# biorxnrestore.restoration.balance.msp_delta
def msp_delta(msp: MissingSpeciesPattern) -> CompositionDelta:
    return delta_for_sides(msp.reactants, msp.products)


# biorxnrestore.restoration.full_library.build_evidence_group_index
def build_evidence_group_index(evidence: list[EvidenceRecord]) -> EvidenceGroupIndex:
    """Group evidence by normalized complete reaction using stable identifiers."""
    if not evidence:
        raise ValueError('BioRxnRestore requires a non-empty evidence library')
    if [record.index for record in evidence] != list(range(len(evidence))):
        raise ValueError('Evidence indices must be consecutive and match table order')
    records_by_group: defaultdict[str, list[EvidenceRecord]] = defaultdict(list)
    for record in evidence:
        records_by_group[record.full_reaction.key].append(record)
    group_keys = tuple(sorted(records_by_group))
    group_ids = tuple(('group_' + sha256(f'full-reaction-group:v2:{key}'.encode()).hexdigest()[:20] for key in group_keys))
    group_by_full = {key: index for index, key in enumerate(group_keys)}
    record_group_indexes = np.empty(len(evidence), dtype=np.int64)
    group_patterns: list[tuple[MissingSpeciesPattern, ...]] = []
    group_record_ids: list[tuple[str, ...]] = []
    group_source_databases: list[tuple[str, ...]] = []
    group_source_record_ids: list[tuple[str, ...]] = []
    pattern_groups: defaultdict[str, set[int]] = defaultdict(set)
    pattern_values: dict[str, MissingSpeciesPattern] = {}
    source_masters: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    source_reactions: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    skeletons: defaultdict[str, set[int]] = defaultdict(set)
    for group_index, group_key in enumerate(group_keys):
        records = records_by_group[group_key]
        patterns = tuple(sorted({record.msp for record in records}, key=lambda item: item.key))
        group_patterns.append(patterns)
        group_record_ids.append(tuple(sorted((record.evidence_id for record in records))))
        group_source_databases.append(tuple(sorted({record.source_database for record in records})))
        group_source_record_ids.append(tuple(sorted((f'{record.source_database}:{record.source_reaction_id}' for record in records))))
        for pattern in patterns:
            pattern_groups[pattern.key].add(group_index)
            pattern_values[pattern.key] = pattern
        for record in records:
            record_group_indexes[record.index] = group_index
            if record.source_master_id:
                source_masters[record.source_database, record.source_master_id].add(group_index)
            if record.source_reaction_id:
                source_reactions[record.source_database, record.source_reaction_id].add(group_index)
            skeletons[record.skeleton.key].add(group_index)
    patterns_by_delta_values: defaultdict[CompositionDelta, list[MissingSpeciesPattern]] = defaultdict(list)
    for pattern in pattern_values.values():
        patterns_by_delta_values[msp_delta(pattern)].append(pattern)
    patterns_by_delta = {delta: tuple(sorted(patterns, key=lambda item: item.key)) for delta, patterns in patterns_by_delta_values.items()}
    record_sort_order = np.argsort(record_group_indexes, kind='stable')
    sorted_groups = record_group_indexes[record_sort_order]
    group_starts = np.r_[0, np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1].astype(np.int64)
    if len(group_starts) != len(group_keys):
        raise AssertionError('Complete-reaction group index is internally inconsistent')
    return EvidenceGroupIndex(group_keys=group_keys, group_ids=group_ids, record_sort_order=record_sort_order, group_starts=group_starts, group_patterns=tuple(group_patterns), group_record_ids=tuple(group_record_ids), group_source_databases=tuple(group_source_databases), group_source_record_ids=tuple(group_source_record_ids), pattern_support_groups={key: np.asarray(sorted(groups), dtype=np.int64) for key, groups in pattern_groups.items()}, patterns_by_delta=patterns_by_delta, groups_by_source_master=_index_arrays(source_masters), groups_by_source_reaction=_index_arrays(source_reactions), groups_by_skeleton=_index_arrays(skeletons), group_by_full_reaction=group_by_full)


# biorxnrestore.restoration.full_library.deterministic_group_order
def deterministic_group_order(scores: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Order by score then the frozen canonical complete-reaction key order."""
    group_indexes = np.flatnonzero(eligible)
    return group_indexes[np.lexsort((group_indexes, -scores[group_indexes]))]


# biorxnrestore.models.evidence.QueryRecord
@dataclass(frozen=True, slots=True)
class QueryRecord:
    index: int
    query_id: str
    label_database: str
    label_release: str
    label_master_id: str
    label_reaction_id: str
    direction: str
    skeleton: Reaction
    raw_skeleton_smiles: str
    full_reaction: Reaction | None = None
    target_msp: MissingSpeciesPattern | None = None
    ec_numbers: tuple[str, ...] = ()
    source_skeleton_smiles: str | None = None
    coa_wildcard_expanded: bool = False
    form_original_skeleton: Reaction | None = None
    query_form_audit: dict | None = None


# biorxnrestore.restoration.full_library.eligible_group_mask
def eligible_group_mask(query: QueryRecord, index: EvidenceGroupIndex, strict_exclusions: bool, exclude_reverse_complete_reaction: bool=False) -> np.ndarray:
    """Apply exact-label exclusions at complete-reaction-group level."""
    eligible = np.ones(index.group_count, dtype=np.bool_)
    if not strict_exclusions:
        return eligible
    excluded: list[np.ndarray] = []
    master_key = (query.label_database, query.label_master_id)
    if query.label_master_id and master_key in index.groups_by_source_master:
        excluded.append(index.groups_by_source_master[master_key])
    reaction_key = (query.label_database, query.label_reaction_id)
    if query.label_reaction_id and reaction_key in index.groups_by_source_reaction:
        excluded.append(index.groups_by_source_reaction[reaction_key])
    if query.full_reaction is not None:
        group_index = index.group_by_full_reaction.get(query.full_reaction.key)
        if group_index is not None:
            excluded.append(np.asarray([group_index], dtype=np.int64))
        if exclude_reverse_complete_reaction:
            reverse = Reaction(query.full_reaction.products, query.full_reaction.reactants)
            reverse_index = index.group_by_full_reaction.get(reverse.key)
            if reverse_index is not None:
                excluded.append(np.asarray([reverse_index], dtype=np.int64))
    for skeleton in (query.skeleton, query.form_original_skeleton):
        if skeleton is not None:
            skeleton_groups = index.groups_by_skeleton.get(skeleton.key)
            if skeleton_groups is not None:
                excluded.append(skeleton_groups)
    for group_indexes in excluded:
        eligible[group_indexes] = False
    return eligible


# biorxnrestore.restoration.full_library.group_best_scores
def group_best_scores(record_scores: np.ndarray, index: EvidenceGroupIndex) -> np.ndarray:
    if record_scores.shape != (len(index.record_sort_order),):
        raise ValueError('Evidence score vector has the wrong length')
    return np.maximum.reduceat(record_scores[index.record_sort_order], index.group_starts)


# biorxnrestore.restoration.parent_engine.jread
def jread(path):
    return json.loads(path.read_text())


# biorxnrestore.restoration.parent_engine.Engine
class Engine:

    def __init__(self, out, es):
        self.parents = jread(out / 'parents.json')
        self.children = jread(out / 'children.json')
        self.templates = {r['parent_id']: r for r in (json.loads(l) for l in (out / 'templates.jsonl').read_text().splitlines())}
        assert len(self.templates) == len(self.parents)
        self.index = build_evidence_group_index(es)
        self.records = {r.evidence_id: r for r in es}
        self.record_group = {rid: i for i, ids in enumerate(self.index.group_record_ids) for rid in ids}
        self.containment = ExactContainmentIndex([normalize_reaction(k) for k in self.index.group_keys])
        self.single_support = defaultdict(set)
        self.catalog = []
        seen = set()
        for child in self.children:
            pid = child['parent_id']
            template = self.templates[pid]
            group = self.record_group[child['evidence_id']]
            if template['status'] == 'single_parent_direct_msp':
                self.single_support[child['msp']].add(group)
            elif template['status'] == 'ready':
                key = (group, pid, tuple(child['parent_left_slots']), child['parent_product_slot'])
                if key in seen:
                    continue
                seen.add(key)
                self.catalog.append(dict(catalog_id=len(self.catalog), group=group, **child))
        self.single_support = {k: np.array(sorted(v), dtype=np.int64) for k, v in self.single_support.items()}
        self.worker = SupervisedWorker()

    @staticmethod
    @lru_cache(maxsize=131072)
    def _matches(pattern, smiles):
        return Chem.MolFromSmiles(smiles).HasSubstructMatch(Chem.MolFromSmarts(pattern), useChirality=True)

    def masks(self, query, record_scores, strict_exclusions=True):
        scores = group_best_scores(record_scores, self.index)
        eligible = eligible_group_mask(query, self.index, strict_exclusions, exclude_reverse_complete_reaction=True)
        ordinary = np.flatnonzero(~eligible).tolist()
        exact = self.containment.find(query.skeleton) if strict_exclusions else []
        if strict_exclusions and query.form_original_skeleton is not None:
            exact = sorted(set(exact) | set(self.containment.find(query.form_original_skeleton)))
        eligible[exact] = False
        order = deterministic_group_order(scores, eligible)
        ranks = np.zeros(len(scores), dtype=np.int64)
        ranks[order] = np.arange(1, len(order) + 1)
        return (scores, eligible & (scores > 0), ranks, dict(ordinary=ordinary, exact=exact))


# biorxnrestore.restoration.full_library.required_msp_delta
def required_msp_delta(query: QueryRecord) -> CompositionDelta:
    delta = reaction_delta(query.skeleton)
    return CompositionDelta(elements=tuple(((element, -count) for element, count in delta.elements)), charge=-delta.charge)


# biorxnrestore.restoration.unified.BranchEngine
class BranchEngine(Engine):
    """Reuse chemical generation, but never borrow another branch's score."""

    def __init__(self, out, evidence):
        super().__init__(out, evidence)
        self.direct_branches = defaultdict(list)
        self.catalog = []
        self.source_catalog = {}
        for child in self.children:
            record = self.records[child['evidence_id']]
            parent = self.parents[child['parent_id']]
            template = self.templates[child['parent_id']]
            group = self.record_group[record.evidence_id]
            c = dict(child, group=group, evidence_index=record.index)
            if template['status'] == 'single_parent_direct_msp':
                self.direct_branches[child['msp']].append(c)
            elif template['status'] == 'ready':
                self.catalog.append(dict(c, catalog_id=len(self.catalog)))
            self.source_catalog[record.evidence_id] = dict(evidence_id=record.evidence_id, source_group_id=self.index.group_ids[group], source_database=record.source_database, source_release=record.source_release, source_reaction_id=record.source_reaction_id, source_master_id=record.source_master_id, direction=record.direction, source_full_reaction=record.full_reaction.key, retrieval_skeleton=record.skeleton.key, source_msp=record.msp.key, source_parent=parent['parent'], parent_auxiliary=parent['auxiliary'], parent_id=child['parent_id'], parent_left_slots=child['parent_left_slots'], parent_product_slot=child['parent_product_slot'], template_status=template['status'], template_smarts=template.get('smarts'))
        self.by_product_pattern = defaultdict(list)
        for child in self.catalog:
            pattern = self.templates[child['parent_id']]['product_patterns'][child['parent_product_slot']]
            self.by_product_pattern[len(child['parent_left_slots']), pattern].append(child)
        self._product_cache = {}

    def _product_compatible(self, product, reactant_count):
        key = (product, reactant_count)
        if key not in self._product_cache:
            if len(self._product_cache) >= 32768:
                self._product_cache.clear()
            self._product_cache[key] = tuple((c for (count, pattern), children in self.by_product_pattern.items() if count == reactant_count and self._matches(pattern, product) for c in children))
        return self._product_cache[key]

    def possible(self, q):
        if len(q.products) != 1:
            return []
        assignments = set(permutations(q.reactants))
        result = []
        for c in self._product_compatible(q.products[0], len(q.reactants)):
            t = self.templates[c['parent_id']]
            pats = [t['reactant_patterns'][slot] for slot in c['parent_left_slots']]
            if any((all((self._matches(pat, value) for pat, value in zip(pats, values))) for values in assignments)):
                result.append(c)
        return result

    def predict(self, query, record_scores, config, strict_exclusions=True):
        _, positive, _, excluded = self.masks(query, record_scores, strict_exclusions)
        candidates, supports = ({}, defaultdict(dict))

        def eligible(child):
            return positive[child['group']] and record_scores[child['evidence_index']] > 0

        def add(child, candidate, path):
            record_id = child['evidence_id']
            score = float(record_scores[child['evidence_index']])
            support = dict(evidence_id=record_id, source_group_id=self.index.group_ids[child['group']], similarity=score, path=path, restored_parent=candidate.get('restored_parent'))
            key = candidate['msp']
            supports[key][record_id] = support
            order = (-score, self.index.group_keys[child['group']], record_id)
            if key not in candidates or order < candidates[key]['_order']:
                candidates[key] = dict(candidate, similarity=score, group=child['group'], path=path, _order=order)
        for msp in self.index.patterns_by_delta.get(required_msp_delta(query), ()):
            for child in self.direct_branches.get(msp.key, ()):
                if eligible(child):
                    full = reconstruct_reaction(query.skeleton, msp)
                    add(child, dict(msp=msp.key, completed=full.key), child.get('path', 'direct_msp'))
        attempts = []
        possible = self.possible(query.skeleton)
        for child in possible:
            if not eligible(child):
                continue
            pid = child['parent_id']
            result = self.worker.call('restore', (query.skeleton.key, self.parents[pid], child, self.templates[pid]), config['query_template_timeout_seconds'])
            if result['status'] not in {'completed', 'no_conserved_restoration'}:
                raise RuntimeError(f"Incomplete template execution: {query.query_id}, {child['evidence_id']}: {result}")
            attempts.append(dict(evidence_id=child['evidence_id'], **{k: v for k, v in result.items() if k != 'candidates'}))
            for candidate in result['candidates']:
                add(child, candidate, 'parent_restore')
        ordered = sorted(candidates.values(), key=lambda c: (-c['similarity'], self.index.group_keys[c['group']], c['msp']))
        for rank, row in enumerate(ordered, 1):
            del row['_order']
            row['rank'] = rank
            row['branch_supports'] = [supports[row['msp']][k] for k in sorted(supports[row['msp']])]
        return dict(query_id=query.query_id, query=query.skeleton.key, excluded=excluded, positive_groups=int(positive.sum()), possible_templates=len(possible), attempts=attempts, new=ordered, supports={k: sorted({self.record_group[eid] for eid in v}) for k, v in supports.items()})


# biorxnrestore.restoration.fixed_reverse.REVERSE_PATH
REVERSE_PATH = 'fixed_partition_reverse_msp'


# biorxnrestore.restoration.fixed_reverse.REVERSE_PREFIX
REVERSE_PREFIX = 'REV:'


# biorxnrestore.restoration.fixed_reverse.reverse_pattern
def reverse_pattern(pattern):
    return MissingSpeciesPattern(pattern.products, pattern.reactants)


# biorxnrestore.restoration.fixed_reverse.reverse_reaction
def reverse_reaction(reaction):
    return Reaction(reaction.products, reaction.reactants)


# biorxnrestore.restoration.fixed_reverse.mirror_evidence
def mirror_evidence(record, index):
    """Swap sides without changing structures, multiplicities or membership."""
    if record.evidence_id.startswith(REVERSE_PREFIX):
        raise ValueError('Prepared evidence IDs must not use the reserved REV: prefix')
    skeleton = reverse_reaction(record.skeleton)
    full = reverse_reaction(record.full_reaction)
    msp = reverse_pattern(record.msp)
    if reconstruct_reaction(skeleton, msp) != full:
        raise ValueError(f'Invalid source skeleton/MSP partition: {record.evidence_id}')
    return replace(record, index=index, evidence_id=REVERSE_PREFIX + record.evidence_id, skeleton=skeleton, raw_retrieval_smiles=skeleton.key, full_reaction=full, msp=msp, direction={'LR': 'RL', 'RL': 'LR'}[record.direction])


# biorxnrestore.restoration.fixed_reverse.FixedPartitionEngine
class FixedPartitionEngine(BranchEngine):
    """Native parent restoration plus mirrored, branch-scored MSP transfer."""

    def __init__(self, out, evidence):
        super().__init__(out, evidence)
        native = {r.evidence_id: r for r in evidence if not r.evidence_id.startswith(REVERSE_PREFIX)}
        if len(evidence) != 2 * len(native):
            raise ValueError('Each native evidence branch requires exactly one mirror')
        if set(native) != {c['evidence_id'] for c in self.children}:
            raise ValueError('Native parent assets and mirrored evidence do not align')
        for child in self.children:
            original = native[child['evidence_id']]
            record = self.records[REVERSE_PREFIX + original.evidence_id]
            if record != mirror_evidence(original, record.index):
                raise ValueError(f'Mirrored evidence differs from its original: {record.evidence_id}')
            group = self.record_group[record.evidence_id]
            self.direct_branches[record.msp.key].append(dict(evidence_id=record.evidence_id, evidence_index=record.index, group=group, path=REVERSE_PATH))
            source = self.source_catalog[original.evidence_id]
            parent = normalize_reaction(source['source_parent'])
            auxiliary = extract_msp(original.full_reaction, parent)
            self.source_catalog[record.evidence_id] = dict(source, evidence_id=record.evidence_id, source_group_id=self.index.group_ids[group], direction=record.direction, source_full_reaction=record.full_reaction.key, retrieval_skeleton=record.skeleton.key, source_msp=record.msp.key, source_parent=reverse_reaction(parent).key, parent_auxiliary=reverse_pattern(auxiliary).key, parent_id=REVERSE_PREFIX + str(source['parent_id']), parent_left_slots=[source['parent_product_slot']], parent_product_slot=None, parent_product_slots=list(source['parent_left_slots']), template_status='not_applied_fixed_partition_reverse_msp', template_smarts=None, orientation_policy='fixed_partition_reverse', original_evidence_id=original.evidence_id, original_direction=original.direction, original_full_reaction=original.full_reaction.key, original_retrieval_skeleton=original.skeleton.key, original_msp=original.msp.key, original_parent_id=source['parent_id'], original_source_parent=source['source_parent'], original_parent_auxiliary=source['parent_auxiliary'], original_template_status=source['template_status'], original_template_smarts=source['template_smarts'])


# biorxnrestore.provenance.sha256_file
def sha256_file(path: str | Path, chunk_size: int=1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open('rb') as handle:
        while (chunk := handle.read(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()


# biorxnrestore.data.net_reactions.source_reaction_sides
def source_reaction_sides(value: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Parse prepared source SMILES, allowing RDKit dative bonds within molecules."""
    parts = value.split('>>')
    if len(parts) != 2 or not all(parts):
        raise ValueError('Source reaction requires two nonempty sides separated by >>')
    sides = tuple((tuple((canonicalize_molecule(s) for s in split_side_smiles(side))) for side in parts))
    if not all(sides):
        raise ValueError('Source reaction requires molecules on both sides')
    return sides


# biorxnrestore.restoration.lookup_first.FullSourceLookup
class FullSourceLookup:
    """Exact directed molecular multiset containment, independent of fingerprints."""

    def __init__(self, rows):
        grouped = defaultdict(list)
        self.source_directions = {}
        for row in rows:
            left, right = source_reaction_sides(row['standardized_reaction_smiles'])
            reaction = Reaction(tuple(sorted(left)), tuple(sorted(right)))
            if '*' in reaction.key or not reaction_delta(reaction).is_zero:
                raise ValueError('Lookup sources must be concrete balanced full reactions')
            grouped[reaction].append({k: row[k] for k in ('source_database', 'source_master_id', 'source_reaction_id')})
            self.source_directions[row['source_database'], row['source_reaction_id']] = row.get('direction', '')
        self.reactions = sorted(grouped, key=lambda r: r.key)
        self.sources = [sorted(grouped[r], key=lambda v: (v['source_database'], v['source_reaction_id'])) for r in self.reactions]
        self.postings = defaultdict(set)
        self.counts = []
        for i, r in enumerate(self.reactions):
            sides = (Counter(r.reactants), Counter(r.products))
            self.counts.append(sides)
            for side, counts in enumerate(sides):
                for molecule in counts:
                    self.postings[side, molecule].add(i)

    @classmethod
    def from_config(cls, values):
        settings = values['full_reaction_lookup']
        if settings.get('method') != 'side_count_containment_v1':
            raise ValueError('The fixed workflow requires side/count full-reaction lookup')
        if set(settings) != {'method', 'sources'}:
            raise ValueError('Unsupported lookup options; stage routing is fixed')
        configured = settings['sources']
        expected = values['evidence_contract']['sources']
        if [s['database'] for s in configured] != expected:
            raise ValueError('Lookup sources must match the configured similarity-library scope')
        rows = []
        for source in configured:
            path = Path(source['path'])
            if sha256_file(path) != source['sha256']:
                raise ValueError(f'Full-reaction lookup source hash mismatch: {path}')
            with path.open() as f:
                data = list(csv.DictReader(f, delimiter='\t'))
            if not data or any((r['source_database'] != source['database'] for r in data)):
                raise ValueError('Lookup source database identity mismatch')
            rows.extend(data)
        return cls(rows)

    def match(self, query):
        counts = (Counter(query.reactants), Counter(query.products))
        postings = [self.postings.get((side, m), set()) for side, cs in enumerate(counts) for m in cs]
        possible = set.intersection(*sorted(postings, key=len)) if postings else set()
        return [i for i in sorted(possible) if all((not cs - self.counts[i][side] for side, cs in enumerate(counts)))]


# biorxnrestore.data.chemical_forms.FORM_POLICY
FORM_POLICY = 'official_known_forms_joint_stereo_v2'


# biorxnrestore.data.chemical_forms.ChemicalForm
@dataclass(frozen=True)
class ChemicalForm:
    original_smiles: str
    prepared_smiles: str
    status: str
    official_references: tuple[tuple[str, str, str], ...] = ()

    @property
    def mapped(self) -> bool:
        return self.status in {'official_changed', 'official_unchanged', 'free_proton'}


# biorxnrestore.data.chemical_forms._molecule
@lru_cache(maxsize=None)
def _molecule(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or any((a.GetAtomicNum() == 0 for a in molecule.GetAtoms())):
        raise ValueError(f'Invalid concrete molecule: {smiles}')
    return Chem.RemoveHs(molecule)


# biorxnrestore.data.chemical_forms._attributes
def _attributes(atom: Chem.Atom) -> tuple[int, int, int, int]:
    return (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetNumRadicalElectrons(), atom.GetFormalCharge() - atom.GetTotalNumHs(includeNeighbors=True))


# biorxnrestore.data.chemical_forms.protonation_lookup_key
@lru_cache(maxsize=None)
def protonation_lookup_key(smiles: str) -> str:
    """Coarse labelled-graph index; never sufficient to authorize a conversion."""
    molecule = Chem.Mol(_molecule(smiles))
    if not any((a.GetAtomicNum() > 1 for a in molecule.GetAtoms())):
        return 'literal:' + canonicalize_molecule(smiles)
    Chem.RemoveStereochemistry(molecule)
    for atom in molecule.GetAtoms():
        _, _, radicals, charge_minus_h = _attributes(atom)
        if not -128 <= charge_minus_h < 128:
            raise ValueError('q-H exceeds the labelled-graph encoding range')
        atom.SetAtomMapNum(1024 + charge_minus_h + 256 * radicals)
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
    return 'graph:' + Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


# biorxnrestore.data.chemical_forms._query
@lru_cache(maxsize=None)
def _query(smiles: str) -> Chem.Mol:
    molecule = _molecule(smiles)
    query = Chem.RWMol(molecule)
    for atom in molecule.GetAtoms():
        replacement = rdqueries.AtomNumEqualsQueryAtom(atom.GetAtomicNum())
        replacement.SetChiralTag(atom.GetChiralTag())
        query.ReplaceAtom(atom.GetIdx(), replacement)
    return query


# biorxnrestore.data.chemical_forms._matches
def _matches(source: str, target: str, stereo: bool) -> bool:
    source_mol, target_mol = (_molecule(source), _molecule(target))
    expected = tuple((_attributes(a) for a in source_mol.GetAtoms()))
    parameters = Chem.SubstructMatchParameters()
    parameters.useChirality = stereo
    parameters.useEnhancedStereo = stereo
    parameters.specifiedStereoQueryMatchesUnspecified = False
    parameters.setExtraFinalCheck(lambda molecule, mapping: all((_attributes(molecule.GetAtomWithIdx(int(index))) == expected[i] for i, index in enumerate(mapping))))
    return target_mol.HasSubstructMatch(_query(source), parameters)


# biorxnrestore.data.chemical_forms.same_protonation_form
@lru_cache(maxsize=None)
def same_protonation_form(a: str, b: str, *, stereo: bool=True) -> bool:
    """Same chemistry under proton addition/removal, with strict stereo by default.

    Bidirectional matching retains specified versus unspecified stereo. Atom
    isotope/radical/q-H attributes are checked within each accepted stereo
    mapping, not with a second potentially incompatible graph isomorphism.
    """
    ma, mb = (_molecule(a), _molecule(b))
    if ma.GetNumAtoms() != mb.GetNumAtoms() or ma.GetNumBonds() != mb.GetNumBonds():
        return False
    if protonation_lookup_key(a) != protonation_lookup_key(b):
        return False
    if a == b:
        return True
    return _matches(a, b, stereo) and _matches(b, a, stereo)


# biorxnrestore.data.chemical_forms.OfficialChemicalForms
class OfficialChemicalForms:
    """A frozen compound dictionary independent of query labels and predictions."""

    def __init__(self, mapping_path: str | Path, structures_path: str | Path, *, mapping_sha256: str, structures_sha256: str) -> None:
        self.paths = (Path(mapping_path), Path(structures_path))
        for path, expected in zip(self.paths, (mapping_sha256, structures_sha256)):
            if sha256_file(path) != expected:
                raise ValueError(f'Chemical-form input SHA-256 mismatch: {path}')
        self.input_sha256 = {str(p): sha256_file(p) for p in self.paths}
        self.stats: Counter = Counter()
        structures: dict[str, str] = {}
        with self.paths[1].open() as handle:
            for row in csv.reader(handle, delimiter='\t'):
                if len(row) != 2:
                    raise ValueError('Official structures require identifier and SMILES')
                identifier = row[0].removeprefix('CHEBI:')
                try:
                    value = canonicalize_molecule(row[1])
                    _molecule(value)
                except ValueError:
                    self.stats['invalid_official_structure'] += 1
                    continue
                if identifier in structures and structures[identifier] != value:
                    raise ValueError(f'Conflicting official structures for ChEBI {identifier}')
                structures[identifier] = value
        index: dict[str, dict[str, set[tuple[str, str, str]]]] = defaultdict(lambda: defaultdict(set))
        with self.paths[0].open() as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            if not {'CHEBI', 'CHEBI_PH7_3', 'ORIGIN'} <= set(reader.fieldnames or []):
                raise ValueError('Missing official chemical-form mapping columns')
            for row in reader:
                source, target = (row['CHEBI'].removeprefix('CHEBI:'), row['CHEBI_PH7_3'].removeprefix('CHEBI:'))
                if target not in structures:
                    self.stats['target_structure_unavailable'] += 1
                    continue
                value = structures[target]
                if source in structures and (not same_protonation_form(structures[source], value)):
                    self.stats['source_target_correspondence_failed'] += 1
                    continue
                index[protonation_lookup_key(value)][value].add((source, target, row['ORIGIN']))
                self.stats['accepted_mapping_rows'] += 1
        self.index = {key: {s: tuple(sorted(refs)) for s, refs in values.items()} for key, values in index.items()}
        self._cache: dict[str, ChemicalForm] = {}

    def resolve(self, smiles: str) -> ChemicalForm:
        if smiles in self._cache:
            return self._cache[smiles]
        if smiles == '[H+]':
            result = ChemicalForm(smiles, smiles, 'free_proton')
        else:
            targets = {target: refs for target, refs in self.index.get(protonation_lookup_key(smiles), {}).items() if same_protonation_form(smiles, target)}
            if len(targets) == 1:
                target, refs = next(iter(targets.items()))
                result = ChemicalForm(smiles, target, 'official_unchanged' if target == smiles else 'official_changed', refs)
            else:
                result = ChemicalForm(smiles, smiles, 'unmapped' if not targets else 'conflicting_official_targets')
        self._cache[smiles] = result
        return result


# biorxnrestore.restoration.query_forms.QueryFormPreparer
class QueryFormPreparer:

    def __init__(self, settings):
        if settings.get('policy') != FORM_POLICY:
            raise ValueError('Unsupported query chemical-form policy')
        if settings.get('unresolved_policy') != 'retain_original':
            raise ValueError('Unresolved query forms must retain the original molecule')
        required = {'policy', 'unresolved_policy', 'mapping_path', 'structures_path', 'mapping_sha256', 'structures_sha256'}
        if set(settings) != required:
            raise ValueError('Incomplete or unsupported query chemical-form settings')
        self.settings = dict(settings)
        self.mapper = OfficialChemicalForms(**{k: v for k, v in settings.items() if k not in {'policy', 'unresolved_policy'}})

    def prepare(self, query):
        if query.query_form_audit is not None:
            if query.query_form_audit['settings'] != self.settings:
                raise ValueError('Query was already prepared with a different chemical-form policy')
            if query.query_form_audit['prepared_skeleton'] != query.skeleton.key:
                raise ValueError('Prepared query differs from its chemical-form audit')
            return query
        molecules = {}
        for smiles in sorted(set(query.skeleton.reactants + query.skeleton.products)):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f'Invalid query molecule: {smiles}')
            if any((a.GetAtomicNum() == 0 for a in mol.GetAtoms())):
                molecules[smiles] = dict(original_smiles=smiles, prepared_smiles=smiles, status='unresolved_wildcard', official_references=())
            else:
                form = self.mapper.resolve(smiles)
                if self.mapper.resolve(form.prepared_smiles).prepared_smiles != form.prepared_smiles:
                    raise ValueError('Official query chemical forms are not idempotent')
                molecules[smiles] = asdict(form)
        prepared = Reaction(*(tuple(sorted((molecules[s]['prepared_smiles'] for s in side))) for side in (query.skeleton.reactants, query.skeleton.products)))
        changed = prepared != query.skeleton
        target = query.target_msp
        if changed and query.full_reaction is not None:
            try:
                target = extract_msp(query.full_reaction, prepared)
            except MSPExtractionError as exc:
                raise ValueError(f'Prepared query is incompatible with its reference label: {query.query_id}; prepare reference forms independently, or use the historical configuration') from exc
        audit = dict(settings=self.settings, original_input=query.raw_skeleton_smiles, original_skeleton=query.skeleton.key, prepared_skeleton=prepared.key, changed=changed, molecules=list(molecules.values()), label_policy='reference_unchanged; residual_recomputed_if_input_changed')
        return replace(query, skeleton=prepared, raw_skeleton_smiles=prepared.key, target_msp=target, source_skeleton_smiles=query.source_skeleton_smiles or query.raw_skeleton_smiles, form_original_skeleton=query.skeleton, query_form_audit=audit)


# biorxnrestore.restoration.query_forms.configured_query_forms
def configured_query_forms(values):
    settings = values.get('query_chemical_forms')
    return None if settings is None else QueryFormPreparer(settings)


# biorxnrestore.io.datasets.TableSchemaError
class TableSchemaError(ValueError):
    """Raised when a table does not satisfy the requested explicit schema."""


# biorxnrestore.io.datasets._require_columns
def _require_columns(path: Path, fieldnames: list[str] | None, required: set[str]) -> None:
    missing = required - set(fieldnames or ())
    if missing:
        raise TableSchemaError(f'{path} is missing required columns: {sorted(missing)}')


# biorxnrestore.io.datasets._require_file
def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f'Required table does not exist: {path}')


# biorxnrestore.representation.msp.normalize_msp
def normalize_msp(reactants: tuple[str, ...] | list[str], products: tuple[str, ...] | list[str]) -> MissingSpeciesPattern:
    return MissingSpeciesPattern(reactants=normalize_molecules(tuple(reactants)), products=normalize_molecules(tuple(products)))


# biorxnrestore.representation.msp.parse_missing_species_column
def parse_missing_species_column(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    separator = '|' if '|' in value else '.'
    return tuple((item for item in value.split(separator) if item))


# biorxnrestore.representation.msp.msp_from_columns
def msp_from_columns(removed_reactants: str, removed_products: str) -> MissingSpeciesPattern:
    return normalize_msp(parse_missing_species_column(removed_reactants), parse_missing_species_column(removed_products))


# biorxnrestore.io.datasets.split_ec_numbers
def split_ec_numbers(value: str) -> tuple[str, ...]:
    return tuple(sorted({part for part in re.split('[|;,\\s]+', value) if part}))


# biorxnrestore.io.datasets.read_clean_evidence
def read_clean_evidence(path: str | Path) -> list[EvidenceRecord]:
    table_path = Path(path)
    _require_file(table_path)
    required = {'schema_version', 'evidence_index', 'evidence_id', 'source_case_id', 'source_database', 'source_release', 'source_master_id', 'source_reaction_id', 'direction', 'retrieval_reaction_smiles', 'full_reaction_smiles', 'removed_reactant_smiles', 'removed_product_smiles'}
    evidence: list[EvidenceRecord] = []
    with table_path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        _require_columns(table_path, reader.fieldnames, required)
        for expected_index, row in enumerate(reader):
            if row['schema_version'] != '2':
                raise TableSchemaError(f"Unsupported evidence schema in {table_path}: {row['schema_version']}")
            if int(row['evidence_index']) != expected_index:
                raise TableSchemaError(f"{table_path} evidence_index must be contiguous and file ordered; expected {expected_index}, got {row['evidence_index']}")
            raw_skeleton = row['retrieval_reaction_smiles']
            evidence.append(EvidenceRecord(index=expected_index, evidence_id=row['evidence_id'], source_case_id=row['source_case_id'], source_database=row['source_database'], source_release=row['source_release'], source_master_id=row['source_master_id'], source_reaction_id=row['source_reaction_id'], direction=row['direction'], skeleton=normalize_reaction(raw_skeleton), raw_retrieval_smiles=raw_skeleton, full_reaction=normalize_reaction(row['full_reaction_smiles']), msp=msp_from_columns(row['removed_reactant_smiles'], row['removed_product_smiles']), ec_numbers=split_ec_numbers(row.get('ec_numbers', ''))))
    return evidence


# biorxnrestore.retrieval.similarity.EncodedBatch
@dataclass(frozen=True, slots=True)
class EncodedBatch:
    matrix: np.ndarray
    valid: np.ndarray


# biorxnrestore.retrieval.index.FINGERPRINT_INDEX_SCHEMA_VERSION
FINGERPRINT_INDEX_SCHEMA_VERSION = 2


# biorxnrestore.retrieval.index.FingerprintIndex
@dataclass(frozen=True, slots=True)
class FingerprintIndex:
    record_ids: tuple[str, ...]
    fingerprints: EncodedBatch
    encoder: str = 'RDKitDiff'
    schema_version: int = FINGERPRINT_INDEX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.fingerprints.matrix.ndim != 2:
            raise ValueError('Fingerprint matrix must be two-dimensional')
        rows = self.fingerprints.matrix.shape[0]
        if len(self.record_ids) != rows or self.fingerprints.valid.shape != (rows,):
            raise ValueError('Fingerprint rows, validity flags, and record IDs must align')
        if len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError('Fingerprint record IDs must be unique')


# biorxnrestore.retrieval.index.read_fingerprint_index
def read_fingerprint_index(path: str | Path, expected_record_ids: list[str] | tuple[str, ...] | None=None, expected_dimensions: int | None=None, expected_encoder: str | None=None) -> FingerprintIndex:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f'Fingerprint index does not exist: {source}')
    with np.load(source, allow_pickle=False) as values:
        required = {'schema_version', 'encoder', 'record_ids', 'matrix', 'valid'}
        missing = required - set(values.files)
        if missing:
            raise ValueError(f'Fingerprint index is missing arrays: {sorted(missing)}')
        schema_version = int(values['schema_version'].item())
        if schema_version != FINGERPRINT_INDEX_SCHEMA_VERSION:
            raise ValueError(f'Unsupported fingerprint index schema: {schema_version}')
        matrix = np.asarray(values['matrix'])
        valid = np.asarray(values['valid'], dtype=np.bool_)
        record_ids = tuple((str(value) for value in values['record_ids'].tolist()))
        encoder = str(values['encoder'].item())
    if not encoder:
        raise ValueError('Fingerprint index has an empty encoder name')
    if expected_encoder is not None and encoder != expected_encoder:
        raise ValueError(f'Fingerprint encoder does not match: expected {expected_encoder}, observed {encoder}')
    if expected_dimensions is not None and matrix.shape[1] != expected_dimensions:
        raise ValueError(f'Fingerprint dimensions do not match: expected {expected_dimensions}, observed {matrix.shape[1]}')
    if expected_record_ids is not None and record_ids != tuple(expected_record_ids):
        raise ValueError('Fingerprint record IDs do not match the current input order')
    return FingerprintIndex(record_ids=record_ids, fingerprints=EncodedBatch(matrix=matrix, valid=valid), encoder=encoder, schema_version=schema_version)


# biorxnrestore.retrieval.numerics.EXACT_INTEGER_LIMIT
EXACT_INTEGER_LIMIT = 2 ** 53


# biorxnrestore.retrieval.numerics.NUMERICAL_POLICY
NUMERICAL_POLICY = 'exact_integer_dot_float64_cosine_v1'


# biorxnrestore.retrieval.numerics._prepare
def _prepare(matrix: np.ndarray, integer: bool) -> tuple[np.ndarray, np.ndarray, int]:
    values = np.asarray(matrix)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Cosine fingerprints must be finite two-dimensional arrays')
    if integer:
        if not np.equal(values, np.rint(values)).all():
            raise ValueError('AtomPair cosine requires unnormalized integer counts')
        maximum = max(abs(int(values.min(initial=0))), abs(int(values.max(initial=0))))
        norm_bound = values.shape[1] * maximum ** 2
        if norm_bound >= EXACT_INTEGER_LIMIT:
            raise ValueError('Integer norm bound exceeds the exact float64 accumulation range')
        counts = values.astype(np.int64, copy=False)
        squares = np.sum(counts * counts, axis=1, dtype=np.int64)
        norms = np.sqrt(squares.astype(np.float64))
    else:
        maximum = 0
        values = values.astype(np.float64, copy=False)
        norms = np.sqrt(np.sum(values * values, axis=1, dtype=np.float64))
    return (values.astype(np.float64, copy=False), norms, maximum)


# biorxnrestore.retrieval.numerics.StableCosine
class StableCosine:
    """Prepared evidence; raw integer dot sign precedes cosine normalization."""

    def __init__(self, evidence: np.ndarray, *, integer: bool):
        self.integer = integer
        self.evidence, self.evidence_norms, self.evidence_max = _prepare(evidence, integer)
        self.audit: dict[str, object] = {'policy': NUMERICAL_POLICY, 'integer_counts': integer, 'dot_accumulation': 'certified_exact_float64' if integer else 'fixed_row_float64', 'zero_norm': 'no_support', 'similarity_cutoff': 0, 'evidence_zero_vectors': int((self.evidence_norms == 0).sum())}

    def batches(self, queries: np.ndarray, batch_size: int, backend: str) -> Iterator[tuple[int, np.ndarray]]:
        values, query_norms, query_max = _prepare(queries, self.integer)
        if values.shape[1] != self.evidence.shape[1]:
            raise ValueError('Query and evidence fingerprint dimensions differ')
        if batch_size < 1:
            raise ValueError('Query batch size must be positive')
        if backend != 'numpy':
            raise ValueError(f'Unsupported cosine backend: {backend}')
        self.audit['query_zero_vectors'] = int((query_norms == 0).sum())
        if self.integer:
            bound = values.shape[1] * query_max * self.evidence_max
            if bound >= EXACT_INTEGER_LIMIT:
                raise ValueError('Integer dot bound exceeds the exact float64 accumulation range')
            self.audit.update(absolute_dot_bound=bound, query_square_norm_bound=values.shape[1] * query_max ** 2, evidence_square_norm_bound=values.shape[1] * self.evidence_max ** 2, exact_integer_limit=EXACT_INTEGER_LIMIT)
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size]
            if self.integer:
                dots = batch @ self.evidence.T
            else:
                dots = np.stack([self.evidence @ row for row in batch])
            denominator = query_norms[start:start + batch_size, None] * self.evidence_norms
            scores = np.divide(dots, denominator, out=np.zeros_like(dots), where=denominator > 0)
            yield (start, scores)


# biorxnrestore.retrieval.similarity.SimilarityMetric
SimilarityMetric = Literal['cosine', 'tanimoto']


# biorxnrestore.retrieval.similarity.RDKitDiffEncoder
@dataclass(frozen=True, slots=True)
class RDKitDiffEncoder:
    dimensions: int = 2048
    name: ClassVar[str] = 'RDKitDiff'
    metric: ClassVar[SimilarityMetric] = 'cosine'

    def encode(self, reaction_smiles: list[str]) -> EncodedBatch:
        rows: list[np.ndarray] = []
        valid: list[bool] = []
        for value in reaction_smiles:
            try:
                reaction = rdChemReactions.ReactionFromSmarts(value, useSmiles=True)
                if reaction is None:
                    raise ValueError('RDKit returned no reaction')
                fingerprint = rdChemReactions.CreateDifferenceFingerprintForReaction(reaction)
                row = np.zeros(self.dimensions, dtype=np.int64)
                for bit, count in fingerprint.GetNonzeroElements().items():
                    if 0 <= bit < self.dimensions:
                        row[int(bit)] = int(count)
                rows.append(row)
                valid.append(True)
            except Exception:
                rows.append(np.zeros(self.dimensions, dtype=np.int64))
                valid.append(False)
        matrix = np.vstack(rows) if rows else np.zeros((0, self.dimensions), dtype=np.int64)
        return EncodedBatch(matrix=matrix, valid=np.asarray(valid, dtype=np.bool_))

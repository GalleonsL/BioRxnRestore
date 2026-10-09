"""Source preparation primitives retained from the adopted scientific workflow."""
from __future__ import annotations

from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from collections import deque
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import replace
from .restoration import ChemicalForm
from .restoration import CompositionError
from .restoration import EvidenceRecord
from .restoration import FORM_POLICY
from .restoration import FingerprintIndex
from .restoration import OfficialChemicalForms
from .restoration import Reaction
from .restoration import TableSchemaError
from .restoration import _require_columns
from .restoration import _require_file
from .restoration import canonical
from .restoration import canonicalize_molecule
from .restoration import delta_for_sides
from .restoration import mol
from .restoration import molecule_composition
from .restoration import msp_from_columns
from .restoration import normalize_reaction
from .restoration import reaction_delta
from .restoration import reaction_template
from .restoration import same_protonation_form
from .restoration import sha256_file
from .restoration import source_reaction_sides
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdRascalMCES
from typing import Iterable
import csv
import hashlib
import json
import numpy as np
import os
import tempfile


# biorxnrestore.data.rhea.RheaInputPaths
@dataclass(frozen=True, slots=True)
class RheaInputPaths:
    reaction_smiles: Path
    directions: Path
    reactions: Path
    ec: Path

    def validate(self) -> None:
        for path in asdict(self).values():
            _require_file(Path(path))


# biorxnrestore.data.rhea.RheaReactionRecord
@dataclass(frozen=True, slots=True)
class RheaReactionRecord:
    schema_version: int
    source_database: str
    source_release: str
    source_master_id: str
    source_reaction_id: str
    direction: str
    ec_numbers: str
    reaction_smiles: str
    standardized_reaction_smiles: str
    reaction_key: str
    is_balanced: bool
    is_approved: bool

    def to_tsv_row(self) -> dict[str, object]:
        row = asdict(self)
        row['is_balanced'] = 'true' if self.is_balanced else 'false'
        row['is_approved'] = 'true' if self.is_approved else 'false'
        return row


# biorxnrestore.data.rhea._canonicalize_preserving_participant_order
def _canonicalize_preserving_participant_order(reaction_smiles: str) -> str:
    sides = reaction_smiles.split('>>')
    if len(sides) != 2 or not sides[0] or (not sides[1]):
        raise ValueError("reaction must contain exactly one non-empty '>>' delimiter")
    normalized_sides = []
    for side in sides:
        normalized_sides.append('.'.join((canonicalize_molecule(value) for value in side.split('.') if value)))
    return '>>'.join(normalized_sides)


# biorxnrestore.data.rhea._direction_map
def _direction_map(path: Path) -> dict[str, tuple[str, str]]:
    values: dict[str, tuple[str, str]] = {}
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        _require_columns(path, reader.fieldnames, {'RHEA_ID_MASTER', 'RHEA_ID_LR', 'RHEA_ID_RL'})
        for row in reader:
            for direction, column in (('LR', 'RHEA_ID_LR'), ('RL', 'RHEA_ID_RL')):
                rhea_id = row[column]
                if rhea_id:
                    values[rhea_id] = (row['RHEA_ID_MASTER'], direction)
    return values


# biorxnrestore.data.rhea._ec_numbers
def _ec_numbers(path: Path) -> dict[str, str]:
    values: dict[str, OrderedDict[str, None]] = defaultdict(OrderedDict)
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        _require_columns(path, reader.fieldnames, {'MASTER_ID', 'ID'})
        for row in reader:
            if row['MASTER_ID'] and row['ID']:
                values[row['MASTER_ID']][row['ID']] = None
    return {master: '|'.join(entries) for master, entries in values.items()}


# biorxnrestore.data.rhea._iter_reaction_smiles
def _iter_reaction_smiles(path: Path):
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle, delimiter='\t')
        for line_number, row in enumerate(reader, start=1):
            if not row or not any((value.strip() for value in row)):
                continue
            if len(row) != 2:
                raise TableSchemaError(f'{path}:{line_number} expected two tab-separated fields, got {len(row)}')
            yield (line_number, row[0], row[1])


# biorxnrestore.data.rhea._reaction_metadata
def _reaction_metadata(path: Path) -> dict[str, tuple[bool, bool, str]]:
    values: dict[str, tuple[bool, bool, str]] = {}
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        _require_columns(path, reader.fieldnames, {'reaction_id', 'status', 'is_transport'})
        for row in reader:
            status = row['status'].rsplit('/', 1)[-1]
            values[row['reaction_id']] = (status == 'Approved', row['is_transport'].strip().lower() == 'true', row.get('is_balanced', '').strip().lower())
    return values


# biorxnrestore.data.rhea.build_rhea_reactions
def build_rhea_reactions(paths: RheaInputPaths, source_release: int) -> tuple[list[RheaReactionRecord], Counter]:
    if source_release < 1:
        raise ValueError('Rhea release must be positive')
    paths.validate()
    directions = _direction_map(paths.directions)
    metadata = _reaction_metadata(paths.reactions)
    ecs = _ec_numbers(paths.ec)
    records: list[RheaReactionRecord] = []
    stats: Counter = Counter()
    for line_number, rhea_id, raw_reaction in _iter_reaction_smiles(paths.reaction_smiles):
        stats['input_rows'] += 1
        if rhea_id not in directions:
            raise TableSchemaError(f'{paths.reaction_smiles}:{line_number} Rhea ID {rhea_id} has no direction')
        master_id, direction = directions[rhea_id]
        if master_id not in metadata:
            raise TableSchemaError(f'Master Rhea ID {master_id} has no row in {paths.reactions}')
        approved, transport, source_balance = metadata[master_id]
        if transport:
            stats['skipped_transport'] += 1
            continue
        if '*' in raw_reaction:
            stats['skipped_wildcard'] += 1
            continue
        try:
            reaction = normalize_reaction(raw_reaction)
            delta = reaction_delta(reaction)
            standardized = _canonicalize_preserving_participant_order(raw_reaction)
        except (ValueError, CompositionError) as exc:
            raise TableSchemaError(f'{paths.reaction_smiles}:{line_number} invalid Rhea reaction {rhea_id}: {exc}') from exc
        balanced = delta.is_zero
        stats['standardized_rows'] += 1
        stats[f'computed_balanced_{str(balanced).lower()}'] += 1
        if source_balance in {'true', 'false'}:
            stats['source_balance_compared'] += 1
            stats['source_balance_disagrees'] += source_balance != str(balanced).lower()
        if not approved:
            stats['skipped_not_approved'] += 1
            continue
        if not balanced:
            stats['skipped_unbalanced'] += 1
            continue
        stats['changed_by_standardization'] += standardized != raw_reaction
        records.append(RheaReactionRecord(schema_version=2, source_database='rhea', source_release=str(source_release), source_master_id=master_id, source_reaction_id=rhea_id, direction=direction, ec_numbers=ecs.get(master_id, ''), reaction_smiles=raw_reaction, standardized_reaction_smiles=standardized, reaction_key=reaction.to_smiles(), is_balanced=True, is_approved=True))
        stats['with_ec'] += bool(ecs.get(master_id))
        stats[f'direction_{direction}'] += 1
        stats['output_rows'] += 1
    return (records, stats)


# biorxnrestore.data.kegg.KeggReactionRecord
@dataclass(frozen=True, slots=True)
class KeggReactionRecord:
    schema_version: int
    source_database: str
    source_release: str
    source_master_id: str
    source_reaction_id: str
    direction: str
    ec_numbers: str
    equation: str
    reaction_smiles: str
    standardized_reaction_smiles: str
    reaction_key: str
    is_balanced: bool

    def to_tsv_row(self) -> dict[str, object]:
        row = asdict(self)
        row['is_balanced'] = 'true' if self.is_balanced else 'false'
        return row


# biorxnrestore.data.kegg.build_kegg_reactions
def build_kegg_reactions(snapshot: str | Path, *, expected_sha256: str, source_release: str, direction_policy: str='recorded_only') -> tuple[list[KeggReactionRecord], Counter]:
    """Expand full reactions before skeleton extraction; never infer physiology.

    ``recorded_only`` preserves the frozen v0.2 representation. ``both`` emits
    LR/RL records with distinct directed IDs and one shared KEGG master ID.
    The equation column retains the original source equation in either case;
    the direction and reaction-SMILES columns describe the emitted orientation.
    """
    if direction_policy not in {'recorded_only', 'both'}:
        raise ValueError(f'Unsupported KEGG direction policy: {direction_policy}')
    source = Path(snapshot)
    _require_file(source)
    observed_sha256 = sha256_file(source)
    if observed_sha256 != expected_sha256:
        raise TableSchemaError(f'KEGG snapshot SHA-256 mismatch: expected {expected_sha256}, observed {observed_sha256}')
    required = {'reaction_id', 'equation', 'reaction_smiles', 'element_balanced', 'charge_balanced', 'balanced', 'status'}
    records: list[KeggReactionRecord] = []
    stats: Counter = Counter()
    seen: set[str] = set()
    with source.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        _require_columns(source, reader.fieldnames, required)
        for line_number, row in enumerate(reader, start=2):
            stats['input_rows'] += 1
            reaction_id = row['reaction_id'].strip()
            if not reaction_id:
                raise TableSchemaError(f'{source}:{line_number} has no KEGG reaction ID')
            if reaction_id in seen:
                raise TableSchemaError(f'{source}:{line_number} contains duplicate KEGG ID {reaction_id}')
            seen.add(reaction_id)
            flags = (row['element_balanced'].lower(), row['charge_balanced'].lower(), row['balanced'].lower())
            if flags != ('true', 'true', 'true') or row['status'] != 'complete_balanced':
                raise TableSchemaError(f'{source}:{line_number} contains a non-strict-balanced row: {reaction_id}')
            if '*' in row['reaction_smiles']:
                raise TableSchemaError(f'{source}:{line_number} contains a wildcard reaction: {reaction_id}')
            try:
                reaction = normalize_reaction(row['reaction_smiles'])
                delta = reaction_delta(reaction)
            except (ValueError, CompositionError) as exc:
                raise TableSchemaError(f'{source}:{line_number} invalid KEGG reaction {reaction_id}: {exc}') from exc
            if not delta.is_zero:
                raise TableSchemaError(f'{source}:{line_number} fails the local balance audit: {reaction_id}')
            directions = ('LR', 'RL') if direction_policy == 'both' else ('LR',)
            for direction in directions:
                oriented = reaction if direction == 'LR' else Reaction(reaction.products, reaction.reactants)
                raw_left, raw_right = row['reaction_smiles'].split('>>')
                oriented_raw = row['reaction_smiles'] if direction == 'LR' else f'{raw_right}>>{raw_left}'
                records.append(KeggReactionRecord(schema_version=2, source_database='kegg', source_release=source_release, source_master_id=reaction_id, source_reaction_id=f'{reaction_id}:{direction}' if direction_policy == 'both' else reaction_id, direction=direction, ec_numbers='', equation=row['equation'], reaction_smiles=oriented_raw, standardized_reaction_smiles=oriented.to_smiles(), reaction_key=oriented.key, is_balanced=True))
                stats['recorded_direction_rows' if direction == 'LR' else 'reverse_direction_rows'] += 1
                stats['output_rows'] += 1
    return (records, stats)


# biorxnrestore.data.chemical_forms.convert_balanced_reference
def convert_balanced_reference(reaction: Reaction, forms: dict[str, ChemicalForm]) -> Reaction:
    """Convert an already balanced reference and reconcile only free protons."""
    if not reaction_delta(reaction).is_zero:
        raise ValueError('Chemical-form conversion requires an already balanced reference')
    sides: list[list[str]] = []
    for original_side in (reaction.reactants, reaction.products):
        side = []
        for smiles in original_side:
            if smiles == '[H+]':
                continue
            form = forms[smiles]
            if form.original_smiles != smiles or not same_protonation_form(smiles, form.prepared_smiles):
                raise ValueError('Chemical-form conversion changes molecular chemistry or stereochemistry')
            if not form.mapped and form.prepared_smiles != smiles:
                raise ValueError('An unresolved molecular form must remain unchanged')
            before, after = (molecule_composition(smiles), molecule_composition(form.prepared_smiles))
            if dict(after.elements).get('H', 0) - dict(before.elements).get('H', 0) != after.charge - before.charge:
                raise ValueError('Molecular hydrogen and charge changes disagree')
            side.append(form.prepared_smiles)
        sides.append(side)
    if not all(sides):
        raise ValueError('Reference has an empty side after removing free protons')
    bare = Reaction(*(tuple(sorted(side)) for side in sides))
    delta = reaction_delta(bare)
    elements = dict(delta.elements)
    if any((n for element, n in elements.items() if element != 'H')) or elements.get('H', 0) != delta.charge:
        raise ValueError('Reference conversion leaves a non-proton composition difference')
    h = elements.get('H', 0)
    sides[0 if h > 0 else 1].extend(['[H+]'] * abs(h))
    converted = Reaction(*(tuple(sorted(side)) for side in sides))
    if not reaction_delta(converted).is_zero:
        raise AssertionError('Reference conversion failed exact conservation')
    return converted


# biorxnrestore.data.chemical_forms.prepare_kegg_chemical_forms
def prepare_kegg_chemical_forms(records: list[KeggReactionRecord], mapper: OfficialChemicalForms, *, direction_policy: str) -> tuple[list[KeggReactionRecord], list[dict], list[dict], Counter]:
    """Preserve every source reaction, with unresolved forms explicitly unchanged."""
    import json
    if direction_policy not in {'both', 'recorded_only'}:
        raise ValueError(f'Unsupported direction policy: {direction_policy}')
    if any((r.direction != 'LR' for r in records)) or len({r.source_master_id for r in records}) != len(records):
        raise ValueError('Chemical preparation expects one recorded full reaction per source master')
    reactions = [normalize_reaction(r.standardized_reaction_smiles) for r in records]
    molecules = sorted({s for r in reactions for s in r.reactants + r.products})
    forms = {s: mapper.resolve(s) for s in molecules}
    molecule_audit = []
    for smiles, form in forms.items():
        if mapper.resolve(form.prepared_smiles).prepared_smiles != form.prepared_smiles:
            raise ValueError('Official molecular mapping is not idempotent')
        before, after = (molecule_composition(smiles), molecule_composition(form.prepared_smiles))
        molecule_audit.append({'original_smiles': smiles, 'prepared_smiles': form.prepared_smiles, 'official_ph73_smiles': form.prepared_smiles if form.mapped else '', 'status': form.status, 'hydrogen_change': dict(after.elements).get('H', 0) - dict(before.elements).get('H', 0), 'charge_change': after.charge - before.charge, 'official_references_json': json.dumps(form.official_references)})
    output, reaction_audit = ([], [])
    stats = Counter(input_rows=len(records))
    for record, original in zip(records, reactions):
        converted = convert_balanced_reference(original, forms)
        unresolved = sorted({s for s in original.reactants + original.products if not forms[s].mapped})
        status = 'fully_mapped' if not unresolved else 'contains_unresolved_forms'
        identity_forms = {s: ChemicalForm(s, s, 'official_unchanged') for s in converted.reactants + converted.products}
        if convert_balanced_reference(converted, identity_forms) != converted:
            raise AssertionError('Converted reference is not idempotent')
        reaction_audit.append({'source_master_id': record.source_master_id, 'status': status, 'original_reaction_smiles': original.to_smiles(), 'prepared_reaction_smiles': converted.to_smiles(), 'reaction_changed': original != converted, 'unresolved_smiles_json': json.dumps(unresolved), 'changed_molecule_occurrences': sum((forms[s].prepared_smiles != s for s in original.reactants + original.products)), 'original_protons_left': original.reactants.count('[H+]'), 'original_protons_right': original.products.count('[H+]'), 'prepared_protons_left': converted.reactants.count('[H+]'), 'prepared_protons_right': converted.products.count('[H+]'), 'identity_reaction_before': original.reactants == original.products, 'identity_reaction_after': converted.reactants == converted.products})
        stats[status] += 1
        stats['changed_reactions'] += int(original != converted)
        directions = ('LR', 'RL') if direction_policy == 'both' else ('LR',)
        for direction in directions:
            prepared = converted if direction == 'LR' else Reaction(converted.products, converted.reactants)
            raw_left, raw_right = record.reaction_smiles.split('>>')
            raw = record.reaction_smiles if direction == 'LR' else f'{raw_right}>>{raw_left}'
            output.append(replace(record, source_release=f'{record.source_release};{FORM_POLICY}', direction=direction, source_reaction_id=f'{record.source_master_id}:{direction}' if direction_policy == 'both' else record.source_master_id, reaction_smiles=raw, standardized_reaction_smiles=prepared.to_smiles(), reaction_key=prepared.key))
            stats['recorded_direction_rows' if direction == 'LR' else 'reverse_direction_rows'] += 1
    stats['output_rows'] = len(output)
    return (output, molecule_audit, reaction_audit, stats)


# biorxnrestore.data.net_reactions.NET_REACTION_POLICY
NET_REACTION_POLICY = 'exact_structure_stoichiometric_cancellation_v1'


# biorxnrestore.data.net_reactions.NetReaction
@dataclass(frozen=True)
class NetReaction:
    reactants: tuple[str, ...]
    products: tuple[str, ...]
    cancelled: tuple[tuple[str, int], ...]

    @property
    def status(self) -> str:
        if not self.reactants and (not self.products):
            return 'empty_net_reaction'
        if not self.reactants or not self.products:
            return 'empty_net_side'
        return 'cancelled' if self.cancelled else 'unchanged'

    @property
    def smiles(self) -> str:
        return f"{'.'.join(self.reactants)}>>{'.'.join(self.products)}"


# biorxnrestore.data.net_reactions.cancel_identical_participants
def cancel_identical_participants(reaction_smiles: str) -> NetReaction:
    """Cancel min(left count, right count), retaining surviving source order.

    Identity is canonical isomeric SMILES with atom maps removed. Empty sides
    are represented explicitly here, never as an invalid Reaction instance.
    """
    sides = source_reaction_sides(reaction_smiles)
    shared = Counter(sides[0]) & Counter(sides[1])
    remaining = []
    for side in sides:
        to_remove = shared.copy()
        kept = []
        for molecule in side:
            if to_remove[molecule]:
                to_remove[molecule] -= 1
            else:
                kept.append(molecule)
        remaining.append(tuple(kept))
    result = NetReaction(*remaining, tuple(sorted(shared.items())))
    if delta_for_sides(*sides) != delta_for_sides(result.reactants, result.products):
        raise AssertionError('Cancellation changed element/formal-charge delta')
    return result


# biorxnrestore.data.net_reactions.prepare_net_source_rows
def prepare_net_source_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], Counter]:
    """Return extraction-ready rows, an all-source audit, and counts.

    Existing schema-v2 consumers use standardized_reaction_smiles/reaction_key;
    these fields therefore describe the net reaction in the new output only.
    Raw fields stay unchanged and the previous standardized form is explicit.
    """
    output, audit = ([], [])
    stats = Counter(input_rows=len(rows), output_rows=0)
    for row in rows:
        if row.get('net_reaction_policy'):
            raise ValueError('Source already net-prepared; use its original source table')
        if row.get('schema_version') != '2':
            raise ValueError('Net preparation requires schema-v2 source reactions')
        if row.get('source_database') not in {'rhea', 'kegg'}:
            raise ValueError('Net preparation requires Rhea or KEGG source records')
        before = row['standardized_reaction_smiles']
        reaction = Reaction(*(tuple(sorted(side)) for side in source_reaction_sides(before)))
        if '*' in before or not reaction_delta(reaction).is_zero:
            raise ValueError(f"Source is not concrete and balanced: {row['source_reaction_id']}")
        net = cancel_identical_participants(before)
        status = net.status
        stats[status] += 1
        stats['cancelled_occurrences_per_side'] += sum((n for _, n in net.cancelled))
        enriched = {**row, 'net_reaction_policy': NET_REACTION_POLICY, 'pre_cancellation_reaction_smiles': before, 'pre_cancellation_reaction_key': reaction.key, 'net_reaction_smiles': net.smiles, 'cancelled_participants_json': json.dumps(dict(net.cancelled), sort_keys=True), 'net_reaction_status': status}
        audit.append(enriched)
        if status in {'empty_net_reaction', 'empty_net_side'}:
            continue
        normalized_net = Reaction(tuple(sorted(net.reactants)), tuple(sorted(net.products)))
        output.append({**enriched, 'standardized_reaction_smiles': net.smiles, 'reaction_key': normalized_net.key})
        stats['output_rows'] += 1
    return (output, audit, stats)


# biorxnrestore.data.auxiliary_species.AuxiliarySpeciesRecord
@dataclass(frozen=True, slots=True)
class AuxiliarySpeciesRecord:
    cofactor_id: str
    name: str
    smiles: str
    smiles_source: str
    source_inchi: str
    source_inchikey_prefix: str
    rdkit_inchi_smiles: str
    rhea_occurrences: int
    rhea_candidate_count: int

    def to_tsv_row(self) -> dict[str, object]:
        return asdict(self)


# biorxnrestore.data.auxiliary_species._inchikey_prefix
def _inchikey_prefix(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f'Could not parse Rhea molecule for InChIKey: {smiles}')
    key = inchi.MolToInchiKey(molecule)
    return key.split('-', 1)[0] if key else None


# biorxnrestore.data.auxiliary_species._rhea_indexes
def _rhea_indexes(records: list[RheaReactionRecord]) -> tuple[Counter[str], dict[str, list[tuple[str, int]]], int]:
    counts: Counter[str] = Counter()
    for record in records:
        for side in record.standardized_reaction_smiles.split('>>'):
            counts.update((value for value in side.split('.') if value))
    by_prefix: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
    missing_inchikey = 0
    for smiles, count in counts.items():
        prefix = _inchikey_prefix(smiles)
        if prefix is not None:
            by_prefix[prefix].append((smiles, count))
        else:
            missing_inchikey += 1
    for values in by_prefix.values():
        values.sort(key=lambda value: (-value[1], value[0]))
    return (counts, dict(by_prefix), missing_inchikey)


# biorxnrestore.data.auxiliary_species.build_auxiliary_species
def build_auxiliary_species(cofactors_path: str | Path, rhea_records: list[RheaReactionRecord]) -> tuple[list[AuxiliarySpeciesRecord], Counter]:
    source = Path(cofactors_path)
    _require_file(source)
    counts, by_prefix, missing_inchikey = _rhea_indexes(rhea_records)
    records: list[AuxiliarySpeciesRecord] = []
    stats: Counter = Counter()
    stats['rhea_molecules_without_inchikey'] = missing_inchikey
    with source.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        _require_columns(source, reader.fieldnames, {'INCHI_PREFIX', 'INCHIKEY_PREFIX', 'NAME'})
        for index, row in enumerate(reader, start=1):
            molecule = inchi.MolFromInchi(row['INCHI_PREFIX'], sanitize=True, removeHs=True)
            if molecule is None:
                raise ValueError(f"Could not parse cofactor InChI for {row['NAME']!r}")
            rdkit_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            candidates = by_prefix.get(row['INCHIKEY_PREFIX'], [])
            if rdkit_smiles in counts:
                selected = rdkit_smiles
                source_label = 'rhea_exact_smiles'
                occurrences = counts[selected]
                candidate_count = 1
            elif candidates:
                selected, occurrences = candidates[0]
                source_label = 'rhea_inchikey_prefix_match'
                candidate_count = len(candidates)
            else:
                selected = rdkit_smiles
                source_label = 'rdkit_inchi_fallback'
                occurrences = 0
                candidate_count = 0
            records.append(AuxiliarySpeciesRecord(cofactor_id=f'RRCOF{index:04d}', name=row['NAME'], smiles=selected, smiles_source=source_label, source_inchi=row['INCHI_PREFIX'], source_inchikey_prefix=row['INCHIKEY_PREFIX'], rdkit_inchi_smiles=rdkit_smiles, rhea_occurrences=occurrences, rhea_candidate_count=candidate_count))
            stats[source_label] += 1
    stats['output_rows'] = len(records)
    return (records, stats)


# biorxnrestore.data.auxiliary_species.write_auxiliary_species
def write_auxiliary_species(path: str | Path, records: list[AuxiliarySpeciesRecord], force: bool=False) -> None:
    destination = Path(path)
    if destination.exists() and (not force):
        raise FileExistsError(f'Refusing to overwrite auxiliary-species table: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, delimiter='\t', fieldnames=['schema_version', *[field.name for field in fields(AuxiliarySpeciesRecord)]], lineterminator='\n')
        writer.writeheader()
        writer.writerows(({'schema_version': 2, **record.to_tsv_row()} for record in records))


# biorxnrestore.benchmarks.controlled.ControlledBenchmarkConfig
@dataclass(frozen=True, slots=True)
class ControlledBenchmarkConfig:
    min_heavy_atoms_after_coa_mask: int = 3
    min_carbon_atoms_after_coa_mask: int = 1
    mcs_product_ratio_threshold: float = 0.4
    mcs_timeout_seconds: int = 1
    frozen_mcs_exclusion_pair_sha256: tuple[str, ...] = ()
    include_empty_msp: bool = False

    def __post_init__(self) -> None:
        if self.min_heavy_atoms_after_coa_mask < 1:
            raise ValueError('Minimum heavy-atom count must be positive')
        if self.min_carbon_atoms_after_coa_mask < 1:
            raise ValueError('Minimum carbon-atom count must be positive')
        if not 0 <= self.mcs_product_ratio_threshold < 1:
            raise ValueError('MCS product-ratio threshold must be in [0, 1)')
        if self.mcs_timeout_seconds < 1:
            raise ValueError('MCS timeout must be at least one second')
        if any((len(value) != 64 or any((character not in '0123456789abcdef' for character in value)) for value in self.frozen_mcs_exclusion_pair_sha256)):
            raise ValueError('Frozen MCS exclusion keys must be lowercase SHA-256')


# biorxnrestore.benchmarks.controlled.ControlledBenchmarkRecord
@dataclass(frozen=True, slots=True)
class ControlledBenchmarkRecord:
    schema_version: int
    source_database: str
    source_release: str
    case_id: str
    source_master_id: str
    source_reaction_id: str
    direction: str
    ec_numbers: str
    standardized_reaction_smiles: str
    skeleton_reaction_smiles: str
    main_reactants_smiles: str
    main_product_smiles: str
    removed_reactant_smiles: str
    removed_product_smiles: str
    removed_species_count: int
    main_reactant_count: int
    reactant_candidate_count: int
    product_candidate_count: int
    valid_pair_count: int
    valid_product_count: int
    product_rank: int
    max_mcs_atoms: int
    max_mcs_product_ratio: float
    contains_coa_masked_molecule: bool

    def to_tsv_row(self) -> dict[str, object]:
        row = asdict(self)
        row['max_mcs_product_ratio'] = f'{self.max_mcs_product_ratio:.6f}'
        row['contains_coa_masked_molecule'] = 'true' if self.contains_coa_masked_molecule else 'false'
        return row


# biorxnrestore.benchmarks.controlled._MoleculeInfo
@dataclass(frozen=True, slots=True)
class _MoleculeInfo:
    smiles: str
    molecule: Chem.Mol
    mcs_molecule: Chem.Mol
    heavy_atoms: int
    mcs_heavy_atoms: int
    mcs_carbon_atoms: int
    mcs_element_counts: dict[int, int]
    coa_masked: bool


# biorxnrestore.benchmarks.controlled._Candidate
@dataclass(frozen=True, slots=True)
class _Candidate:
    side_index: int
    smiles: str
    info: _MoleculeInfo


# biorxnrestore.benchmarks.controlled._carbon_atom_count
def _carbon_atom_count(molecule: Chem.Mol) -> int:
    return sum((atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()))


# biorxnrestore.benchmarks.controlled._element_counts
def _element_counts(molecule: Chem.Mol) -> dict[int, int]:
    counts: defaultdict[int, int] = defaultdict(int)
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() > 1:
            counts[atom.GetAtomicNum()] += 1
    return dict(counts)


# biorxnrestore.benchmarks.controlled._heavy_atom_count
def _heavy_atom_count(molecule: Chem.Mol) -> int:
    return sum((atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()))


# biorxnrestore.benchmarks.controlled._mask_coa
def _mask_coa(molecule: Chem.Mol, coa_query: Chem.Mol | None) -> tuple[Chem.Mol, bool]:
    if coa_query is None or not molecule.HasSubstructMatch(coa_query):
        return (molecule, False)
    masked = Chem.DeleteSubstructs(molecule, coa_query, onlyFrags=False)
    if masked.GetNumAtoms():
        Chem.SanitizeMol(masked)
    return (masked, True)


# biorxnrestore.benchmarks.controlled._mcs_atom_count
def _mcs_atom_count(reactant: Chem.Mol, product: Chem.Mol, timeout_seconds: int) -> tuple[int, bool]:
    if rdRascalMCES is not None:
        options = rdRascalMCES.RascalOptions()
        options.similarityThreshold = 0.0
        options.returnEmptyMCES = True
        options.ignoreBondOrders = True
        options.ringMatchesRingOnly = True
        options.singleLargestFrag = True
        options.timeout = timeout_seconds
        results = rdRascalMCES.FindMCES(reactant, product, options)
        if not results:
            return (0, False)
        best = max(results, key=lambda result: result.largestFragmentSize)
        return (best.largestFragmentSize, bool(best.timedOut))
    result = rdFMCS.FindMCS([reactant, product], atomCompare=rdFMCS.AtomCompare.CompareElements, bondCompare=rdFMCS.BondCompare.CompareAny, ringMatchesRingOnly=True, completeRingsOnly=True, matchValences=False, timeout=timeout_seconds)
    return (result.numAtoms, bool(getattr(result, 'canceled', False)))


# biorxnrestore.benchmarks.controlled._MoleculeContext
class _MoleculeContext:

    def __init__(self, cofactors: set[str], coa_query: Chem.Mol | None, config: ControlledBenchmarkConfig) -> None:
        self.cofactors = cofactors
        self.coa_query = coa_query
        self.config = config
        self.info_cache: dict[str, _MoleculeInfo | None] = {}
        self.mcs_cache: dict[tuple[str, str, int], tuple[int, bool]] = {}

    def info(self, smiles: str) -> _MoleculeInfo | None:
        if smiles not in self.info_cache:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                self.info_cache[smiles] = None
            else:
                mcs_molecule, masked = _mask_coa(molecule, self.coa_query)
                self.info_cache[smiles] = _MoleculeInfo(smiles=smiles, molecule=molecule, mcs_molecule=mcs_molecule, heavy_atoms=_heavy_atom_count(molecule), mcs_heavy_atoms=_heavy_atom_count(mcs_molecule), mcs_carbon_atoms=_carbon_atom_count(mcs_molecule), mcs_element_counts=_element_counts(mcs_molecule), coa_masked=masked)
        return self.info_cache[smiles]

    def is_main_candidate(self, smiles: str) -> bool:
        if smiles in self.cofactors:
            return False
        info = self.info(smiles)
        return bool(info is not None and info.mcs_heavy_atoms >= self.config.min_heavy_atoms_after_coa_mask and (info.mcs_carbon_atoms >= self.config.min_carbon_atoms_after_coa_mask))

    def mcs(self, reactant: _Candidate, product: _Candidate) -> tuple[int, bool]:
        key = tuple(sorted((reactant.smiles, product.smiles))) + (self.config.mcs_timeout_seconds,)
        if key not in self.mcs_cache:
            self.mcs_cache[key] = _mcs_atom_count(reactant.info.mcs_molecule, product.info.mcs_molecule, self.config.mcs_timeout_seconds)
        return self.mcs_cache[key]


# biorxnrestore.benchmarks.controlled._pair_sha256
def _pair_sha256(left: str, right: str) -> str:
    return hashlib.sha256('\n'.join(sorted((left, right))).encode('utf-8')).hexdigest()


# biorxnrestore.benchmarks.controlled._apply_frozen_mcs_exclusions
def _apply_frozen_mcs_exclusions(records: list[ControlledBenchmarkRecord], excluded_pair_hashes: tuple[str, ...], stats: Counter) -> list[ControlledBenchmarkRecord]:
    """Replay the audited timeout boundary of the frozen Rhea benchmark.

    The historical one-second RASCAL-MCES wall-clock boundary contains one
    molecule pair whose completion is hardware-load dependent.  Persisting its
    source-neutral molecule-pair digest makes the controlled benchmark stable
    without keying behavior to a database record identifier.
    """
    excluded = set(excluded_pair_hashes)
    if not excluded:
        return records
    retained: list[ControlledBenchmarkRecord] = []
    for record in records:
        main_reactants = [value for value in record.main_reactants_smiles.split('.') if value]
        hashes = {_pair_sha256(reactant, record.main_product_smiles) for reactant in main_reactants}
        if hashes & excluded:
            stats['frozen_mcs_excluded_rows'] += 1
        else:
            retained.append(record)
    return retained


# biorxnrestore.benchmarks.controlled._candidates
def _candidates(side: list[str], context: _MoleculeContext) -> tuple[list[_Candidate], int]:
    candidates: list[_Candidate] = []
    failures = 0
    for side_index, smiles in enumerate(side):
        info = context.info(smiles)
        if info is None:
            failures += 1
        elif context.is_main_candidate(smiles):
            candidates.append(_Candidate(side_index, smiles, info))
    return (candidates, failures)


# biorxnrestore.benchmarks.controlled._ValidPair
@dataclass(frozen=True, slots=True)
class _ValidPair:
    reactant: _Candidate
    product: _Candidate
    mcs_atoms: int
    mcs_product_ratio: float


# biorxnrestore.benchmarks.controlled._ProductGroup
@dataclass(frozen=True, slots=True)
class _ProductGroup:
    product: _Candidate
    pairs: tuple[_ValidPair, ...]
    max_mcs_atoms: int
    max_mcs_product_ratio: float


# biorxnrestore.benchmarks.controlled._product_groups
def _product_groups(pairs: list[_ValidPair]) -> list[_ProductGroup]:
    grouped: defaultdict[int, list[_ValidPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.product.side_index].append(pair)
    groups = []
    for values in grouped.values():
        ordered = tuple(sorted(values, key=lambda pair: pair.reactant.side_index))
        groups.append(_ProductGroup(product=ordered[0].product, pairs=ordered, max_mcs_atoms=max((pair.mcs_atoms for pair in ordered)), max_mcs_product_ratio=max((pair.mcs_product_ratio for pair in ordered))))
    return sorted(groups, key=lambda group: (-group.max_mcs_product_ratio, -group.max_mcs_atoms, group.product.side_index))


# biorxnrestore.benchmarks.controlled._valid_pairs
def _valid_pairs(reactants: list[_Candidate], products: list[_Candidate], context: _MoleculeContext, stats: Counter) -> list[_ValidPair]:
    valid: list[_ValidPair] = []
    threshold = context.config.mcs_product_ratio_threshold
    for reactant in reactants:
        for product in products:
            stats['candidate_pairs'] += 1
            if product.info.mcs_heavy_atoms == 0:
                stats['skipped_empty_masked_product'] += 1
                continue
            upper_bound = sum((min(count, product.info.mcs_element_counts.get(element, 0)) for element, count in reactant.info.mcs_element_counts.items()))
            if upper_bound / product.info.mcs_heavy_atoms <= threshold:
                stats['skipped_mcs_upper_bound'] += 1
                continue
            mcs_atoms, timed_out = context.mcs(reactant, product)
            if timed_out:
                stats['mcs_timeout_pairs'] += 1
                continue
            ratio = mcs_atoms / product.info.mcs_heavy_atoms
            if ratio > threshold:
                valid.append(_ValidPair(reactant, product, mcs_atoms, ratio))
    return sorted(valid, key=lambda pair: (pair.product.side_index, pair.reactant.side_index, -pair.mcs_product_ratio, -pair.mcs_atoms))


# biorxnrestore.benchmarks.controlled._records_for_reaction
def _records_for_reaction(source: dict[str, str], context: _MoleculeContext, stats: Counter) -> list[ControlledBenchmarkRecord]:
    sides = source['standardized_reaction_smiles'].split('>>')
    if len(sides) != 2:
        raise TableSchemaError(f"Invalid standardized reaction for {source.get('source_database', 'unknown')}:{source.get('source_reaction_id', '')}")
    reactants = [value for value in sides[0].split('.') if value]
    products = [value for value in sides[1].split('.') if value]
    reactant_candidates, reactant_failures = _candidates(reactants, context)
    product_candidates, product_failures = _candidates(products, context)
    stats['molecule_parse_failures'] += reactant_failures + product_failures
    if not reactant_candidates:
        stats['skipped_no_reactant_candidates'] += 1
        return []
    if not product_candidates:
        stats['skipped_no_product_candidates'] += 1
        return []
    valid_pairs = _valid_pairs(reactant_candidates, product_candidates, context, stats)
    if not valid_pairs:
        stats['skipped_no_mcs_valid_pair'] += 1
        return []
    groups = _product_groups(valid_pairs)
    records: list[ControlledBenchmarkRecord] = []
    for product_rank, group in enumerate(groups, start=1):
        selected_left = {pair.reactant.side_index for pair in group.pairs}
        selected_right = {group.product.side_index}
        main_left = [value for index, value in enumerate(reactants) if index in selected_left]
        removed_left = [value for index, value in enumerate(reactants) if index not in selected_left]
        removed_right = [value for index, value in enumerate(products) if index not in selected_right]
        removed_count = len(removed_left) + len(removed_right)
        if not removed_count:
            if not context.config.include_empty_msp:
                stats['skipped_no_removed_species'] += 1
                continue
            if Counter(main_left) == Counter((group.product.smiles,)):
                stats['skipped_empty_msp_no_transformation'] += 1
                continue
            stats['included_empty_msp'] += 1
        records.append(ControlledBenchmarkRecord(schema_version=2, source_database=source['source_database'], source_release=source['source_release'], case_id='', source_master_id=source['source_master_id'], source_reaction_id=source['source_reaction_id'], direction=source['direction'], ec_numbers=source.get('ec_numbers', ''), standardized_reaction_smiles=source['standardized_reaction_smiles'], skeleton_reaction_smiles=f"{'.'.join(main_left)}>>{group.product.smiles}", main_reactants_smiles='.'.join(main_left), main_product_smiles=group.product.smiles, removed_reactant_smiles='.'.join(removed_left), removed_product_smiles='.'.join(removed_right), removed_species_count=removed_count, main_reactant_count=len(main_left), reactant_candidate_count=len(reactant_candidates), product_candidate_count=len(product_candidates), valid_pair_count=len(valid_pairs), valid_product_count=len(groups), product_rank=product_rank, max_mcs_atoms=group.max_mcs_atoms, max_mcs_product_ratio=group.max_mcs_product_ratio, contains_coa_masked_molecule=any((pair.reactant.info.coa_masked or pair.product.info.coa_masked for pair in group.pairs))))
    if records:
        stats['multi_product_reactions'] += len(groups) > 1
        stats['multi_reactant_rows'] += sum((record.main_reactant_count != 1 for record in records))
    return records


# biorxnrestore.benchmarks.controlled.build_controlled_benchmark
def build_controlled_benchmark(source_rows: Iterable[dict[str, str]], cofactors: set[str], coa_query: Chem.Mol | None, config: ControlledBenchmarkConfig | None=None) -> tuple[list[ControlledBenchmarkRecord], Counter]:
    settings = config or ControlledBenchmarkConfig()
    context = _MoleculeContext(cofactors, coa_query, settings)
    records: list[ControlledBenchmarkRecord] = []
    stats: Counter = Counter()
    for source in source_rows:
        stats['input_rows'] += 1
        generated = _records_for_reaction(source, context, stats)
        records.extend(generated)
        stats['reactions_with_output'] += bool(generated)
    records = _apply_frozen_mcs_exclusions(records, settings.frozen_mcs_exclusion_pair_sha256, stats)
    source_counts: defaultdict[str, int] = defaultdict(int)
    identified: list[ControlledBenchmarkRecord] = []
    prefixes = {'rhea': 'RBNP', 'kegg': 'KBNP'}
    for record in records:
        source_counts[record.source_database] += 1
        prefix = prefixes.get(record.source_database, 'SBNP')
        identified.append(ControlledBenchmarkRecord(**{**asdict(record), 'case_id': f'{prefix}{source_counts[record.source_database]:06d}'}))
    records = identified
    stats['output_rows'] = len(records)
    stats['coa_masked_rows'] = sum((record.contains_coa_masked_molecule for record in records))
    return (records, stats)


# biorxnrestore.benchmarks.controlled.read_auxiliary_species
def read_auxiliary_species(path: str | Path) -> tuple[set[str], Chem.Mol | None]:
    source = Path(path)
    _require_file(source)
    cofactors: set[str] = set()
    coa_query: Chem.Mol | None = None
    with source.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        _require_columns(source, reader.fieldnames, {'schema_version', 'name', 'smiles'})
        for row in reader:
            if row['schema_version'] != '2':
                raise TableSchemaError(f"Unsupported auxiliary-species schema in {source}: {row['schema_version']}")
            smiles = row['smiles']
            if not smiles:
                continue
            cofactors.add(smiles)
            if row['name'].strip().lower() == 'coa':
                coa_query = Chem.MolFromSmiles(smiles)
                if coa_query is None:
                    raise TableSchemaError(f'Invalid CoA SMILES in {source}: {smiles}')
    return (cofactors, coa_query)


# biorxnrestore.benchmarks.evidence.build_evidence_library
def build_evidence_library(records: list[ControlledBenchmarkRecord]) -> list[EvidenceRecord]:
    evidence: list[EvidenceRecord] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        evidence_id = f'{record.source_database.upper()}:{record.case_id}'
        if evidence_id in seen_ids:
            raise ValueError(f'Duplicate source-qualified evidence ID: {evidence_id}')
        seen_ids.add(evidence_id)
        evidence.append(EvidenceRecord(index=index, evidence_id=evidence_id, source_case_id=record.case_id, source_database=record.source_database, source_release=record.source_release, source_master_id=record.source_master_id, source_reaction_id=record.source_reaction_id, direction=record.direction, skeleton=normalize_reaction(record.skeleton_reaction_smiles), raw_retrieval_smiles=record.skeleton_reaction_smiles, full_reaction=normalize_reaction(record.standardized_reaction_smiles), msp=msp_from_columns(record.removed_reactant_smiles, record.removed_product_smiles), ec_numbers=tuple(sorted((value for value in record.ec_numbers.replace(';', '|').split('|') if value)))))
    return evidence


# biorxnrestore.io.datasets.write_clean_evidence
def write_clean_evidence(path: str | Path, evidence: list[EvidenceRecord], force: bool=False) -> None:
    table_path = Path(path)
    if table_path.exists() and (not force):
        raise FileExistsError(f'Refusing to overwrite evidence library: {table_path}')
    table_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['schema_version', 'evidence_index', 'evidence_id', 'source_case_id', 'source_database', 'source_release', 'source_master_id', 'source_reaction_id', 'direction', 'ec_numbers', 'retrieval_reaction_smiles', 'full_reaction_smiles', 'removed_reactant_smiles', 'removed_product_smiles']
    with table_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, delimiter='\t', fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        for record in evidence:
            writer.writerow({'schema_version': 2, 'evidence_index': record.index, 'evidence_id': record.evidence_id, 'source_case_id': record.source_case_id, 'source_database': record.source_database, 'source_release': record.source_release, 'source_master_id': record.source_master_id, 'source_reaction_id': record.source_reaction_id, 'direction': record.direction, 'ec_numbers': '|'.join(record.ec_numbers), 'retrieval_reaction_smiles': record.raw_retrieval_smiles, 'full_reaction_smiles': record.full_reaction.to_smiles(), 'removed_reactant_smiles': '.'.join(record.msp.reactants), 'removed_product_smiles': '.'.join(record.msp.products)})


# biorxnrestore.retrieval.index.write_fingerprint_index
def write_fingerprint_index(path: str | Path, index: FingerprintIndex, force: bool=False) -> None:
    target = Path(path)
    if target.exists() and (not force):
        raise FileExistsError(f'Refusing to overwrite fingerprint index: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+b', dir=target.parent, prefix=f'.{target.name}.', suffix='.tmp', delete=False) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, schema_version=np.asarray(index.schema_version, dtype=np.int64), encoder=np.asarray(index.encoder), record_ids=np.asarray(index.record_ids, dtype=np.str_), matrix=index.fingerprints.matrix, valid=index.fingerprints.valid.astype(np.bool_, copy=False))
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


# biorxnrestore.restoration.parent_templates.apply
@lru_cache(maxsize=512)
def apply(smarts, inputs):
    rxn = reaction_template(smarts)
    generated, invalid = (set(), 0)
    for products in rxn.RunReactants(tuple((mol(s) for s in inputs)), maxProducts=0):
        try:
            values = tuple((canonical(p) for p in products))
            if any(('.' in value for value in values)):
                invalid += 1
            else:
                generated.add(values)
        except (ValueError, RuntimeError):
            invalid += 1
    return (sorted(generated), invalid)


# biorxnrestore.restoration.parent_templates.make_template
def make_template(parent, mapped):
    from rdchiral.template_extractor import get_changed_atoms, get_fragments_for_changed_atoms, expand_changed_atom_tags
    full = normalize_reaction(parent['full'])
    p = normalize_reaction(parent['parent'])
    assert normalize_reaction(mapped) == full, 'mapping_changed_source'
    mapped_sides = []
    atom_maps = []
    for side, target in zip(mapped.split('>>'), (full.reactants, full.products)):
        queues = defaultdict(deque)
        maps = {}
        for value in side.split('.'):
            m = Chem.MolFromSmiles(value)
            for a in m.GetAtoms():
                assert not a.GetIsotope(), 'source_isotope_unsupported'
                if a.GetAtomicNum() > 1:
                    tag = a.GetAtomMapNum()
                    assert tag and tag not in maps, 'invalid_heavy_atom_mapping'
                    maps[tag] = a.GetAtomicNum()
            queues[canonical(m)].append(m)
        mapped_sides.append([queues[value].popleft() for value in target])
        atom_maps.append(maps)
    for tag in atom_maps[0].keys() & atom_maps[1].keys():
        assert atom_maps[0][tag] == atom_maps[1][tag], 'mapped_element_mismatch'
    left = [mapped_sides[0][i] for i in parent['full_left_slots']]
    right = [mapped_sides[1][i] for i in parent['full_right_slots']]
    right_maps = {a.GetAtomMapNum() for m in right for a in m.GetAtoms() if a.GetAtomicNum() > 1}
    assert right_maps <= atom_maps[0].keys(), 'unaccounted_parent_product_atoms'
    donor_slots = [i for i, m in enumerate(mapped_sides[0]) if i not in parent['full_left_slots'] and any((a.GetAtomMapNum() in right_maps for a in m.GetAtoms() if a.GetAtomicNum() > 1))]
    left.extend((mapped_sides[0][i] for i in donor_slots))
    inputs = [canonical(m) for m in left]
    np.random.seed(int(sha256(parent['full'].encode()).hexdigest()[:8], 16))
    _, tags, err = get_changed_atoms(left, right)
    assert not err, 'changed_atoms_failed'
    tags = set(tags)
    for m in left + right:
        mtags = {str(a.GetAtomMapNum()) for a in m.GetAtoms() if a.GetAtomMapNum()}
        if not mtags & tags:
            tags.update(mtags)
    tags = sorted(tags, key=int)
    rf, _, _ = get_fragments_for_changed_atoms(left, tags, radius=1, category='reactants')
    pf, _, _ = get_fragments_for_changed_atoms(right, tags, radius=0, category='products', expansion=expand_changed_atom_tags(tags, rf))
    smarts = rf + '>>' + pf
    rxn = reaction_template(smarts)
    assert rxn.GetNumReactantTemplates() == len(left), 'input_role_count_changed'
    assert rxn.GetNumProductTemplates() == len(right), 'output_role_count_changed'
    generated, invalid = apply(smarts, tuple(inputs))
    assert p.products in generated, 'source_parent_roundtrip_failed'
    return dict(status='ready', smarts=smarts, inputs=inputs, donor_full_left_slots=donor_slots, product_count=len(right), self_roundtrip=True, invalid_self_products=invalid, reactant_patterns=[Chem.MolToSmarts(rxn.GetReactantTemplate(i)) for i in range(len(left))], product_patterns=[Chem.MolToSmarts(rxn.GetProductTemplate(i)) for i in range(len(right))])

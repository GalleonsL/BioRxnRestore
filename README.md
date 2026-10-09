# BioRxnRestore

BioRxnRestore helps recover the cofactors, cosubstrates and coproducts missing from a biochemical reaction skeleton. It searches known reactions for direct matches and similar transformations, then returns balanced reaction candidates along with scores and the supporting source reactions.

## Installation

BioRxnRestore requires Python 3.11 and has been tested on Linux using a CPU. From the repository directory, create an environment and install the dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

The repository includes a ready-to-use Rhea 139 database in `data/`. KEGG data are not bundled because they are subject to separate [licensing terms](https://www.genome.jp/kegg/legal.html). To add KEGG, supply your own two-column reaction SMILES table and build the combined database using the script described in [Adding KEGG](#adding-kegg).

Give BioRxnRestore a reaction SMILES containing the known substrates and one observed product. For example, this command completes **allyl alcohol → acrolein**:

```bash
python scripts/restore.py --data-dir data \
  --reaction 'C=CCO>>C=CC=O' --output result.json
```

Output:

```text
BioRxnRestore | default | 23 candidates
Saved: result.json

Rank  | Method     | Score  | Completed reaction
------+------------+--------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Top 1 | Match      | 1.0000 | C=CCO.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)([O-])[O-])[C@@H]3O)[C@@H](O)[C@H]2O)c1 -> C=CC=O.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)([O-])[O-])[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.[H+]
Top 2 | Similarity | 0.4364 | C=CCO.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1 -> C=CC=O.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.[H+]
Top 3 | Similarity | 0.3333 | C=CCO.O=O -> C=CC=O.OO
```

For benchmark evaluation, add `--strict-holdout` to exclude related source reactions. You can also supply known complete references with `--reference` and source IDs with `--exclude-source`. When evaluating against reference labels, include all known references for each query. Run `python scripts/restore.py --help` to see the available options.

Selected paper examples—CEA–DGPC, catechol and NMA-Glc pathways—are in [examples/examples.json](examples/examples.json). These use the paper’s Rhea–KEGG library; the bundled database uses Rhea only.

## Database preparation

If you want to rebuild the database, start with the original source files listed below. The Rhea build uses the input snapshots listed below; KEGG reactions can be supplied in your own table.

### Rhea

For Rhea, put the following seven TSV files in one directory. The two chemical-form tables are already available in `data/` and can be copied from there.

| Input file | Source | Contents |
| --- | --- | --- |
| `rhea-reaction-smiles.tsv` | Rhea 139 | Complete reactions as SMILES |
| `rhea-directions.tsv` | Rhea 139 | Reaction directions and master IDs |
| `rhea_reactions.tsv` | Project's Rhea annotation export | Approval and transport status |
| `rhea2ec.tsv` | Rhea 139 | Reaction-to-EC associations |
| `cofactors_biochem.tsv` | RetroRules | Cofactor names and structures |
| `chebi_pH7_3_mapping.tsv` | Rhea 139; also included in `data/` | Chemical-form mappings |
| `rhea-chebi-smiles.tsv` | Rhea 139; also included in `data/` | ChEBI structures used by those mappings |

Install the additional dependencies and run the build, replacing `/path/to/raw` with the directory containing those files:

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-build.txt
python scripts/build_database.py --input-dir /path/to/raw --output-dir data/rebuilt
```

The script cleans the reactions and uses them to build the skeleton library, fingerprints and restoration templates. It saves everything in `data/rebuilt/`. To use this database, change the example command above to `--data-dir data/rebuilt`. If you want to run source cleaning and library construction separately, use `--stage sources` followed by `--stage library`.

### Adding KEGG

Provide a tab-separated file with two columns: `reaction_id` (the KEGG reaction ID) and `reaction_smiles` (the complete reaction, including cofactors and repeated molecules for stoichiometric coefficients). See [the example table](examples/kegg_reactions.tsv):

```tsv
reaction_id	reaction_smiles
R01082	O=C(O)[C@@H](O)CC(=O)O>>O=C(O)/C=C/C(=O)O.O
```

This example represents [KEGG R01082](https://www.kegg.jp/entry/R01082): (S)-malate → fumarate + water, written here using neutral acid forms. Provide additional KEGG reactions obtained under the applicable terms. No balance flags are needed: the script checks structures and element/charge conservation and stops on invalid rows.

With the seven Rhea input files above in `/path/to/raw`, run:

```bash
python scripts/build_database.py --database rhea-kegg \
  --input-dir /path/to/raw --kegg-file /path/to/kegg_reactions.tsv \
  --output-dir data/rhea-kegg
```

Then use `--data-dir data/rhea-kegg` for reaction completion. The build records your input file's checksum; optionally add `--kegg-release YYYY-MM-DD` to record its source date.

## Repository structure

```text
biorxnrestore/    Reaction completion and preprocessing algorithms
scripts/         Command-line entrypoints
data/            Source manifests and bundled Rhea database
examples/        Example inputs and results
tests/           Tests
```

Run the tests with `python -m unittest discover -s tests -v`.

## Citation

Please cite [BioRxnRestore](https://github.com/GalleonsL/BioRxnRestore) and specify the version or commit used in your work.

## License

The code is released under the [MIT License](LICENSE). The Rhea data are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with attribution and processing details in [data/LICENSE_DATA.txt](data/LICENSE_DATA.txt). Please credit [Rhea](https://doi.org/10.1093/nar/gkab1016), [ChEBI](https://www.ebi.ac.uk/chebi/about) and [RetroRules](https://retrorules.org/download).

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

The repository includes a prepared Rhea 139 database in `data/`, so you can run the example below after installing the dependencies.

Give BioRxnRestore a reaction SMILES containing the known substrates and one observed product. For example, the following command completes **isoamyl acetate → acetate**:

```bash
python scripts/restore.py --data-dir data \
  --reaction 'CC(=O)OCCC(C)C>>CC(=O)[O-]' --output result.json
```

Output:

```text
BioRxnRestore | default | 3 candidates
Saved: result.json

Rank  | Method     | Score  | Completed reaction
------+------------+--------+-----------------------------------------------
Top 1 | Match      | 1.0000 | CC(=O)OCCC(C)C.O -> CC(=O)[O-].CC(C)CCO.[H+]
Top 2 | Similarity | 0.4065 | CC(=O)OCCC(C)C.CS -> CC(=O)[O-].CSCCC(C)C.[H+]
Top 3 | Similarity | 0.4065 | CC(=O)OCCC(C)C.[SH-] -> CC(=O)[O-].CC(C)CCS
```

In the first candidate, BioRxnRestore adds water to the reactants and isoamyl alcohol and a proton to the products. You can inspect the top three candidates in the terminal, or open `result.json` for the full candidate list and supporting reactions.

For benchmark evaluation, add `--strict-holdout` to exclude related source reactions. You can also supply known complete references with `--reference` and source IDs with `--exclude-source`. When evaluating against reference labels, include all known references for each query. Run `python scripts/restore.py --help` to see the available options.

Selected paper examples—CEA–DGPC, catechol and NMA-Glc pathways—are in [examples/examples.json](examples/examples.json). These use the paper’s Rhea–KEGG library; the bundled database uses Rhea only.

## Database preparation

If you want to rebuild the database, start with the original source files listed below. The build script checks their versions using saved checksums, so it expects the same input snapshots used in this project.

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

To include KEGG, you will also need `kegg_balanced_reaction_smiles_full.tsv`, the prepared reaction table used in this project. This snapshot is dated 2026-08-14 and should be obtained under your KEGG authorization. Each row describes a complete reaction as SMILES and records whether its elements and charge are balanced:

| Required columns | Contents |
| --- | --- |
| `reaction_id`, `equation` | KEGG reaction ID and source equation |
| `reaction_smiles` | Complete reaction, including stoichiometric multiplicities |
| `element_balanced`, `charge_balanced`, `balanced` | All three must be `true` |
| `status` | Must be `complete_balanced` |

Put this table alongside the seven Rhea input files, then build the combined library:

```bash
python scripts/build_database.py --database rhea-kegg \
  --input-dir /path/to/raw --output-dir data/rhea-kegg
```

This command reads the table you provide locally; downloading and preparing the KEGG table is a separate step. Once the build finishes, use `--data-dir data/rhea-kegg` to run completion with the combined library.

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

"""Build the challenge ZIP for one method.

    python scripts/make_submission.py --method method3_gbdt --team myteam \\
        --outputs outputs/method3_gbdt/test --doc path/to/Documentation_template.md \\
        --data-dir dataset

Layout produced (as required by the challenge):

    <team>_submission.zip
    |-- output/matching_results.tsv, candidate_pairs.tsv
    |-- code/business_entity_resolution/{src/, README.md, requirements.txt}
    `-- Documentation_template.md

``src/`` receives the shared core plus every method package the chosen method imports, so the
archive is self-contained. The two output files are re-validated before zipping.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from er_common.io import load_split, parse_id_list, read_tsv, validate_submission  # noqa: E402

DEPENDENCIES = {
    "method1_fuzzy_rules": ["er_common", "method1_fuzzy_rules"],
    "method2_tfidf_retrieval": ["er_common", "method2_tfidf_retrieval"],
    "method3_gbdt": ["er_common", "method2_tfidf_retrieval", "method3_gbdt"],
}
EXCLUDE = {"__pycache__", "synth.py"}  # the synthetic generator is not part of the pipeline


def _check_outputs(outputs: Path, data_dir: Path | None) -> None:
    m = read_tsv(outputs / "matching_results.tsv")
    c = read_tsv(outputs / "candidate_pairs.tsv")
    if list(m.columns) != ["source1_entity_id", "matched_entity_ids"]:
        raise SystemExit(f"bad header in matching_results.tsv: {list(m.columns)}")
    if list(c.columns) != ["source1_entity_id", "candidate_entity_ids"]:
        raise SystemExit(f"bad header in candidate_pairs.tsv: {list(c.columns)}")
    matches = {s: list(parse_id_list(v)) for s, v in zip(m.iloc[:, 0], m.iloc[:, 1], strict=True)}
    cands = {s: list(parse_id_list(v)) for s, v in zip(c.iloc[:, 0], c.iloc[:, 1], strict=True)}
    if data_dir is not None:
        test = load_split(data_dir, "test")
        s1_ids, s23_ids = test.s1_ids, test.s23["entity_id"].tolist()
        if m.iloc[:, 0].tolist() != s1_ids or c.iloc[:, 0].tolist() != s1_ids:
            raise SystemExit("output rows do not cover the test Source 1 file exactly once each")
    else:
        s1_ids = m.iloc[:, 0].tolist()
        s23_ids = sorted({x for v in cands.values() for x in v})
        print("warning: --data-dir not given; ids checked for format/consistency only", file=sys.stderr)
    errors = validate_submission(matches, cands, s1_ids, s23_ids)
    if errors:
        raise SystemExit("output validation failed:\n  " + "\n  ".join(errors[:20]))


def _package_readme(method: str) -> str:
    body = (ROOT / method / "README.md").read_text(encoding="utf-8")
    return f"""# Business Entity Resolution -- `{method}`

## Reproduce end to end

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# challenge data expected at ./dataset/{{train,test}}/*.tsv (same layout as distributed)
cd src
python -m {method} all --data-dir ../dataset --out-dir ../outputs
```

* `../outputs/test/matching_results.tsv` and `../outputs/test/candidate_pairs.tsv` are the
  submission files (identical to `output/` in this archive for the same data and code).
* `../outputs/validation/report.json` holds the cross-validated F0.5 on the training data.
* `python -m {method} validate|predict` run the two halves separately.
* Runs are deterministic (fixed seeds, deterministic LightGBM, sorted tie-breaks); the manifest
  in `../outputs/test/manifest.json` records data hashes, package versions and configuration.

---

{body}"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=sorted(DEPENDENCIES))
    ap.add_argument("--team", required=True)
    ap.add_argument("--outputs", type=Path, default=None, help="dir with the two TSVs (default outputs/<method>/test)")
    ap.add_argument("--doc", type=Path, default=None, help="filled Documentation_template.md (or .pdf)")
    ap.add_argument("--data-dir", type=Path, default=None, help="dataset dir, enables full id validation")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "submissions")
    args = ap.parse_args()

    outputs = args.outputs or ROOT / "outputs" / args.method / "test"
    _check_outputs(outputs, args.data_dir)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        (stage / "output").mkdir()
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            shutil.copy2(outputs / f, stage / "output" / f)
        code = stage / "code" / "business_entity_resolution"
        for pkg in DEPENDENCIES[args.method]:
            shutil.copytree(ROOT / pkg, code / "src" / pkg, ignore=lambda _d, names: [n for n in names if n in EXCLUDE])
        shutil.copy2(ROOT / "requirements.txt", code / "requirements.txt")
        (code / "README.md").write_text(_package_readme(args.method), encoding="utf-8")
        if args.doc is not None:
            shutil.copy2(args.doc, stage / f"Documentation_template{args.doc.suffix}")
        else:
            print("warning: --doc not given; packaging the method README as Documentation_template.md", file=sys.stderr)
            shutil.copy2(ROOT / args.method / "README.md", stage / "Documentation_template.md")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = args.out_dir / f"{args.team}_submission.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(stage).as_posix())
    print(f"wrote {zip_path}")


if __name__ == "__main__":
    main()

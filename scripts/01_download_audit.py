"""Step 1 - retrieve and verify the four UCI files and write the per-hospital data audit."""

from __future__ import annotations

import argparse
import shutil

from fedheart.data import build_audit, download_raw, load_dataset
from fedheart.report import write_audit_markdown
from fedheart.splits import minority_class_feasibility
from fedheart.utils import PROJECT_ROOT, environment_info, load_config, results_dir, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--force-download", action="store_true", help="re-download even if present")
    args = p.parse_args()
    cfg = load_config(args.config)

    manifest = download_raw(cfg, force=args.force_download)
    out = results_dir(cfg) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    audit = build_audit(cfg)
    for name, table in audit.items():
        table.to_csv(out / f"{name}.csv", index=False)
    feas = minority_class_feasibility(load_dataset(cfg), cfg["split"]["fractions"])
    feas.to_csv(out / "minority_class_feasibility.csv", index=False)
    write_audit_markdown(audit, feas, manifest, PROJECT_ROOT / "docs" / "DATA_AUDIT.md")
    write_json(environment_info(), results_dir(cfg) / "environment.json")
    shutil.copy(cfg["_config_path"], results_dir(cfg) / "config_used.yaml")

    cols = ["hospital", "n_records", "n_positive", "n_negative", "n_extra_duplicate_copies",
            "fully_missing_predictors"]  # fmt: skip
    print(audit["hospital_summary"][cols].to_string(index=False))
    print(feas[["hospital", "minority_class", "n_minority", "feasible"]].to_string(index=False))
    if not feas["feasible"].all():
        raise SystemExit("a hospital cannot support the stratified split; see the audit")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from typing import Any, Dict
import argparse

from kiroframe_importer.modules import modules


def _strtobool(v):
    return str(v).strip().lower() in {"1", "true", "yes", "on", "y"}


def _load_env_file(path: str | None):
    if not path:
        return
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))


def build_cfg(args) -> Dict[str, Any]:
    e = os.environ
    cfg: Dict[str, Any] = {
        "import_type": args.import_type or e.get("IMPORT_TYPE"),
        "tracking_url": args.tracking_url or e.get("TRACKING_URI"),
        "profiling_token": args.profiling_token or e.get("PROFILING_TOKEN"),
        "base_dir": args.base_dir or e.get("EXPORT_BASE_DIR"),
        "include_schema_profile": (
            args.include_schema_profile
            if args.include_schema_profile is not None
            else (
                _strtobool(e.get("INCLUDE_SCHEMA_PROFILE"))
                if e.get("INCLUDE_SCHEMA_PROFILE")
                else True
            )
        ),
        "export_models": (
            args.export_models
            if args.export_models is not None
            else (
                _strtobool(e.get("EXPORT_MODELS")) if e.get("EXPORT_MODELS") else True
            )
        ),
        "download_artifacts": (
            args.download_artifacts
            if args.download_artifacts is not None
            else (
                _strtobool(e.get("DOWNLOAD_ARTIFACTS"))
                if e.get("DOWNLOAD_ARTIFACTS")
                else False
            )
        ),
        "experiment_page_size": args.experiment_page_size or int(e.get("EXP_PAGE_SIZE", "200")),
        "run_page_size": args.run_page_size or int(e.get("RUN_PAGE_SIZE", "1000")),
        "kiroframe_url": args.kiroframe_url or e.get("KIROFRAME_URL"),
        "do_import": args.do_import,
    }
    return cfg


def parse_args(argv=None):
    p = argparse.ArgumentParser(" export wrapper.")
    p.add_argument("--import-type", help="type of ml application", default="mlflow")
    p.add_argument("--tracking-url")
    p.add_argument("--kiroframe-url")
    p.add_argument("--profiling-token")
    p.add_argument("--base-dir")
    p.add_argument(
        "--include-schema-profile", dest="include_schema_profile", action="store_true"
    )
    p.add_argument(
        "--no-include-schema-profile",
        dest="include_schema_profile",
        action="store_false",
    )
    p.set_defaults(include_schema_profile=None)
    p.add_argument("--export-models", dest="export_models", action="store_true")
    p.add_argument("--no-export-models", dest="export_models", action="store_false")
    p.set_defaults(export_models=None)
    p.add_argument(
        "--download-artifacts", dest="download_artifacts", action="store_true"
    )
    p.add_argument(
        "--no-download-artifacts", dest="download_artifacts", action="store_false"
    )
    p.set_defaults(download_artifacts=None)
    p.add_argument("--experiment-page-size", type=int)
    p.add_argument("--run-page-size", type=int)
    p.add_argument(
        "--import-to-kiroframe", dest="do_import", default=True, action="store_true"
    )
    p.add_argument(
        "--no-import-to-kiroframe",
        dest="do_import",
        action="store_false",
        help="Skip import to Kiroframe",
    )
    p.set_defaults(do_import=True)
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return p.parse_args(argv)


def _require_core_params(cfg: Dict[str, Any]) -> None:
    """
    Ensure the core parameters are provided either via CLI or env.
    Exits with code 2 and prints a clear error if any are missing.
    """
    missing = []
    if not cfg.get("tracking_url"):
        missing.append("--tracking-url or env TRACKING_URI")
    if not cfg.get("kiroframe_url"):
        missing.append("--kiroframe-url or env KIROFRAME_URL")
    if not cfg.get("profiling_token"):
        missing.append("--profiling-token or env PROFILING_TOKEN")

    if missing:
        sys.stderr.write(
            "ERROR: missing required parameter(s): " + ", ".join(missing) + "\n"
        )
        sys.exit(2)


def main(argv=None):
    args = parse_args(argv)
    cfg = build_cfg(args)

    _require_core_params(cfg)

    import_type = cfg["import_type"]
    importer_class = modules.get(import_type)
    if not importer_class:
        raise RuntimeError(f"Unsupported type: {import_type}")

    importer = importer_class(
        tracking_url=cfg["tracking_url"],
        profiling_token=cfg["profiling_token"],
        experiment_page_size=cfg["experiment_page_size"],
        run_page_size=cfg["run_page_size"],
        base_dir=cfg["base_dir"],
        include_schema_profile=cfg["include_schema_profile"],
        export_models=cfg["export_models"],
        download_artifacts=cfg["download_artifacts"],
    )
    importer.kiroframe_url = cfg["kiroframe_url"]
    importer.export_data()

    if cfg["do_import"]:
        importer.import_data()


if __name__ == "__main__":
    main()

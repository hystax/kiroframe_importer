#!/usr/bin/env python3
import os
import inspect
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from tqdm import tqdm

from kiroframe_importer.arcee_cl import ArceeMiniCl as Arceelient


def _setup_logger(
    name: str = "mlf_export", level: int = logging.INFO
) -> logging.Logger:
    """
    Base logger with console handler. File handler is attached later in prepare()
    when the working directory is known.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured by caller
        return logger
    logger.setLevel(level)
    logger.propagate = False
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    return logger


def _attach_file_handler(
    logger: logging.Logger, logfile: Path, level: int = logging.INFO
) -> None:
    """Attach a file handler once per process."""
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(
            h, "_mlf_logfile", None
        ) == str(logfile):
            return
    fh = logging.FileHandler(str(logfile), encoding="utf-8")
    fh._mlf_logfile = str(logfile)  # mark to avoid duplicates
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(filename)s:%(lineno)d: %(message)s"
        )
    )
    logger.addHandler(fh)


logger = _setup_logger()


class MLF:

    runs_csv = "runs.csv"
    runs_basic_csv = "runs_basic.csv"

    _export_folder = "mlflow_export"

    runs_json = "runs.json"
    datasets_json = "experiments_runs_datasets.json"
    artifacts_manifest_json = "artifacts_manifest.json"
    experiment_summary_json = "experiment_summary.json"
    summary_json = "summary.json"
    models_json = "models.json"

    artifacts_dir = "artifacts"

    def __init__(
        self,
        tracking_url,
        profiling_token,
        experiment_page_size=200,
        run_page_size=1000,
        base_dir=None,
        include_schema_profile=True,
        export_models=True,
        download_artifacts=False,
    ):

        self.tracking_url = tracking_url
        self.profiling_token = profiling_token

        self.experiment_page_size = experiment_page_size
        self.run_page_size = run_page_size
        self.download_artifacts = download_artifacts
        self.include_schema_profile = include_schema_profile
        self.export_models = export_models

        self.base_dir = base_dir

        url = self.tracking_url
        clean_uri = url[:-1] if (url and url.endswith("/")) else url
        self._source_id = hashlib.md5(clean_uri.encode("utf-8")).hexdigest()[:10]

        mlflow.set_tracking_uri(url)
        self._client = MlflowClient()
        self._kiroframe_url = None

        now = datetime.now()
        self._export_name = now.strftime("%Y%m%d_%H%M_%f")

    # TODO: ->
    @property
    def kiroframe_url(self):
        return self._kiroframe_url

    @kiroframe_url.setter
    def kiroframe_url(self, value):
        self._kiroframe_url = value

    @property
    def client(self):
        return self._client

    @property
    def source_id(self):
        return self._source_id

    @property
    def import_source(self):
        return "mlflow"

    @property
    def home_dir(self):
        return Path.home()

    @property
    def default_export_folder(self):
        return os.path.join(self.home_dir, self._export_folder)

    @property
    def export_name(self):
        return self._export_name

    @property
    def full_work_dir(self):
        if self.base_dir is None:
            self.base_dir = self.default_export_folder
        return Path(os.path.join(self.base_dir, self.export_name))

    @property
    def full_runs_csv_path(self):
        return Path(os.path.join(self.full_work_dir, self.runs_csv))

    @property
    def full_runs_basic_csv_path(self):
        return Path(os.path.join(self.full_work_dir, self.runs_basic_csv))

    @property
    def full_runs_json_path(self):
        return Path(os.path.join(self.full_work_dir, self.runs_json))

    @property
    def full_datasets_json_path(self):
        return Path(os.path.join(self.full_work_dir, self.datasets_json))

    @property
    def full_artifacts_path(self):
        return Path(os.path.join(self.full_work_dir, self.artifacts_dir))

    @property
    def full_experiment_summary_path(self):
        return Path(os.path.join(self.full_work_dir, self.experiment_summary_json))

    @property
    def full_artifact_manifest_path(self):
        return Path(os.path.join(self.full_work_dir, self.artifacts_manifest_json))

    @property
    def full_summary_json_path(self):
        return Path(os.path.join(self.full_work_dir, self.summary_json))

    @property
    def full_models_json_path(self):
        return Path(os.path.join(self.full_work_dir, self.models_json))

    @staticmethod
    def dataset_context_from(di):
        if getattr(di, "tags", None):
            for t in di.tags:
                if t.key == "mlflow.data.context":
                    return t.value
        return None

    @staticmethod
    def ms_to_iso_utc(ms):
        if ms is None:
            return None
        try:
            return (
                datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        except Exception:
            return None

    def prepare(self):
        self.full_work_dir.mkdir(exist_ok=True, parents=True)
        self.full_artifacts_path.mkdir(exist_ok=True, parents=True)
        # logging: attach file handler in working dir
        _attach_file_handler(logger, self.full_work_dir / "export.log")
        logger.info("Prepared working directory: %s", self.full_work_dir)

    def iter_experiments_paged(self, view_type=ViewType.ACTIVE_ONLY, page_size=200):
        """
        Prefer search_experiments(view_type, max_results, page_token) if present.
        Fallback to list_experiments(view_type) (no paging), then very old store API.
        """
        if hasattr(self.client, "search_experiments"):
            token = None
            while True:
                page = self.client.search_experiments(
                    view_type=view_type, max_results=page_size, page_token=token
                )
                for e in page:
                    yield e
                token = getattr(page, "token", None)
                if not token:
                    break
            return

        if hasattr(self.client, "list_experiments"):
            for e in self.client.list_experiments(view_type=view_type):
                yield e
            return

        # Very old fallback
        from mlflow.tracking._tracking_service.utils import _get_store

        store = _get_store()
        for e in store.list_experiments(view_type):
            yield e

    def iter_runs_paged(
        self, experiment_id, filter_string="", order_by=None, page_size=1000
    ):
        """
        Page search_runs if client supports page_token; otherwise single page.
        Yields RunInfos (lightweight). Fetch full runs via client.get_run(run_id) when needed.
        """
        sig = inspect.signature(self.client.search_runs)
        supports_token = "page_token" in sig.parameters

        if supports_token:
            token = None
            while True:
                page = self.client.search_runs(
                    [experiment_id],
                    filter_string=filter_string,
                    max_results=page_size,
                    order_by=order_by,
                    page_token=token,
                )
                for r in page:
                    yield r
                token = getattr(page, "token", None)
                if not token:
                    break
        else:
            page = self.client.search_runs(
                [experiment_id],
                filter_string=filter_string,
                max_results=page_size,
                order_by=order_by,
            )
            for r in page:
                yield r

    def write_runs_csv_all_experiments(self):
        """
        Convenience CSV via mlflow.search_runs(experiment_ids).
        For very large servers consider paging per experiment instead.
        """
        exp_ids = [
            e.experiment_id
            for e in self.iter_experiments_paged(page_size=self.experiment_page_size)
        ]
        # dataframe
        df = mlflow.search_runs(exp_ids)
        df.to_csv(self.full_runs_csv_path, index=False)
        logger.info("Wrote CSV: %s", self.full_runs_csv_path.name)

    def export_details_and_artifacts(self):
        """
        Exports details and artifacts data.
        """
        summary = {
            "mlflow_version": mlflow.__version__,
            "experiments": 0,
            "runs": 0,
            "dataset_rows": 0,
            "artifact_entries": 0,
            "experiments_detail": {},
        }

        artifacts_manifest = []

        self.full_runs_basic_csv_path.write_text(
            "experiment_id,experiment_name,run_id,run_name,status,start_time_iso,end_time_iso,dataset_names\n"
        )

        exps = list(self.iter_experiments_paged(page_size=self.experiment_page_size))
        summary["experiments"] = len(exps)
        logger.info("Found %d experiments", len(exps))

        for exp in exps:
            exp_stats = summary["experiments_detail"].setdefault(
                exp.experiment_id, {"name": exp.name, "runs": 0, "dataset_rows": 0}
            )
            exp_metric_names = set()
            exp_param_names = set()
            exp_tag_keys = set()
            exp_dataset_names = set()

            status_counts = {
                "FINISHED": 0,
                "FAILED": 0,
                "SCHEDULED": 0,
                "RUNNING": 0,
                "KILLED": 0,
            }
            first_run_start_ms = None
            last_run_end_ms = None

            filter_str = ""  # e.g., 'attributes.status = "FINISHED"'
            order_by = ["attributes.start_time DESC"]

            runs_iter = self.iter_runs_paged(
                exp.experiment_id, filter_str, order_by, page_size=self.run_page_size
            )

            runs_iter = tqdm(
                runs_iter,
                desc=f"Exporting runs from exp '{exp.name}' ({exp.experiment_id})",
            )

            with (
                self.full_runs_json_path.open("a", encoding="utf-8") as runs_jf,
                self.full_datasets_json_path.open("a", encoding="utf-8") as ds_jf,
                self.full_runs_basic_csv_path.open("a", encoding="utf-8") as basic_csv,
            ):

                for r_stub in runs_iter:
                    summary["runs"] += 1
                    exp_stats["runs"] += 1

                    run = self.client.get_run(r_stub.info.run_id)

                    start_iso = self.ms_to_iso_utc(run.info.start_time)
                    end_iso = self.ms_to_iso_utc(run.info.end_time)

                    ds_inputs = getattr(run.inputs, "dataset_inputs", []) or []
                    datasets_list = []
                    dataset_names = []

                    # Collect metrics
                    exp_metric_names.update(run.data.metrics.keys())
                    exp_param_names.update(run.data.params.keys())
                    exp_tag_keys.update(run.data.tags.keys())

                    if dataset_names:
                        exp_dataset_names.update(dataset_names)

                    status_counts[run.info.status] = (
                        status_counts.get(run.info.status, 0) + 1
                    )

                    st = run.info.start_time
                    et = run.info.end_time
                    if st is not None:
                        first_run_start_ms = (
                            st
                            if first_run_start_ms is None
                            else min(first_run_start_ms, st)
                        )
                    if et is not None:
                        last_run_end_ms = (
                            et if last_run_end_ms is None else max(last_run_end_ms, et)
                        )

                    for di in ds_inputs:
                        ds = getattr(di, "dataset", None)
                        ds_name = getattr(ds, "name", None)
                        if ds_name:
                            dataset_names.append(ds_name)

                        item = {
                            "name": ds_name,
                            "digest": getattr(ds, "digest", None),
                            "context": self.dataset_context_from(di) if di else None,
                            "source_type": getattr(ds, "source_type", None),
                            "source": getattr(ds, "source", None),
                        }
                        if self.include_schema_profile:
                            item["schema"] = getattr(ds, "schema", None)  # JSON
                            item["profile"] = getattr(ds, "profile", None)  # JSON
                        datasets_list.append(item)

                    metrics_hist = {}
                    for k in run.data.metrics.keys():
                        try:
                            hist = self.client.get_metric_history(run.info.run_id, k)
                            metrics_hist[k] = [
                                {"step": m.step, "ts": m.timestamp, "value": m.value}
                                for m in hist
                            ]
                        except Exception as ex:
                            metrics_hist[k] = {"error": str(ex)}

                    run_row = {
                        "experiment_name": exp.name,
                        "experiment_id": exp.experiment_id,
                        "run_id": run.info.run_id,
                        "status": run.info.status,
                        "lifecycle_stage": run.info.lifecycle_stage,
                        "start_time": run.info.start_time,
                        "end_time": run.info.end_time,
                        "start_time_iso": start_iso,
                        "end_time_iso": end_iso,
                        "tags": run.data.tags,
                        "params": run.data.params,
                        "metrics_latest": run.data.metrics,
                        "metrics_history": metrics_hist,
                        "datasets": datasets_list,
                        "dataset_names": dataset_names or [],
                    }
                    runs_jf.write(json.dumps(run_row, ensure_ascii=False) + "\n")

                    wrote_any_ds_row = False

                    if ds_inputs:
                        for di in ds_inputs:
                            ds = getattr(di, "dataset", None)

                            ctx = self.dataset_context_from(di)
                            if ctx == "output":
                                role = "produced"
                            else:
                                role = "used"

                            ds_id = getattr(ds, "id", None) or getattr(
                                ds, "dataset_id", None
                            )
                            if not ds_id:
                                ds_id = f"{getattr(ds, 'name', 'ds')}:{getattr(ds, 'digest', '')[:8]}"

                            ds_row = {
                                "experiment_id": exp.experiment_id,
                                "experiment_name": exp.name,
                                "run_id": run.info.run_id,
                                "run_name": run.data.tags.get("mlflow.runName"),
                                "status": run.info.status,
                                "start_time": run.info.start_time,
                                "end_time": run.info.end_time,
                                "start_time_iso": start_iso,
                                "end_time_iso": end_iso,
                                "dataset_name": getattr(ds, "name", None),
                                "dataset_digest": getattr(
                                    ds, "digest", None
                                ),  # dataset version
                                "dataset_context": self.dataset_context_from(di),
                                "dataset_source_type": getattr(ds, "source_type", None),
                                "dataset_source": getattr(ds, "source", None),
                                "dataset_role": role,
                                "dataset_id": ds_id,
                            }
                            if self.include_schema_profile:
                                ds_row["dataset_schema"] = getattr(ds, "schema", None)
                                ds_row["dataset_profile"] = getattr(ds, "profile", None)

                            ds_jf.write(json.dumps(ds_row, ensure_ascii=False) + "\n")
                            summary["dataset_rows"] += 1
                            exp_stats["dataset_rows"] += 1
                            wrote_any_ds_row = True

                    if not wrote_any_ds_row:
                        # Ensure one datasets row even if the run has no dataset inputs
                        ds_row = {
                            "experiment_id": exp.experiment_id,
                            "experiment_name": exp.name,
                            "run_id": run.info.run_id,
                            "run_name": run.data.tags.get("mlflow.runName"),
                            "status": run.info.status,
                            "start_time": run.info.start_time,
                            "end_time": run.info.end_time,
                            "start_time_iso": start_iso,
                            "end_time_iso": end_iso,
                            "dataset_name": None,
                            "dataset_digest": None,
                            "dataset_context": None,
                            "dataset_source_type": None,
                            "dataset_source": None,
                        }
                        if self.include_schema_profile:
                            ds_row["dataset_schema"] = None
                            ds_row["dataset_profile"] = None
                        ds_jf.write(json.dumps(ds_row, ensure_ascii=False) + "\n")
                        summary["dataset_rows"] += 1
                        exp_stats["dataset_rows"] += 1

                    ds_joined = ";".join(dataset_names) if dataset_names else ""
                    basic_csv.write(
                        f"{exp.experiment_id},{json.dumps(exp.name, ensure_ascii=False)},{run.info.run_id},"
                        f"{json.dumps(run.data.tags.get('mlflow.runName'), ensure_ascii=False)},"
                        f"{run.info.status},{start_iso},{end_iso},{json.dumps(ds_joined, ensure_ascii=False)}\n"
                    )

                    if self.download_artifacts:
                        try:
                            dst = self.full_artifacts_path / run.info.run_id
                            mlflow.artifacts.download_artifacts(
                                run_id=run.info.run_id, dst_path=str(dst)
                            )
                            artifacts_manifest.append(
                                {"run_id": run.info.run_id, "artifacts_path": str(dst)}
                            )
                        except Exception as ex:
                            artifacts_manifest.append(
                                {"run_id": run.info.run_id, "artifacts_error": str(ex)}
                            )
                exp_description = None
                try:
                    # MLflow often keeps experiment note in this tag
                    exp_description = exp.tags.get(
                        "mlflow.note.content"
                    ) or exp.tags.get("description")
                except Exception:
                    pass

                exp_owner = None
                try:
                    exp_owner = exp.tags.get("mlflow.owner") or exp.tags.get("owner")
                except Exception:
                    pass

                exp_summary = {
                    "experiment_id": exp.experiment_id,
                    "experiment_name": exp.name,
                    "description": exp_description,  # may be None
                    "owner": exp_owner,  # may be None
                    "artifact_location": getattr(exp, "artifact_location", None),
                    "lifecycle_stage": getattr(exp, "lifecycle_stage", None),
                    "runs_total": exp_stats["runs"],
                    "status_counts": status_counts,
                    "first_run_start_time": first_run_start_ms,
                    "first_run_start_time_iso": self.ms_to_iso_utc(first_run_start_ms),
                    "last_run_end_time": last_run_end_ms,
                    "last_run_end_time_iso": self.ms_to_iso_utc(last_run_end_ms),
                    "metrics": sorted(exp_metric_names),
                    "params": sorted(exp_param_names),
                    "tag_keys": sorted(exp_tag_keys),
                    "dataset_names": sorted(exp_dataset_names),
                }

                with self.full_experiment_summary_path.open(
                    "a", encoding="utf-8"
                ) as esf:
                    esf.write(json.dumps(exp_summary, ensure_ascii=False) + "\n")

        with self.full_artifact_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(artifacts_manifest, f, indent=2, ensure_ascii=False)

        summary["artifact_entries"] = len(artifacts_manifest)
        with self.full_summary_json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("MLflow version: %s", mlflow.__version__)
        logger.info("Wrote RUNS_JSONL (with dataset_names[]): %s", self.runs_json)
        logger.info("Wrote DATASETS_JSONL: %s", self.datasets_json)
        logger.info("Wrote RUNS_BASIC_CSV: %s", self.runs_basic_csv)
        logger.info("Wrote manifest: %s", self.artifacts_manifest_json)
        logger.info("Wrote summary: %s", self.summary_json)
        logger.info("Wrote experiment summary JSONL: %s", self.experiment_summary_json)

    def get_models(self):
        """
        Return a list of registered models with their versions (latest first).
        Also builds two helper indexes:
          - model_to_runs: {model_name: [run_id, ...]}
          - run_to_models: {run_id: [(model_name, version), ...]}
        """
        reg_models = self.client.search_registered_models()

        results = []
        model_to_runs = {}
        run_to_models = {}

        for m in reg_models:
            item = {
                "name": m.name,
                "description": getattr(m, "description", None),
                "tags": dict(getattr(m, "tags", {}) or {}),
                "created": getattr(m, "creation_timestamp", None),
                "last_updated": getattr(m, "last_updated_timestamp", None),
                "latest_versions": [],
            }

            try:
                vers = self.client.search_model_versions(
                    filter_string=f"name='{m.name}'", order_by=["version_number DESC"]
                )
            except TypeError:
                vers = self.client.search_model_versions(
                    filter_string=f"name='{m.name}'"
                )

            versions_out = []
            for v in vers:
                v_rec = {
                    "version": v.version,
                    "current_stage": getattr(v, "current_stage", None),
                    "status": getattr(v, "status", None),
                    "run_id": getattr(v, "run_id", None),
                    "source": getattr(v, "source", None),
                    "description": getattr(v, "description", None),
                    "created": getattr(v, "creation_timestamp", None),
                    "last_updated": getattr(v, "last_updated_timestamp", None),
                    "tags": dict(getattr(v, "tags", {}) or {}),
                    "aliases": list(getattr(v, "aliases", []) or []),
                }
                versions_out.append(v_rec)

                rid = v_rec["run_id"]
                if rid:
                    model_to_runs.setdefault(m.name, []).append(rid)
                    run_to_models.setdefault(rid, []).append((m.name, v_rec["version"]))

            item["latest_versions"] = versions_out
            results.append(item)

        return results

    def maybe_write_models_json(self):
        """
        If model export enabled, dumps models.
        Keeps the rest of the script behavior unchanged unless the flag is set.
        """
        if self.export_models:
            models = self.get_models()
            with self.full_models_json_path.open("w", encoding="utf-8") as f:
                json.dump(models, f, indent=2, ensure_ascii=False)
            logger.info("Wrote MODELS_JSON: %s", self.models_json)

    def export_data(self):
        self.prepare()
        self.write_runs_csv_all_experiments()
        self.export_details_and_artifacts()
        self.maybe_write_models_json()

    @staticmethod
    def reader(filename, reverse=False):
        with open(filename, "r", encoding="utf-8") as f:
            if not reverse:
                for line in f:
                    # exception handling
                    yield json.loads(line)
            else:
                for line in reversed(list(f)):
                    yield json.loads(line)

    def import_data(self):
        if not self.kiroframe_url and self.profiling_token:
            raise Exception(
                "kiroframe url and profiling token are required for importing data."
            )
        logger.info("Importing data...")
        logger.info("%s:%s", self.import_source, self.source_id)

        arcee_cl = Arceelient(url=self.kiroframe_url, token=self.profiling_token)
        # 1. import experiments
        for i, line in enumerate(self.reader(self.full_experiment_summary_path)):
            logger.debug("Experiment line %d: %s", i, line)
            code, res = arcee_cl.import_data(
                self.import_source, self.source_id, 1, None, line
            )
            logger.info("importing experiments(task): %s", res)
        logger.info("Experiments import completed")
        # 2. import runs
        for i, line in enumerate(self.reader(self.full_runs_json_path, reverse=True)):
            code, res = arcee_cl.import_data(
                self.import_source, self.source_id, 2, None, line
            )
            logger.info("importing run: %s", res)
        logger.info("Runs import completed")
        # 3. import models
        with open(self.full_models_json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError:
                logger.error("JSON decode error: %s", self.models_json)
            else:
                code, res = arcee_cl.import_data(
                    self.import_source, self.source_id, 3, None, data
                )
                logger.info("imported models: %s", res)
        logger.info("model import completed")

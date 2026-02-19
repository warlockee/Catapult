"""
MLflow metadata consumer service.

Parses MLflow URLs, fetches metadata from MLflow REST API,
and normalizes it for storage in version metadata.
"""
import logging
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

import httpx

from app.core.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)


class MlflowResourceType(str, Enum):
    RUN = "run"
    EXPERIMENT = "experiment"
    REGISTERED_MODEL = "registered_model"


@dataclass
class MlflowUrlInfo:
    base_url: str
    resource_type: MlflowResourceType
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None


def parse_mlflow_url(url: str) -> MlflowUrlInfo:
    """
    Parse an MLflow URL to extract resource type and identifiers.

    Supports:
      - Run: http://host:port/#/experiments/{exp_id}/runs/{run_id}
      - Experiment: http://host:port/#/experiments/{exp_id}
      - Registered Model: http://host:port/#/models/{name}
      - Registered Model Version: http://host:port/#/models/{name}/versions/{ver}

    Raises:
        ValueError: If URL cannot be parsed as a valid MLflow URL.
    """
    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    fragment = parsed.fragment.strip("/")

    if not fragment:
        raise ValueError(f"No hash fragment found in MLflow URL: {url}")

    # experiments/{exp_id}/runs/{run_id}
    m = re.match(r"experiments/([^/]+)/runs/([^/]+)", fragment)
    if m:
        return MlflowUrlInfo(
            base_url=base_url,
            resource_type=MlflowResourceType.RUN,
            experiment_id=m.group(1),
            run_id=m.group(2),
        )

    # experiments/{exp_id}
    m = re.match(r"experiments/([^/]+)$", fragment)
    if m:
        return MlflowUrlInfo(
            base_url=base_url,
            resource_type=MlflowResourceType.EXPERIMENT,
            experiment_id=m.group(1),
        )

    # models/{name}/versions/{version}
    m = re.match(r"models/([^/]+)/versions/([^/]+)", fragment)
    if m:
        return MlflowUrlInfo(
            base_url=base_url,
            resource_type=MlflowResourceType.REGISTERED_MODEL,
            model_name=unquote(m.group(1)),
            model_version=m.group(2),
        )

    # models/{name}
    m = re.match(r"models/([^/]+)$", fragment)
    if m:
        return MlflowUrlInfo(
            base_url=base_url,
            resource_type=MlflowResourceType.REGISTERED_MODEL,
            model_name=unquote(m.group(1)),
        )

    raise ValueError(
        f"Cannot parse MLflow URL fragment: {fragment}. "
        "Expected experiments/{{id}}, experiments/{{id}}/runs/{{id}}, "
        "or models/{{name}}"
    )


class MlflowService:
    """
    Service for fetching and normalizing MLflow metadata.

    Uses httpx for async HTTP calls to MLflow REST API (no auth required).
    """

    DEFAULT_TIMEOUT = 15.0

    async def fetch_metadata(self, mlflow_url: str) -> Dict[str, Any]:
        """
        Parse the MLflow URL, fetch metadata from MLflow API,
        and return a normalized metadata dict.

        Raises:
            ValueError: If URL is invalid or MLflow resource not found.
            ServiceUnavailableError: If MLflow server is unreachable.
        """
        url_info = parse_mlflow_url(mlflow_url)

        try:
            if url_info.resource_type == MlflowResourceType.RUN:
                return await self._fetch_run_metadata(url_info)
            elif url_info.resource_type == MlflowResourceType.EXPERIMENT:
                return await self._fetch_experiment_metadata(url_info)
            else:
                return await self._fetch_model_metadata(url_info)
        except httpx.ConnectError as e:
            raise ServiceUnavailableError(
                "MLflow", f"Cannot connect to {url_info.base_url}: {e}"
            )
        except httpx.TimeoutException:
            raise ServiceUnavailableError(
                "MLflow", f"Timeout connecting to {url_info.base_url}"
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(
                    f"MLflow resource not found. The URL may be incorrect: {mlflow_url}"
                )
            raise ServiceUnavailableError(
                "MLflow", f"API error: HTTP {e.response.status_code}"
            )

    async def _fetch_run_metadata(self, url_info: MlflowUrlInfo) -> Dict[str, Any]:
        api_url = f"{url_info.base_url}/api/2.0/mlflow/runs/get?run_id={url_info.run_id}"

        async with httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT) as client:
            response = await client.get(api_url)
            response.raise_for_status()
            data = response.json()

        run = data.get("run", {})
        run_info = run.get("info", {})
        run_data = run.get("data", {})

        params = {p["key"]: p["value"] for p in run_data.get("params", [])}
        metrics = {
            m["key"]: m["value"]
            for m in run_data.get("metrics", [])
            if isinstance(m["value"], (int, float)) and math.isfinite(m["value"])
        }
        tags = {t["key"]: t["value"] for t in run_data.get("tags", [])}

        return {
            "resource_type": "run",
            "url": f"{url_info.base_url}/#/experiments/{url_info.experiment_id}/runs/{url_info.run_id}",
            "run_id": url_info.run_id,
            "experiment_id": url_info.experiment_id,
            "run_name": tags.get("mlflow.runName", run_info.get("run_name")),
            "status": run_info.get("status"),
            "start_time": run_info.get("start_time"),
            "end_time": run_info.get("end_time"),
            "artifact_uri": run_info.get("artifact_uri"),
            "params": params,
            "metrics": metrics,
            "tags": {k: v for k, v in tags.items() if not k.startswith("mlflow.")},
        }

    async def _fetch_experiment_metadata(self, url_info: MlflowUrlInfo) -> Dict[str, Any]:
        api_url = f"{url_info.base_url}/api/2.0/mlflow/experiments/get?experiment_id={url_info.experiment_id}"

        async with httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT) as client:
            response = await client.get(api_url)
            response.raise_for_status()
            data = response.json()

        exp = data.get("experiment", {})

        return {
            "resource_type": "experiment",
            "url": f"{url_info.base_url}/#/experiments/{url_info.experiment_id}",
            "experiment_id": url_info.experiment_id,
            "experiment_name": exp.get("name"),
            "artifact_location": exp.get("artifact_location"),
            "lifecycle_stage": exp.get("lifecycle_stage"),
            "tags": {t["key"]: t["value"] for t in exp.get("tags", [])},
        }

    async def _fetch_model_metadata(self, url_info: MlflowUrlInfo) -> Dict[str, Any]:
        api_url = f"{url_info.base_url}/api/2.0/mlflow/registered-models/get?name={url_info.model_name}"

        async with httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT) as client:
            response = await client.get(api_url)
            response.raise_for_status()
            data = response.json()

        model = data.get("registered_model", {})
        latest_versions = model.get("latest_versions", [])

        result = {
            "resource_type": "registered_model",
            "url": f"{url_info.base_url}/#/models/{url_info.model_name}",
            "model_name": url_info.model_name,
            "description": model.get("description"),
            "creation_timestamp": model.get("creation_timestamp"),
            "last_updated_timestamp": model.get("last_updated_timestamp"),
            "latest_versions": [
                {
                    "version": v.get("version"),
                    "current_stage": v.get("current_stage"),
                    "status": v.get("status"),
                    "source": v.get("source"),
                    "run_id": v.get("run_id"),
                }
                for v in latest_versions
            ],
            "tags": {t["key"]: t["value"] for t in model.get("tags", [])},
        }

        if url_info.model_version:
            result["requested_version"] = url_info.model_version

        return result


# Singleton instance
mlflow_service = MlflowService()

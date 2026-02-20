"""
Shared API discovery helpers for probing deployment endpoints.

Used by both local_executor.py (local deployments) and
production_deployments.py (production endpoints).
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_sample_request_body(endpoint_path: str, model_name: str = "model") -> dict:
    """Get a sample request body for probing an endpoint based on path matching."""
    path_lower = endpoint_path.lower()

    if "/chat/completions" in path_lower:
        return {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
            "stream": False,
        }
    elif "/completions" in path_lower and "/chat" not in path_lower:
        return {
            "model": model_name,
            "prompt": "Hi",
            "max_tokens": 1,
            "stream": False,
        }
    elif "/embeddings" in path_lower:
        return {
            "model": model_name,
            "input": "test",
        }
    elif "/audio/speech" in path_lower:
        return {
            "model": model_name,
            "input": "Hello",
            "voice": "alloy",
        }
    elif "/audio/transcriptions" in path_lower:
        return {}
    elif "/inference" in path_lower:
        return {
            "input": "test",
            "parameters": {},
        }
    elif "/generate" in path_lower:
        return {
            "model": model_name,
            "prompt": "Hi",
            "max_tokens": 1,
        }
    elif "/predict" in path_lower:
        return {
            "input": "test",
        }
    else:
        return {}


def requires_file_upload(endpoint_path: str, method_details: dict | None = None) -> bool:
    """
    Check if an endpoint requires file upload (multipart/form-data).

    Checks both the OpenAPI spec and known path patterns.
    """
    if method_details:
        request_body = method_details.get("requestBody", {})
        content = request_body.get("content", {})
        if "multipart/form-data" in content:
            return True
        if "application/octet-stream" in content:
            return True
        for content_type, content_spec in content.items():
            schema = content_spec.get("schema", {})
            if schema.get("format") in ["binary", "byte"]:
                return True
            properties = schema.get("properties", {})
            for prop_name, prop_spec in properties.items():
                if prop_spec.get("type") == "string" and prop_spec.get("format") in ["binary", "byte"]:
                    return True
                if prop_name.lower() in ["file", "files", "audio", "image", "video"]:
                    return True

    path_lower = endpoint_path.lower()
    file_upload_patterns = [
        "/audio/transcriptions",
        "/files",
        "/uploads",
    ]
    return any(pattern in path_lower for pattern in file_upload_patterns)


def sort_endpoints_by_priority(endpoints: list) -> list:
    """Sort endpoints by priority (most important first)."""
    def priority(ep):
        path = ep["path"].lower()
        if path == "/v1/chat/completions":
            return (0, path)
        if path == "/v1/audio/speech":
            return (1, path)
        if path == "/v1/audio/transcriptions":
            return (2, path)
        if path == "/v1/completions":
            return (3, path)
        if path == "/v1/models":
            return (4, path)
        if path == "/v1/embeddings":
            return (5, path)
        if path.startswith("/v1/"):
            return (10, path)
        if path in ["/health", "/healthz", "/ready"]:
            return (20, path)
        if any(k in path for k in ["/generate", "/inference", "/predict"]):
            return (30, path)
        if path in ["/version", "/metrics", "/info"]:
            return (40, path)
        return (100, path)

    return sorted(endpoints, key=priority)


def sort_paths_by_priority(paths: list) -> list:
    """Sort paths by priority."""
    def priority(path):
        p = path.lower()
        if p == "/v1/chat/completions":
            return 0
        if p == "/v1/audio/speech":
            return 1
        if p == "/v1/audio/transcriptions":
            return 2
        if p == "/v1/completions":
            return 3
        if p.startswith("/v1/"):
            return 10
        return 100

    return sorted(paths, key=priority)


def detect_api_type_and_recommend(detected_paths: list, endpoints: list | None = None) -> tuple:
    """
    Detect API type and recommend best benchmark endpoint.

    Only recommends endpoints that can be benchmarked with JSON (not file uploads).
    """
    paths_set = set(p.lower() for p in detected_paths)

    file_upload_paths = set()
    if endpoints:
        for ep in endpoints:
            if ep.get("requires_file_upload"):
                file_upload_paths.add(ep.get("path", "").lower())

    def is_benchmarkable(path: str) -> bool:
        return path.lower() not in file_upload_paths and not requires_file_upload(path)

    # Detect API type based on available paths
    audio_patterns = {"/v1/audio/speech", "/v1/audio/transcriptions", "/transcribe", "/synthesize", "/tts", "/stt"}
    openai_patterns = {"/v1/chat/completions", "/v1/completions", "/v1/models"}
    generic_patterns = {"/v1/inference", "/predict", "/generate"}

    api_type = "unknown"
    if paths_set & audio_patterns:
        api_type = "audio"
    elif paths_set & openai_patterns:
        api_type = "openai"
    elif paths_set & generic_patterns:
        api_type = "generic"

    recommendation_priority = [
        "/v1/chat/completions",
        "/v1/audio/speech",
        "/transcribe",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/inference",
        "/predict",
        "/generate",
    ]

    # First try: benchmarkable (non-file-upload) endpoint
    for path in recommendation_priority:
        if path.lower() in paths_set and is_benchmarkable(path):
            return api_type, path

    if endpoints:
        for ep in endpoints:
            if ep.get("method") == "POST" and not ep.get("requires_file_upload"):
                return api_type, ep.get("path")

    # Second try: any POST endpoint (including file uploads) — better than health
    for path in recommendation_priority:
        if path.lower() in paths_set:
            return api_type, path

    if endpoints:
        for ep in endpoints:
            if ep.get("method") == "POST":
                return api_type, ep.get("path")

    if detected_paths:
        return api_type, detected_paths[0]

    return api_type, None


# --- OpenAPI schema helpers ---


def resolve_schema_ref(ref: str, openapi_spec: dict) -> dict:
    """
    Resolve a $ref string like '#/components/schemas/Foo' to the actual schema dict.

    Handles one level of $ref chaining.
    """
    if not ref or not ref.startswith("#/"):
        return {}
    parts = ref.lstrip("#/").split("/")
    current = openapi_spec
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return {}
    # Resolve one more level if still a $ref
    if isinstance(current, dict) and "$ref" in current:
        return resolve_schema_ref(current["$ref"], openapi_spec)
    return current if isinstance(current, dict) else {}


def extract_request_schema(method_details: dict, openapi_spec: dict) -> dict | None:
    """
    Extract the resolved JSON schema for the request body from OpenAPI endpoint details.

    Handles both application/json and multipart/form-data content types.
    Returns None if no request body schema is found.
    """
    request_body = method_details.get("requestBody", {})
    content = request_body.get("content", {})

    # Prefer application/json, fall back to multipart/form-data
    json_content = content.get("application/json") or content.get("multipart/form-data")
    if not json_content:
        return None

    schema = json_content.get("schema", {})
    if not schema:
        return None

    # Resolve $ref if present
    if "$ref" in schema:
        schema = resolve_schema_ref(schema["$ref"], openapi_spec)

    if not schema:
        return None

    # Resolve $ref in properties
    resolved_props = {}
    for prop_name, prop_spec in schema.get("properties", {}).items():
        if isinstance(prop_spec, dict) and "$ref" in prop_spec:
            resolved_props[prop_name] = resolve_schema_ref(prop_spec["$ref"], openapi_spec)
        else:
            resolved_props[prop_name] = prop_spec

    if resolved_props:
        schema = {**schema, "properties": resolved_props}

    return schema


def generate_sample_from_schema(schema: dict, openapi_spec: dict | None = None) -> dict:
    """
    Generate a minimal valid request body from a JSON schema.

    Uses 'example' values when present, otherwise generates defaults by type.
    Only includes required fields + fields with examples.
    """
    if not schema or schema.get("type") != "object":
        return {}

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    result = {}
    for prop_name, prop_spec in properties.items():
        if not isinstance(prop_spec, dict):
            continue
        # Include required fields and fields with examples
        if prop_name not in required and "example" not in prop_spec and "default" not in prop_spec:
            continue

        result[prop_name] = _generate_value_for_property(prop_name, prop_spec, openapi_spec)

    return result


def _generate_value_for_property(prop_name: str, prop_spec: dict, openapi_spec: dict | None = None) -> Any:
    """Generate a sample value for a single property based on its schema."""
    # Resolve $ref
    if "$ref" in prop_spec and openapi_spec:
        prop_spec = resolve_schema_ref(prop_spec["$ref"], openapi_spec)

    # Use example/default if available
    if "example" in prop_spec:
        return prop_spec["example"]
    if "default" in prop_spec:
        return prop_spec["default"]

    prop_type = prop_spec.get("type", "string")

    # Handle anyOf / oneOf — pick the first non-null type
    if "anyOf" in prop_spec or "oneOf" in prop_spec:
        options = prop_spec.get("anyOf") or prop_spec.get("oneOf", [])
        for option in options:
            if isinstance(option, dict) and option.get("type") != "null":
                return _generate_value_for_property(prop_name, option, openapi_spec)

    if prop_type == "string":
        if prop_spec.get("enum"):
            return prop_spec["enum"][0]
        return "string"
    elif prop_type == "integer":
        return 1
    elif prop_type == "number":
        return 1.0
    elif prop_type == "boolean":
        return False
    elif prop_type == "array":
        items = prop_spec.get("items", {})
        if "$ref" in items and openapi_spec:
            items = resolve_schema_ref(items["$ref"], openapi_spec)
        sample_item = _generate_value_for_property("item", items, openapi_spec) if items else "string"
        return [sample_item]
    elif prop_type == "object":
        if "properties" in prop_spec:
            return generate_sample_from_schema(prop_spec, openapi_spec)
        return {}
    else:
        return "string"


def build_sample_body(
    endpoint_path: str,
    model_name: str,
    method_details: dict | None,
    openapi_spec: dict | None,
) -> dict:
    """
    Build the best possible sample request body for an endpoint.

    Tries OpenAPI schema first, falls back to path-based matching.
    """
    # Try schema-based generation first
    if method_details and openapi_spec:
        schema = extract_request_schema(method_details, openapi_spec)
        if schema:
            sample = generate_sample_from_schema(schema, openapi_spec)
            if sample:
                # Substitute model name if the schema has a "model" field
                if "model" in sample and isinstance(sample["model"], str):
                    sample["model"] = model_name
                return sample

    # Fall back to path-based matching
    return get_sample_request_body(endpoint_path, model_name)

#!/usr/bin/env python3
"""
Test Docker builds for all models.
Triggers a Docker build for each model, waits for completion, and runs GC.
"""
import json
import subprocess
import time
import sys
from urllib.parse import quote

API_BASE = "http://localhost:8080/api/v1"
API_KEY = "admin.admin"

def curl_get(endpoint):
    """Make GET request to API."""
    result = subprocess.run(
        ['curl', '-s', f'{API_BASE}{endpoint}', '-H', f'X-API-Key: {API_KEY}'],
        capture_output=True, text=True
    )
    try:
        return json.loads(result.stdout) if result.stdout else None
    except json.JSONDecodeError:
        print(f"  JSON decode error: {result.stdout[:100]}")
        return None

def curl_post(endpoint, data):
    """Make POST request to API."""
    result = subprocess.run(
        ['curl', '-s', '-X', 'POST', f'{API_BASE}{endpoint}',
         '-H', f'X-API-Key: {API_KEY}',
         '-H', 'Content-Type: application/json',
         '-d', json.dumps(data)],
        capture_output=True, text=True
    )
    try:
        return json.loads(result.stdout) if result.stdout else None
    except json.JSONDecodeError:
        print(f"  JSON decode error: {result.stdout[:200]}")
        return None

def run_gc():
    """Run Docker garbage collection."""
    print("  Running Docker GC...")
    # Prune build cache
    subprocess.run(['docker', 'builder', 'prune', '-f', '--filter', 'until=1h'],
                   capture_output=True)
    # Prune dangling images
    subprocess.run(['docker', 'image', 'prune', '-f'], capture_output=True)
    print("  GC complete")

def wait_for_build(build_id, timeout=600):
    """Wait for build to complete. Returns (success, status, error)."""
    start = time.time()
    while time.time() - start < timeout:
        build = curl_get(f'/docker/builds/{build_id}')
        if not build:
            return False, 'error', 'Failed to fetch build status'

        status = build.get('status')
        if status == 'success':
            return True, status, None
        elif status == 'failed':
            return False, status, build.get('error_message', 'Unknown error')

        time.sleep(5)

    return False, 'timeout', f'Build timed out after {timeout}s'

def main():
    print("Fetching models from registry...", flush=True)
    models_data = curl_get('/models?page=1&size=100')
    if not models_data:
        print("Failed to fetch models")
        return 1

    models = models_data['items']
    print(f"Found {len(models)} models\n", flush=True)

    results = {'success': [], 'failed': [], 'skipped': []}

    for i, model in enumerate(models, 1):
        model_id = model['id']
        model_name = model['name']
        server_type = model['server_type']

        print(f"[{i}/{len(models)}] {model_name} ({server_type})", flush=True)

        # Get first release for this model (URL encode model name)
        encoded_name = quote(model_name, safe='')
        versions_data = curl_get(f'/versions?model_name={encoded_name}&page=1&size=1')
        if not versions_data or not versions_data.get('items'):
            print(f"  SKIP: No versions found", flush=True)
            results['skipped'].append(model_name)
            continue

        release = versions_data['items'][0]
        release_id = release['id']
        version = release['version']

        # Create image tag (lowercase for Docker, sanitize characters)
        safe_name = model_name.lower().replace(' ', '-')
        safe_version = version.replace(' ', '-')
        image_tag = f"model-registry/{safe_name}:{safe_version}"

        print(f"  Version: {version}", flush=True)
        print(f"  Image tag: {image_tag}", flush=True)

        # Trigger Docker build
        build_data = {
            'release_id': release_id,
            'image_tag': image_tag,
            'build_type': 'organic',
        }

        build_response = curl_post('/docker/builds', build_data)
        if not build_response or 'id' not in build_response:
            error = build_response.get('detail', 'Unknown error') if build_response else 'API error'
            print(f"  FAILED to start build: {error}", flush=True)
            results['failed'].append((model_name, f"Start failed: {error}"))
            continue

        build_id = build_response['id']
        print(f"  Build started: {build_id}", flush=True)

        # Wait for build
        success, status, error = wait_for_build(build_id)

        if success:
            print(f"  SUCCESS", flush=True)
            results['success'].append(model_name)
            run_gc()
        else:
            print(f"  FAILED: {status} - {error}", flush=True)
            results['failed'].append((model_name, f"{status}: {error}"))

        print("", flush=True)

    # Summary
    print("=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Success: {len(results['success'])}", flush=True)
    for name in results['success']:
        print(f"  - {name}", flush=True)

    print(f"\nFailed: {len(results['failed'])}", flush=True)
    for name, error in results['failed']:
        print(f"  - {name}: {error}", flush=True)

    print(f"\nSkipped: {len(results['skipped'])}", flush=True)
    for name in results['skipped']:
        print(f"  - {name}", flush=True)

    return 0 if not results['failed'] else 1

if __name__ == '__main__':
    sys.exit(main())

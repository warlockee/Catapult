#!/usr/bin/env python3
import os
import sys
import time
import requests
import subprocess
import json

# Configuration
API_BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8080/api/v1")
MODEL_NAME = "test-fallback-model"
REL_PATH = "test-fallback-model-nonexistent"

def get_api_key():
    print("Getting API key...")
    cmd = ["docker", "compose", "exec", "-T", "backend", "python", "scripts/create_api_key.py", "--name", "admin", "--reset"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if "Key: " in line:
                return line.split("Key: ")[1].strip()
    except Exception as e:
        print(f"Error getting API key: {e}")
        sys.exit(1)
    return "admin"

def wait_for_api(url):
    print(f"Checking API connectivity at {url}...")
    base_check_url = url.replace("/api/v1", "/api")
    for _ in range(30):
        try:
            res = requests.get(f"{base_check_url}/health")
            if res.status_code == 200:
                print("API is up.")
                return
        except:
            pass
        time.sleep(1)
    print("API failed to come up.")
    sys.exit(1)

def main():
    wait_for_api(API_BASE_URL)
    api_key = get_api_key()
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    # 1. Create Model
    print(f"Creating model {MODEL_NAME}...")
    res = requests.post(f"{API_BASE_URL}/models", json={
        "name": MODEL_NAME,
        "storage_path": REL_PATH,
        "description": "Test Fallback Logic"
    }, headers=headers)
    
    if res.status_code == 201:
        model_id = res.json()["id"]
    elif res.status_code == 409:
        res = requests.get(f"{API_BASE_URL}/models?search={MODEL_NAME}", headers=headers)
        model_id = res.json()[0]["id"]
    else:
        print(f"Failed to create model: {res.text}")
        sys.exit(1)
    
    # 2. Create Release
    version = f"1.0.{int(time.time())}"
    print(f"Creating release {version}...")
    res = requests.post(f"{API_BASE_URL}/releases", json={
        "model_id": model_id,
        "version": version,
        "tag": f"v{version}",
        "digest": "sha256:dummy",
        "ceph_path": REL_PATH
    }, headers=headers)
    
    if res.status_code != 201:
        print(f"Failed to create release: {res.text}")
        sys.exit(1)
    release_id = res.json()["id"]
    
    # 3. Trigger DEFAULT Docker Build
    print("Triggering DEFAULT Docker build...")
    res = requests.post(f"{API_BASE_URL}/docker/builds", json={
        "release_id": release_id,
        "image_tag": f"test-fallback:{version}",
        "build_type": "default" 
    }, headers=headers)
    
    if res.status_code != 201:
        print(f"Failed to trigger build: {res.text}")
        sys.exit(1)
    build_id = res.json()["id"]
    print(f"✓ Build ID: {build_id}")
    
    # 4. Poll Status
    print("Waiting for build to process...")
    # We don't need success, just log confirmation.
    # It might fail quickly at 'docker build' step, but COPY happens before.
    status = "pending"
    for i in range(30):
        res = requests.get(f"{API_BASE_URL}/docker/builds/{build_id}", headers=headers)
        data = res.json()
        status = data["status"]
        if status in ["success", "failed"]:
            break
        time.sleep(2)
        
    print(f"Final Status: {status}")
    
    # 5. Check Logs for Build Progress
    print("Checking logs for build progress...")
    res = requests.get(f"{API_BASE_URL}/docker/builds/{build_id}/logs", headers=headers)
    if res.status_code == 200:
        logs_json = res.json()
        logs = logs_json.get("logs", "")
        # print("--- LOG START ---")
        # print(logs[:2000])
        # print("--- LOG END ---")
        
        # Check for context transfer size (should be > 1MB if dependencies were copied)
        # "transferring context: 2.59MB"
        context_size_large = False
        import re
        match = re.search(r"transferring context: ([0-9\.]+)MB", logs)
        if match:
            size = float(match.group(1))
            if size > 1.0:
                context_size_large = True
                print(f"Context size: {size}MB (Expected > 1MB)")
        
        # Check if COPY instructions succeeded (or at least started)
        # If fallback matched, context is large.
        # If build fails LATER (e.g. pip install), that's fine.
        # We just want to avoid "failed to calculate checksum ... : not found"
        
        failure_missing_file = "failed to calculate checksum" in logs and "not found" in logs
        
        if context_size_large and not failure_missing_file:
             print("✅ SUCCESS: Build context is large and no missing file error from COPY.")
        else:
             print("❌ FAILURE: Build context unclear or missing file error found.")
             print(f"Large Context: {context_size_large}")
             print(f"Missing File Error: {failure_missing_file}")
             # print(logs)
             sys.exit(1)
             
    else:
        print("Failed to get logs")
        sys.exit(1)

if __name__ == "__main__":
    main()

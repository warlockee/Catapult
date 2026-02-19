#!/usr/bin/env python3
import os
import sys
import shutil
import time
import requests
import subprocess
from pathlib import Path

# Configuration
API_BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8080/api/v1")
STORAGE_DIR = "./storage/models"
MODEL_NAME = os.environ.get("TEST_MODEL_NAME", "test-optimized-model")
REL_PATH = MODEL_NAME

HEADERS = {"Content-Type": "application/json"}

def get_api_key():
    """Create or get API key using the script."""
    print("Getting API key...")
    # Try to reuse the admin key created earlier
    cmd = ["docker", "compose", "exec", "-T", "backend", "python", "scripts/create_api_key.py", "--name", "admin", "--reset"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if "Key: " in line:
                return line.split("Key: ")[1].strip()
    except subprocess.CalledProcessError:
        pass
        
    # Fallback to creating a new one
    unique_name = f"e2e-optimized-{int(time.time())}"
    cmd = ["docker", "compose", "exec", "-T", "backend", "python", "scripts/create_api_key.py", "--name", unique_name]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if "Key: " in line:
                return line.split("Key: ")[1].strip()
    except Exception as e:
        print(f"Error getting API key: {e}")
        sys.exit(1)
    return "admin" # Default fallthrough

def wait_for_api(url): # url is expected to be base like http://localhost
    print(f"Checking API connectivity at {url}...")
    # Clean base url to remove /api/v1 for the health check
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
    
    # 1. Create/Get Model
    print(f"Creating/Getting model {MODEL_NAME}...")
    res = requests.post(f"{API_BASE_URL}/models", json={
        "name": MODEL_NAME,
        "storage_path": REL_PATH,
        "description": "Optimized Model for E2E Test"
    }, headers=headers)
    
    if res.status_code == 201:
        model_id = res.json()["id"]
    elif res.status_code == 409:
        print("Model already exists, finding it...")
        res = requests.get(f"{API_BASE_URL}/models", headers=headers)
        models = res.json()
        model_id = next((m["id"] for m in models if m["name"] == MODEL_NAME), None)
        if not model_id:
            res = requests.get(f"{API_BASE_URL}/models?search={MODEL_NAME}", headers=headers)
            if res.status_code == 200 and len(res.json()) > 0:
                 model_id = res.json()[0]["id"]
            else:
                 print("Could not find model ID")
                 sys.exit(1)
    else:
        print(f"Failed to create model: {res.text}")
        sys.exit(1)
    print(f"✓ Model ID: {model_id}")

    # 2. Create Release
    version = f"1.0.{int(time.time())}"
    print(f"Creating release {version}...")
    res = requests.post(f"{API_BASE_URL}/releases", json={
        "model_id": model_id,
        "version": version,
        "tag": f"v{version}",
        "digest": "sha256:dummy",
        "ceph_path": REL_PATH,
        "metadata": {"e2e_test": True}
    }, headers=headers)
    
    if res.status_code != 201:
        print(f"Failed to create release: {res.text}")
        sys.exit(1)
    
    release_id = res.json()["id"]
    print(f"✓ Release ID: {release_id}")
    
    # 3. Trigger OPTIMIZED Docker Build
    print("Triggering OPTIMIZED Docker build...")
    res = requests.post(f"{API_BASE_URL}/docker/builds", json={
        "release_id": release_id,
        "image_tag": f"catapult/optimized-e2e:{version}",
        "build_type": "optimized"
    }, headers=headers)
    
    if res.status_code != 201:
        print(f"Failed to trigger build: {res.text}")
        sys.exit(1)
        
    build_id = res.json()["id"]
    print(f"✓ Build ID: {build_id}")
    
    # 4. Poll Status
    print("Waiting for build to complete (this may take 10+ minutes)...")
    status = "pending"
    logs_printed = False
    
    # Poll for up to 20 minutes (1200 seconds) as optimized builds are large
    for i in range(120): 
        res = requests.get(f"{API_BASE_URL}/docker/builds/{build_id}", headers=headers)
        if res.status_code != 200:
             print(f"Error checking status: {res.status_code}")
             time.sleep(5)
             continue
             
        data = res.json()
        new_status = data["status"]
        
        if new_status != status:
            print(f"Status: {new_status}")
            status = new_status
            
        if status in ["success", "failed"]:
            break
            
        if i % 6 == 0: # Print a dot every minute
            sys.stdout.write(".")
            sys.stdout.flush()
            
        time.sleep(10)
    print() # Newline after dots
        
    # 5. Check Result
    if status != "success":
        print(f"Build failed! Status: {status}")
        print("Fetching logs...")
        res = requests.get(f"{API_BASE_URL}/docker/builds/{build_id}/logs", headers=headers)
        if res.status_code == 200:
            log_data = res.json()
            print("--- LOGS ---")
            print(log_data.get("logs", "No logs returned"))
            print("--- END LOGS ---")
            
            # Use raw logs file as backup if API returns truncated/processed logs
            log_path = log_data.get("log_path")
            if log_path and os.path.exists(log_path):
                 print(f"Full logs available at: {log_path}")
        else:
            print("Failed to fetch logs")
        sys.exit(1)
    
    print("✅ Build SUCCESS!")
    print(f"Image: catapult/optimized-e2e:{version}")

if __name__ == "__main__":
    main()

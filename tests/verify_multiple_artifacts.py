import requests
import time
import os
import sys
import uuid

# Configuration
API_BASE_URL = "http://localhost/api/v1"
HEADERS = {"x-api-key": "test-key-123"}
SCHEMA = "model_registry"

def create_dummy_artifact(release_id, model_id, name):
    # Register artifact in DB (using API)
    
    # Create a dummy file
    filename = f"./storage/artifacts/dummy_{name}.txt"
    os.makedirs("./storage/artifacts", exist_ok=True)
    with open(filename, "w") as f:
        f.write(f"Content of {name}")
        
    payload = {
        "release_id": str(release_id),
        "model_id": str(model_id),
        "name": name,
        "artifact_type": "file",
        "file_path": filename,
        "storage_path": filename,
        "size_bytes": 1024,
        "checksum": "sha256:dummy_checksum"
    }
    
    res = requests.post(f"{API_BASE_URL}/artifacts/", json=payload, headers=HEADERS)
    if res.status_code == 201:
        return res.json()["id"]
    else:
        print(f"Failed to create artifact {name}: {res.text}")
        return None

def main():
    # 1. Create Model & Release (Standard flow)
    model_name = f"multi-art-test-{uuid.uuid4().hex[:6]}"
    res = requests.post(f"{API_BASE_URL}/models/", json={
        "name": model_name,
        "owner": "test-user",
        "description": "Test model for multiple artifacts",
        "storage_path": "/dummy/model/path"
    }, headers=HEADERS)
    if res.status_code != 201:
        print(f"Failed to create model: {res.text}")
        sys.exit(1)
    model_id = res.json()["id"]
    
    res = requests.post(f"{API_BASE_URL}/releases/", json={
        "model_id": model_id,
        "version": "1.0.0",
        "semver_major": 1,
        "semver_minor": 0,
        "semver_patch": 0,
        "ceph_path": "./storage/vllm/models/test_model",
        "image_name": "test-image",
        "tag": "v1.0.0",
        "digest": "sha256:dummy"
    }, headers=HEADERS)
    if res.status_code != 201:
        print(f"Failed to create release: {res.text}")
        sys.exit(1)
    release_id = res.json()["id"]

    # 2. Register Artifacts
    id1 = create_dummy_artifact(release_id, model_id, "artifact_one")
    id2 = create_dummy_artifact(release_id, model_id, "artifact_two")
    
    if not id1 or not id2:
        print("Failed to setup artifacts")
        sys.exit(1)
        
    print(f"Created artifacts: {id1}, {id2}")

    # 3. Trigger Docker Build with BOTH artifacts
    print("Triggering Docker Build...")
    build_payload = {
        "release_id": release_id,
        "image_tag": f"test-multi:{uuid.uuid4().hex[:6]}",
        "build_type": "organic",
        "artifact_ids": [id1, id2]
    }
    
    res = requests.post(f"{API_BASE_URL}/docker/builds", json=build_payload, headers=HEADERS)
    if res.status_code != 201:
        print(f"Failed to trigger build: {res.text}")
        sys.exit(1)
        
    build_id = res.json()["id"]
    print(f"Build ID: {build_id}")
    
    # 4. Monitor Log for "Copying artifact..." messages
    print("Monitoring logs...")
    for _ in range(60):
        time.sleep(2)
        res = requests.get(f"{API_BASE_URL}/docker/builds/{build_id}/logs", headers=HEADERS)
        if res.status_code == 200:
            logs = res.json().get("logs", "")
            
            # Check status
            res_status = requests.get(f"{API_BASE_URL}/docker/builds/{build_id}", headers=HEADERS)
            status = res_status.json()["status"]
            if status in ["success", "failed"]:
                print(f"Build finished with status: {status}")
                if status == "failed":
                    print(f"Error: {res_status.json().get('error_message')}")
                break
        else:
            print(f"Failed to get logs: {res.status_code}")

if __name__ == "__main__":
    main()

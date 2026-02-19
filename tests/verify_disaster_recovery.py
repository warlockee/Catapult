
#!/usr/bin/env python3
import subprocess
import time
import sys
import os
import requests
import json

API_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8080/api/v1")
SCHEMA = "model_registry"

def run_cmd(cmd, cwd=None, check=True):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result

def get_row_counts():
    cmd = [
        "docker", "compose", "exec", "-T", "postgres", 
        "psql", "-U", "registry", "-d", "registry", "-t", "-c", 
        f"SELECT (SELECT count(*) FROM {SCHEMA}.models) as m, (SELECT count(*) FROM {SCHEMA}.releases) as r;"
    ]
    res = run_cmd(cmd, check=False)
    if res.returncode != 0:
        if "does not exist" in res.stderr:
            return 0, 0
        raise Exception(f"Query failed: {res.stderr}")
    parts = res.stdout.strip().split("|")
    return int(parts[0].strip()), int(parts[1].strip())

def main():
    print("=== Starting Disaster Recovery Verification ===")
    
    # 1. Check Initial State
    print("\n1. Checking Initial State...")
    initial_models, initial_releases = get_row_counts()
    print(f"Initial: Models={initial_models}, Releases={initial_releases}")
    if initial_models == 0:
        print("Warning: DB is empty. Test might not be meaningful.")

    # 2. Trigger Backup
    print("\n2. Triggering Backup...")
    run_cmd(["docker", "compose", "exec", "-T", "backend", "python", "scripts/trigger_backup.py"])
    
    # Verify backup content
    backups_dir = "./storage/snapshots"
    latest_backup = sorted([f for f in os.listdir(backups_dir) if f.startswith("db_backup_")], reverse=True)[0]
    print(f"Latest backup: {latest_backup}")
    
    with open(os.path.join(backups_dir, latest_backup), "r") as f:
        content = f.read()
        if "DROP TABLE" in content or "DROP SCHEMA" in content or "pg_dump" in content: 
             # pg_dump with -c adds DROP commands. 
             # Note: -c adds DROP TABLE/VIEW/etc.
             if "DROP TABLE" in content:
                 print("✅ Backup contains DROP TABLE statements (Clean backup verified)")
             else:
                 print("⚠️ Backup might NOT contain DROP TABLE statements. Check content.")
        
    
    # 3. Simulate Disaster (Wipe DB)
    print("\n3. Simulating Disaster (Wipe DB)...")
    print("Stopping backend to release locks...")
    run_cmd(["docker", "compose", "stop", "backend"])
    
    run_cmd([
        "docker", "compose", "exec", "-T", "postgres", 
        "psql", "-U", "registry", "-d", "registry", "-c", 
        f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE; CREATE SCHEMA {SCHEMA};"
    ])
    
    # m, r = get_row_counts()
    # Expect error or 0. Since we dropped schema, accessing tables might fail or return 0 if recreated empty?
    # Actually get_row_counts query will fail if tables don't exist.
    # We expect get_row_counts to fail.
    print("DB Wiped. Verifying empty state...")
    try:
        get_row_counts()
        print("Warning: querying wiped DB succeeded (unexpected)")
    except Exception:
        print("✅ DB is confirmed broken/empty.")

    # 4. Trigger Restore (Restart Backend)
    print("\n4. Triggering Restore (Restart Backend)...")
    run_cmd(["docker", "compose", "restart", "backend"])
    
    # Wait for healthy
    print("Waiting for backend to come up...")
    for i in range(30):
        try:
             res = requests.get(API_URL.replace("/v1", "") + "/health")
             if res.status_code == 200:
                 print("Backend is UP.")
                 break
        except:
            pass
        time.sleep(2)
        print(".", end="", flush=True)
    print()

    # 5. Verify Restore
    print("\n5. Verifying Restore...")
    restored_models, restored_releases = get_row_counts()
    print(f"Restored: Models={restored_models}, Releases={restored_releases}")
    
    if restored_models == initial_models and restored_releases == initial_releases:
        print("✅ Data Restored Successfully!")
    else:
        print(f"❌ Data Mismatch! Expected m={initial_models}/r={initial_releases}, Got m={restored_models}/r={restored_releases}")
        sys.exit(1)

    # 6. Verify Idempotency (Restart again)
    print("\n6. Verifying Idempotency (Prevent Duplicates)...")
    run_cmd(["docker", "compose", "restart", "backend"])
    
    print("Waiting for backend...")
    time.sleep(10) # Give it time to attempt restore
    for i in range(30):
        try:
             res = requests.get(API_URL.replace("/v1", "") + "/health")
             if res.status_code == 200:
                 break
        except:
             pass
        time.sleep(1)
        
    final_models, final_releases = get_row_counts()
    print(f"Final: Models={final_models}, Releases={final_releases}")
    
    if final_models == initial_models:
        print("✅ Counts matched (No Duplication).")
    else:
        print(f"❌ Duplicate Records Detected! Expected {initial_models}, Got {final_models}")
        sys.exit(1)

    print("\n=== TEST PASSED ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test Failed with Exception: {e}")
        print("--- Backend Logs ---")
        run_cmd(["docker", "compose", "logs", "backend", "--tail", "100"], check=False)
        sys.exit(1)

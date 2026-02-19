import os
import time
from bso import Registry, RegistryError

# Configuration
# Use the demo API key and local URL
API_KEY = os.getenv("CATAPULT_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("CATAPULT_API_URL", "http://localhost/api")

def main():
    print(f"Connecting to registry at {BASE_URL}...")
    
    # Initialize client
    registry = Registry(
        base_url=BASE_URL,
        api_key=API_KEY
    )

    try:
        # 1. System Info
        print("\n--- 1. System Info ---")
        storage = registry.get_storage_usage()
        print(f"Storage Usage: {storage}")
        files = registry.list_files("/")
        print(f"Root Files: {len(files)}")

        # 2. Create a Model
        model_name = f"demo-model-{int(time.time())}"
        print(f"\n--- 2. Creating model: {model_name} ---")
        model = registry.create_model(
            name=model_name,
            storage_path=f"s3://models/{model_name}",
            description="A demo model created via Catapult SDK",
            metadata={"framework": "pytorch", "task": "classification"}
        )
        print(f"✅ Model created: {model.id}")

        # 3. Create a Release with Auto-Build
        version = "1.0.0"
        print(f"\n--- 3. Creating release: {version} ---")
        # Note: Auto-build requires backend worker, simulating here
        release = registry.create_release(
            model_name=model_name,
            version=version,
            tag=f"v{version}",
            digest=f"sha256:fake-{int(time.time())}", 
            metadata={"accuracy": 0.95},
            auto_build=False # Set to True to trigger build automatically
        )
        print(f"✅ Release created: {release.id}")

        # 4. Trigger Docker Build Manually
        print("\n--- 4. Triggering Docker Build ---")
        build = registry.create_build(
            release_id=release.id,
            image_tag=f"{model_name}:{version}",
            build_type="standard"
        )
        print(f"✅ Build triggered: {build.id}")
        
        # Poll Status
        print("   Polling build status...")
        for _ in range(5):
            b = registry.get_build(build.id)
            print(f"   Status: {b.status}")
            if b.status in ["success", "failed"]:
                break
            time.sleep(1)
            
        # 5. List Artifacts (if any)
        print("\n--- 5. Listing Artifacts ---")
        artifacts = registry.list_artifacts(release_id=release.id)
        print(f"Artifacts found: {len(artifacts)}")
        for art in artifacts:
            print(f" - {art.name} ({art.artifact_type})")

        # 6. Create a Deployment
        environment = "development"
        print(f"\n--- 6. Deploying to {environment} ---")
        deployment = registry.deploy(
            release_id=release.id,
            environment=environment,
            metadata={"replicas": 1, "region": "us-west-2"}
        )
        print(f"✅ Deployment created: {deployment.id}")

        # 7. Audit Logs
        print("\n--- 7. Recent Audit Logs ---")
        logs = registry.list_audit_logs(limit=5)
        for log in logs:
            print(f"[{log.created_at}] {log.action} on {log.resource_type}")

        # 8. Verify Data
        print("\n--- 8. Verifying Data ---")
        latest = registry.get_latest_release(model_name, environment)
        if latest and latest.id == release.id:
             print(f"✅ Verified latest release in {environment} matches: {latest.version}")
        else:
             print(f"❌ Verification failed: Latest release mismatch")

    except RegistryError as e:
        print(f"❌ API Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

    print("\n✨ Example completed successfully!")

if __name__ == "__main__":
    main()

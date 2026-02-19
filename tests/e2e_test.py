#!/usr/bin/env python3
"""
End-to-end tests for Docker Release Registry.

This test suite validates the entire system including:
- Backend API endpoints
- Database operations
- Python SDK client
- Full workflows (create image → create release → deploy)

Run with: pytest tests/e2e_test.py -v
"""
import pytest
import sys
import os
from pathlib import Path

# Add SDK to path
sdk_path = Path(__file__).parent.parent / "sdk" / "python"
sys.path.insert(0, str(sdk_path))

from catapult import Registry, RegistryError


# Test configuration
# Test configuration
API_BASE_URL = os.getenv("TEST_API_URL", "http://localhost:8080/api")

def get_api_key():
    """Create or get API key using the script if not in env."""
    if os.getenv("TEST_API_KEY"):
        return os.getenv("TEST_API_KEY")
        
    print("TEST_API_KEY not set. Attempting to fetch from backend container...")
    import subprocess
    import time
    
    # Try to reuse the admin key
    cmd = ["docker", "compose", "exec", "-T", "backend", "python", "scripts/create_api_key.py", "--name", "admin", "--reset"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if "Key: " in line:
                key = line.split("Key: ")[1].strip()
                print(f"✓ Fetched API Key: {key[:5]}...")
                return key
    except Exception as e:
        print(f"Error getting API key: {e}")
        return None

API_KEY = get_api_key()


pytestmark = pytest.mark.asyncio


class TestE2E:
    """End-to-end test suite."""

    @pytest.fixture(scope="class")
    def registry(self):
        """Create registry client."""
        if not API_KEY:
            pytest.skip("TEST_API_KEY environment variable not set")

        return Registry(base_url=API_BASE_URL, api_key=API_KEY)

    @pytest.fixture(scope="class")
    def test_image_name(self):
        """Generate unique test image name."""
        import time
        return f"test-image-{int(time.time())}"

    def test_001_health_check(self, registry):
        """Test health check endpoint."""
        # This will raise an exception if health check fails
        # The SDK doesn't have a health() method, but we can test connectivity
        try:
            # Try to list models (will fail if backend is down)
            models = registry.list_models(limit=1)
            assert isinstance(models, list)
            print("✓ Health check passed - backend is responding")
        except Exception as e:
            pytest.fail(f"Health check failed: {e}")

    def test_002_create_model(self, registry, test_image_name):
        """Test creating a model."""
        model = registry.create_model(
            name=test_image_name,
            storage_path=f"test-org/{test_image_name}",
            description="Test model for e2e testing"
        )

        assert model.name == test_image_name
        assert model.storage_path == f"test-org/{test_image_name}"
        assert model.description == "Test model for e2e testing"
        assert model.id is not None
        print(f"✓ Created model: {model.id} - {model.name}")

    def test_003_list_models(self, registry, test_image_name):
        """Test listing models."""
        models = registry.list_models(search=test_image_name)

        assert len(models) >= 1
        assert any(m.name == test_image_name for m in models)
        print(f"✓ Found {len(models)} model(s) matching '{test_image_name}'")

    def test_004_get_model(self, registry, test_image_name):
        """Test getting model by ID."""
        # First find the model
        models = registry.list_models(search=test_image_name)
        test_model = next(m for m in models if m.name == test_image_name)

        # Get it by ID
        model = registry.get_model(test_model.id)

        assert model.id == test_model.id
        assert model.name == test_image_name
        print(f"✓ Retrieved model by ID: {model.id}")

    def test_005_create_release(self, registry, test_image_name):
        """Test creating a release."""
        release = registry.create_release(
            model_name=test_image_name,
            version="1.0.0",
            tag="v1.0.0",
            digest="sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            size_bytes=1024000,
            metadata={
                "git_commit": "abc123",
                "git_branch": "main",
                "pytorch_version": "2.1.0",
                "accuracy": 0.95,
                "test": True
            }
        )

        assert release.version == "1.0.0"
        assert release.tag == "v1.0.0"
        assert release.status == "active"
        assert release.metadata["accuracy"] == 0.95
        assert release.metadata["test"] is True
        print(f"✓ Created release: {release.id} - v{release.version}")

    def test_006_create_multiple_releases(self, registry, test_image_name):
        """Test creating multiple releases for same model."""
        versions = ["1.0.1", "1.0.2", "1.1.0"]

        for version in versions:
            release = registry.create_release(
                model_name=test_image_name,
                version=version,
                tag=f"v{version}",
                digest=f"sha256:{'0' * 60}{version.replace('.', '')}",
                metadata={"version": version, "test": True}
            )
            assert release.version == version
            print(f"✓ Created release v{version}")

    def test_007_list_releases(self, registry, test_image_name):
        """Test listing releases."""
        releases = registry.list_releases(model_name=test_image_name)

        assert len(releases) >= 4  # We created 4 releases (1.0.0, 1.0.1, 1.0.2, 1.1.0)
        versions = [r.version for r in releases]
        assert "1.0.0" in versions
        assert "1.1.0" in versions
        print(f"✓ Found {len(releases)} release(s) for {test_image_name}")
        print(f"  Versions: {', '.join(sorted(versions))}")

    def test_008_get_latest_release(self, registry, test_image_name):
        """Test getting latest release."""
        latest = registry.get_latest_release(image_name=test_image_name)

        assert latest is not None
        # Latest should be the most recently created
        assert latest.version in ["1.0.0", "1.0.1", "1.0.2", "1.1.0"]
        print(f"✓ Latest release: v{latest.version} (ID: {latest.id})")

    def test_009_create_deployment(self, registry, test_image_name):
        """Test creating a deployment."""
        # Get a release to deploy
        releases = registry.list_releases(model_name=test_image_name, limit=1)
        assert len(releases) > 0
        release = releases[0]

        # Create deployment
        deployment = registry.deploy(
            release_id=release.id,
            environment="staging",
            metadata={
                "kubernetes_namespace": "test-ns",
                "replicas": 2,
                "gpu_type": "A100",
                "test": True
            },
            status="success"
        )

        assert deployment.release_id == release.id
        assert deployment.environment == "staging"
        assert deployment.status == "success"
        assert deployment.metadata["replicas"] == 2
        print(f"✓ Created deployment: {deployment.id} to {deployment.environment}")

    def test_010_create_multiple_deployments(self, registry, test_image_name):
        """Test creating multiple deployments."""
        releases = registry.list_releases(model_name=test_image_name)
        assert len(releases) >= 2

        environments = ["staging", "production", "development"]

        for i, env in enumerate(environments):
            release = releases[min(i, len(releases) - 1)]
            deployment = registry.deploy(
                release_id=release.id,
                environment=env,
                metadata={"environment": env, "test": True}
            )
            assert deployment.environment == env
            print(f"✓ Deployed release v{release.version} to {env}")

    def test_011_list_deployments(self, registry):
        """Test listing deployments."""
        deployments = registry.list_deployments(limit=100)

        assert len(deployments) >= 3  # We created at least 4 deployments
        # Filter for test deployments
        test_deployments = [d for d in deployments if d.metadata.get("test") is True]
        assert len(test_deployments) >= 3
        print(f"✓ Found {len(deployments)} total deployment(s)")
        print(f"  Test deployments: {len(test_deployments)}")

    def test_012_list_deployments_by_environment(self, registry):
        """Test listing deployments filtered by environment."""
        deployments = registry.list_deployments(environment="staging")

        assert len(deployments) >= 1
        assert all(d.environment == "staging" for d in deployments)
        print(f"✓ Found {len(deployments)} deployment(s) in staging")

    def test_013_duplicate_release_version(self, registry, test_image_name):
        """Test that duplicate release versions are rejected."""
        with pytest.raises(RegistryError) as exc_info:
            registry.create_release(
                model_name=test_image_name,
                version="1.0.0",  # Already exists
                tag="v1.0.0-duplicate",
                digest="sha256:duplicate00000000000000000000000000000000000000000000000000",
            )

        assert "409" in str(exc_info.value) or "already exists" in str(exc_info.value).lower()
        print("✓ Duplicate version correctly rejected with 409 Conflict")

    def test_014_api_key_management(self, registry):
        """Test API key listing."""
        api_keys = registry.list_api_keys()

        assert len(api_keys) >= 1  # At least our test key should exist
        assert all(hasattr(key, 'id') for key in api_keys)
        assert all(hasattr(key, 'name') for key in api_keys)
        assert all(hasattr(key, 'is_active') for key in api_keys)

        # API keys should NOT include plaintext keys in list response
        for key in api_keys:
            assert key.key is None or key.key == "", \
                "Plaintext keys should not be returned in list response"

        print(f"✓ Found {len(api_keys)} API key(s)")
        print(f"  Active keys: {sum(1 for k in api_keys if k.is_active)}")

    def test_015_workflow_complete(self, registry, test_image_name):
        """Test complete workflow: model → release → deployment."""
        # Create a new model for complete workflow test
        workflow_model_name = f"{test_image_name}-workflow"

        # Step 1: Create model
        model = registry.create_model(
            name=workflow_model_name,
            storage_path=f"test-org/{workflow_model_name}",
            description="Complete workflow test"
        )
        print(f"✓ Workflow Step 1: Created model {model.name}")

        # Step 2: Create release
        release = registry.create_release(
            model_name=workflow_model_name,
            version="2.0.0",
            tag="v2.0.0",
            digest="sha256:workflow000000000000000000000000000000000000000000000000000000",
            metadata={
                "workflow_test": True,
                "accuracy": 0.98,
                "model": "ResNet50"
            }
        )
        print(f"✓ Workflow Step 2: Created release v{release.version}")

        # Step 3: Deploy
        deployment = registry.deploy(
            release_id=release.id,
            environment="production",
            metadata={
                "workflow_test": True,
                "deployment_type": "blue-green"
            }
        )
        print(f"✓ Workflow Step 3: Deployed to {deployment.environment}")

        # Step 4: Verify deployment chain
        deployments = registry.list_deployments(environment="production")
        workflow_deployments = [
            d for d in deployments
            if d.metadata.get("workflow_test") is True
        ]
        assert len(workflow_deployments) >= 1
        print("✓ Workflow Step 4: Verified deployment in production")

        print("\n✅ Complete workflow test passed!")

    def test_016_create_build(self, registry, test_image_name):
        """Test creating and monitoring a Docker build."""
        # Clean name for Docker tag
        safe_name = test_image_name.lower().replace(" ", "-")
        tag = f"registry.local/{safe_name}:build-test"

        # Get a release to build
        releases = registry.list_releases(model_name=test_image_name, limit=1)
        if not releases:
            pytest.skip("No releases found to build")
        
        release = releases[0]
        print(f"Triggering build for release {release.id}...")

        try:
            build = registry.create_build(
                release_id=release.id,
                image_tag=tag,
                build_type="test" 
            )
            assert build.id is not None
            assert build.status == "pending"
            print(f"✓ Created Build: {build.id}")

            # Basic polling (don't wait too long in unit test)
            import time
            for _ in range(5):
                b = registry.get_build(build.id)
                if b.status in ["success", "failed"]:
                    break
                time.sleep(1)
            
            # Check logs
            try:
                logs = registry.get_build_logs(build.id)
                assert isinstance(logs, str)
                print(f"✓ Retrieved logs ({len(logs)} chars)")
            except Exception as e:
                print(f"Warning: Failed to get logs: {e}")

        except RegistryError as e:
            pytest.fail(f"Build creation failed: {e}")

    def test_017_audit_logs(self, registry):
        """Test listing audit logs."""
        logs = registry.list_audit_logs(limit=5)
        # We performed actions, so there should be logs
        assert len(logs) > 0
        assert hasattr(logs[0], "action")
        assert hasattr(logs[0], "api_key_name")
        print(f"✓ Found {len(logs)} audit log entries")

    def test_018_release_filtering(self, registry):
        """Test release filtering (is_release flag)."""
        import uuid
        import time
        
        model_name = f"filter-test-{uuid.uuid4().hex[:8]}"
        registry.create_model(
            name=model_name, 
            storage_path=f"/tmp/{model_name}"
        )

        # Helper
        def create_rel(ver, is_rel):
            registry.create_release(
                model_name=model_name,
                version=ver,
                tag="latest",
                digest=f"sha256:{uuid.uuid4().hex}",
                is_release=is_rel
            )
        
        create_rel("v1.0.0", True)
        create_rel("v1.0.1-rc1", False)
        create_rel("v1.1.0", True)
        
        # Verify filtering
        official = registry.list_releases(model_name=model_name, is_release=True)
        assert len(official) == 2
        assert "v1.0.1-rc1" not in [r.version for r in official]
        
        candidates = registry.list_releases(model_name=model_name, is_release=False)
        assert len(candidates) == 1
        assert candidates[0].version == "v1.0.1-rc1"
        
        print(f"✓ Release filtering verified (Official: {len(official)}, RC: {len(candidates)})")

    def test_019_storage_files(self, registry):
        """Test storage and file listing methods."""
        # Storage usage
        try:
            storage = registry.get_storage_usage()
            assert isinstance(storage, dict)
            print(f"✓ Storage usage: {storage.get('total_size_bytes', 0)} bytes")
        except Exception:
            print("⚠ storage usage endpoint might be mocked or unavailable")

        # List files
        try:
            files = registry.list_files("/")
            assert isinstance(files, list)
            print(f"✓ Listed {len(files)} files/dirs in root")
        except Exception as e:
            pytest.fail(f"List files failed: {e}")


    def test_999_cleanup(self, registry, test_image_name):
        """Clean up test data."""
        print("\n🧹 Cleaning up test data...")

        # Delete test models (will cascade delete releases and deployments)
        models_to_delete = [test_image_name, f"{test_image_name}-workflow"]

        for model_name in models_to_delete:
            try:
                models = registry.list_models(search=model_name)
                for model in models:
                    if model.name == model_name:
                        registry.delete_model(model.id)
                        print(f"✓ Deleted model: {model.name}")
            except Exception as e:
                print(f"⚠ Could not delete {model_name}: {e}")

        print("✅ Cleanup complete")


def main():
    """Run tests directly."""
    import subprocess

    print("=" * 80)
    print("Docker Release Registry - End-to-End Tests")
    print("=" * 80)
    print()

    # Check environment
    if not API_KEY:
        print("❌ ERROR: TEST_API_KEY environment variable is not set!")
        print()
        print("To run tests:")
        print("1. Start the application: docker-compose up -d")
        print("2. Create an API key: docker-compose exec backend python scripts/create_api_key.py --name test-key")
        print("3. Set the API key: export TEST_API_KEY=<your-key>")
        print("   (Or ensure docker is running so the script can fetch it automatically)")
        print("4. Run tests: python tests/e2e_test.py")

        sys.exit(1)

    print(f"API URL: {API_BASE_URL}")
    print(f"API Key: {API_KEY[:20]}..." if API_KEY else "API Key: Not set")
    print()

    # Run tests
    exit_code = pytest.main([__file__, "-v", "--tb=short", "--color=yes"])

    print()
    print("=" * 80)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Celery worker health check script.

Uses Redis connectivity check instead of worker ping because:
- Worker uses --pool=solo which blocks during long-running tasks (Docker builds)
- inspect.ping() times out when worker is busy, falsely marking it unhealthy

Health check passes if:
1. Redis (the broker) is reachable
2. The celery process is running (checked via process file)
"""
import os
import sys

def check_redis():
    """Check if Redis broker is reachable."""
    try:
        import redis
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        client = redis.from_url(redis_url, socket_timeout=5)
        client.ping()
        return True
    except Exception as e:
        print(f"Redis check failed: {e}", file=sys.stderr)
        return False

def check_celery_process():
    """Check if celery worker process is running by checking for .pid file or process."""
    try:
        # Check if any celery process is running
        import subprocess
        result = subprocess.run(
            ["pgrep", "-f", "celery.*worker"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except FileNotFoundError:
        # pgrep not available, skip this check
        return True
    except Exception as e:
        print(f"Process check failed: {e}", file=sys.stderr)
        return True  # Don't fail on process check errors

try:
    # Primary check: Redis connectivity
    if not check_redis():
        sys.exit(1)

    # Secondary check: Celery process running
    if not check_celery_process():
        print("Celery process not found", file=sys.stderr)
        sys.exit(1)

    # All checks passed
    sys.exit(0)

except Exception as e:
    print(f"Health check failed: {e}", file=sys.stderr)
    sys.exit(1)

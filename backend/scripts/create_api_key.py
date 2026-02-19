#!/usr/bin/env python3
"""
Script to create API keys.
Usage: python scripts/create_api_key.py --name "my-key"
"""
import argparse
import asyncio
import secrets
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.database import async_session_maker
from app.core.security import hash_api_key
from app.models.api_key import ApiKey
from app.models.docker_build import DockerBuild # Fix SQLAlchemy registry error


async def create_api_key(name: str, key_value: str = None, expires_at: str = None, reset: bool = False, role: str = "operator") -> tuple[str, str]:
    """
    Create a new API key.

    Args:
        name: Name for the API key
        key_value: Optional specific key value to use (plaintext)
        expires_at: Optional expiration date (ISO format)
        reset: Whether to reset the key if it already exists
        role: Role for the API key (admin, operator, viewer)

    Returns:
        Tuple of (key_id, plaintext_key)
    """
    # Generate secure random key if not provided
    if key_value:
        plaintext_key = key_value
        prefix = None # Manually provided keys might not have prefix unless we parse it. 
                      # For simplicity, if value is provided, we check if it has '.'
        if "." in plaintext_key:
            prefix = plaintext_key.split(".", 1)[0]
    else:
        # Generate new format: prefix.secret
        # Prefix: 8 chars (4 bytes)
        prefix = secrets.token_hex(4)
        secret = secrets.token_urlsafe(40)
        plaintext_key = f"{prefix}.{secret}"

    # Hash the key
    key_hash = hash_api_key(plaintext_key, settings.API_KEY_SALT)

    # Create database entry
    async with async_session_maker() as session:
        # Check if key exists
        from sqlalchemy import select
        result = await session.execute(select(ApiKey).where(ApiKey.name == name))
        existing_key = result.scalar_one_or_none()

        if existing_key:
            if not reset:
                raise ValueError(f"API key with name '{name}' already exists. Use --reset to update it.")

            # Update existing key
            existing_key.key_hash = key_hash
            existing_key.prefix = prefix
            existing_key.is_active = True
            existing_key.role = role
            if expires_at:
                from datetime import datetime
                existing_key.expires_at = datetime.fromisoformat(expires_at)

            api_key = existing_key
        else:
            # Create new key
            api_key = ApiKey(
                name=name,
                key_hash=key_hash,
                prefix=prefix,
                is_active=True,
                role=role,
            )

            if expires_at:
                from datetime import datetime
                api_key.expires_at = datetime.fromisoformat(expires_at)

            session.add(api_key)

        await session.commit()
        await session.refresh(api_key)

        return str(api_key.id), plaintext_key


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create API key")
    parser.add_argument("--name", required=True, help="Name for the API key")
    parser.add_argument("--value", help="Specific key value to use (plaintext)")
    parser.add_argument("--expires", help="Expiration date (ISO format)")
    parser.add_argument("--reset", action="store_true", help="Reset key if it already exists")
    parser.add_argument("--role", default="operator", choices=["admin", "operator", "viewer"], help="Role for the API key")

    args = parser.parse_args()

    try:
        key_id, plaintext_key = await create_api_key(args.name, args.value, args.expires, args.reset, args.role)

        print("\n" + "=" * 80)
        print("API Key Created Successfully!")
        print("=" * 80)
        print(f"Name:       {args.name}")
        print(f"Role:       {args.role}")
        print(f"ID:         {key_id}")
        print(f"Key:        {plaintext_key}")
        print("\n⚠️  IMPORTANT: Save this key now! It will not be shown again.")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error creating API key: {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

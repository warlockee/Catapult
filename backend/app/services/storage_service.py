"""
Service for Ceph storage handling.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.core.config import settings


class StorageService:
    """Service for handling file storage operations on Ceph filesystem."""

    def __init__(self):
        """Initialize storage service."""
        self.ceph_mount_path = Path(settings.CEPH_MOUNT_PATH)
        self.local_storage_path = Path(settings.LOCAL_STORAGE_PATH)

    def get_release_path(self, image_name: str, version: str) -> str:
        """
        Generate a storage path for a release.

        Args:
            image_name: Name of the image
            version: Release version

        Returns:
            Storage path relative to Ceph mount
        """
        # Sanitize image name for filesystem
        safe_image_name = image_name.replace("/", "_").replace(":", "_")

        # Create path: {model_dir}/{image_name}/{version}/
        rel_path = f"{settings.MODEL_STORAGE_DIR}/{safe_image_name}/{version}"

        return rel_path

    def get_absolute_path(self, relative_path: str) -> Path:
        """
        Get absolute path from relative path.

        Args:
            relative_path: Relative path from Ceph mount

        Returns:
            Absolute path
        """
        return self.ceph_mount_path / relative_path

    def ensure_directory(self, relative_path: str) -> Path:
        """
        Ensure a directory exists in Ceph storage.

        Args:
            relative_path: Relative path from Ceph mount

        Returns:
            Absolute path to the directory

        Raises:
            OSError: If directory creation fails
        """
        abs_path = self.get_absolute_path(relative_path)
        abs_path.mkdir(parents=True, exist_ok=True)
        return abs_path

    def path_exists(self, relative_path: str) -> bool:
        """
        Check if a path exists in Ceph storage.

        Args:
            relative_path: Relative path from Ceph mount

        Returns:
            True if path exists, False otherwise
        """
        abs_path = self.get_absolute_path(relative_path)
        return abs_path.exists()

    def get_directory_size(self, relative_path: str) -> int:
        """
        Get total size of a directory in bytes.

        Args:
            relative_path: Relative path from Ceph mount

        Returns:
            Total size in bytes
        """
        abs_path = self.get_absolute_path(relative_path)

        if not abs_path.exists():
            return 0

        total_size = 0
        for dirpath, dirnames, filenames in os.walk(abs_path):
            for filename in filenames:
                filepath = Path(dirpath) / filename
                total_size += filepath.stat().st_size

        return total_size

    def list_items(self, relative_path: str) -> list[dict]:
        """
        List items (files and directories) in a directory.

        Args:
            relative_path: Relative path from Ceph mount

        Returns:
            List of dictionaries with item details
        """
        abs_path = self.get_absolute_path(relative_path)
        
        # Security check: Ensure we don't traverse above root
        try:
             abs_path.relative_to(self.ceph_mount_path)
        except ValueError:
             # If path is not relative to ceph mount (e.g. /etc)
             # But get_absolute_path joins them. 
             # Only if relative_path contained ".."
             # resolve() handles symlinks/..
             if not str(abs_path.resolve()).startswith(str(self.ceph_mount_path.resolve())):
                 return []

        if not abs_path.exists() or not abs_path.is_dir():
            return []

        items = []
        for f in abs_path.iterdir():
            try:
                stat = f.stat()
                items.append({
                    "name": f.name,
                    "is_directory": f.is_dir(),
                    "size_bytes": stat.st_size if f.is_file() else 0,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except (OSError, PermissionError):
                continue
                
        # Sort directories first, then files
        items.sort(key=lambda x: (not x["is_directory"], x["name"].lower()))
        return items

    def list_files(self, relative_path: str) -> list[str]:
        """
        List files in a directory (Legacy/Simple).

        Args:
            relative_path: Relative path from Ceph mount

        Returns:
            List of file names
        """
        items = self.list_items(relative_path)
        return [i["name"] for i in items if not i["is_directory"]]

    def delete_directory(self, relative_path: str) -> bool:
        """
        Delete a directory and all its contents.

        Args:
            relative_path: Relative path from Ceph mount

        Returns:
            True if deleted, False if path doesn't exist

        Raises:
            OSError: If deletion fails
        """
        abs_path = self.get_absolute_path(relative_path)

        if not abs_path.exists():
            return False

        import shutil
        shutil.rmtree(abs_path)
        return True

    def get_file_metadata(self, relative_path: str) -> Optional[dict]:
        """
        Get metadata about a file or directory.

        Args:
            relative_path: Relative path from Ceph mount

        Returns:
            Dictionary with file metadata or None if file doesn't exist
        """
        abs_path = self.get_absolute_path(relative_path)

        if not abs_path.exists():
            return None

        stat = abs_path.stat()

        return {
            "size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_directory": abs_path.is_dir(),
            "is_file": abs_path.is_file(),
        }

    def is_ceph_mounted(self) -> bool:
        """
        Check if Ceph filesystem is mounted and accessible.

        Returns:
            True if Ceph is mounted and writable, False otherwise
        """
        if not self.ceph_mount_path.exists():
            return False

        # Try to create a test file
        test_file = self.ceph_mount_path / ".health_check"
        try:
            test_file.touch()
            test_file.unlink()
            return True
        except (OSError, PermissionError):
            return False

    def get_storage_usage(self) -> dict:
        """
        Get storage usage statistics for the Ceph mount.

        Returns:
            Dictionary with total, used, and free space in bytes
        """
        import shutil
        
        if not self.ceph_mount_path.exists():
            return {"total": 0, "used": 0, "free": 0}

        try:
            total, used, free = shutil.disk_usage(self.ceph_mount_path)
            return {
                "total": total,
                "used": used,
                "free": free
            }
        except OSError:
            return {"total": 0, "used": 0, "free": 0}


# Global storage service instance
storage_service = StorageService()

"""Gemini Files API lifecycle manager.

Handles upload, metadata tracking, expiry detection, toggle, delete, and reupload.
Sync API — uses threading.Lock for metadata access, no asyncio.

Extracted from kachaka-gemini file_service.py (gemini-files-api branch).
Key change: sync instead of async, class-based instead of module-level globals.
"""

import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone

from google import genai

logger = logging.getLogger("vlm_service.files")

EXPIRY_HOURS = 48


class FileManager:
    """Manage files uploaded to Gemini Files API with local metadata tracking."""

    def __init__(self, data_dir: str):
        """Initialize with a data directory for local file + metadata storage.

        Args:
            data_dir: Directory path for uploads/ and files.json.
        """
        self._data_dir = data_dir
        self._uploads_dir = os.path.join(data_dir, "uploads")
        self._meta_file = os.path.join(data_dir, "files.json")
        self._lock = threading.Lock()

    def _ensure_dirs(self):
        os.makedirs(self._uploads_dir, exist_ok=True)

    def _load_meta(self) -> list[dict]:
        if not os.path.exists(self._meta_file):
            return []
        with open(self._meta_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                return []
        return data if isinstance(data, list) else []

    def _save_meta(self, records: list[dict]):
        self._ensure_dirs()
        tmp = self._meta_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._meta_file)

    @staticmethod
    def _is_expired(record: dict) -> bool:
        expires_at = record.get("expires_at", "")
        if not expires_at:
            return True
        try:
            exp = datetime.fromisoformat(expires_at)
            return datetime.now(timezone.utc) >= exp
        except (ValueError, TypeError):
            return True

    def upload(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        api_key: str,
    ) -> dict:
        """Save locally, upload to Gemini, store metadata. Returns new record."""
        self._ensure_dirs()

        file_id = uuid.uuid4().hex[:12]
        base_name = os.path.basename(filename)
        safe_name = f"{file_id}_{re.sub(r'[^a-zA-Z0-9._-]', '_', base_name)}"
        local_path = os.path.join(self._uploads_dir, safe_name)

        with open(local_path, "wb") as f:
            f.write(file_bytes)

        client = genai.Client(api_key=api_key)
        try:
            gemini_file = client.files.upload(file=local_path)
        except Exception:
            if os.path.exists(local_path):
                os.remove(local_path)
            raise

        now = datetime.now(timezone.utc)
        record = {
            "id": file_id,
            "filename": filename,
            "mime_type": mime_type,
            "size_bytes": len(file_bytes),
            "local_path": local_path,
            "gemini_name": gemini_file.name,
            "gemini_uri": gemini_file.uri,
            "uploaded_at": now.isoformat(),
            "expires_at": (now + timedelta(hours=EXPIRY_HOURS)).isoformat(),
            "enabled": True,
        }

        with self._lock:
            records = self._load_meta()
            records.append(record)
            self._save_meta(records)

        logger.info("Uploaded %s -> %s", filename, gemini_file.name)
        return record

    def list_files(self) -> list[dict]:
        """Return all records with computed 'expired' field."""
        records = self._load_meta()
        for r in records:
            r["expired"] = self._is_expired(r)
        return records

    def toggle(self, file_id: str, enabled: bool) -> bool:
        """Toggle enabled flag. Returns True if found."""
        with self._lock:
            records = self._load_meta()
            found = False
            for r in records:
                if r["id"] == file_id:
                    r["enabled"] = enabled
                    found = True
                    break
            if found:
                self._save_meta(records)
        return found

    def delete(self, file_id: str, api_key: str = "") -> bool:
        """Delete from Gemini + local disk + metadata. Returns True if found."""
        records = self._load_meta()
        target = None
        for r in records:
            if r["id"] == file_id:
                target = r
                break
        if target is None:
            return False

        # Gemini-side deletion (best-effort)
        gemini_name = target.get("gemini_name", "")
        if gemini_name and api_key:
            try:
                client = genai.Client(api_key=api_key)
                client.files.delete(name=gemini_name)
            except Exception as exc:
                logger.warning("Gemini delete failed for %s: %s", gemini_name, exc)

        # Local file removal
        local_path = target.get("local_path", "")
        if local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
            except OSError as exc:
                logger.warning("Local delete failed for %s: %s", local_path, exc)

        with self._lock:
            records = self._load_meta()
            records = [r for r in records if r["id"] != file_id]
            self._save_meta(records)

        logger.info("Deleted file %s", file_id)
        return True

    def reupload(self, file_id: str, api_key: str) -> dict | None:
        """Re-upload a local file to Gemini (e.g. after expiry). Returns updated record."""
        records = self._load_meta()
        target = None
        for r in records:
            if r["id"] == file_id:
                target = r
                break
        if target is None:
            return None

        local_path = target.get("local_path", "")
        if not local_path or not os.path.exists(local_path):
            logger.warning("Local file missing for %s: %s", file_id, local_path)
            return None

        client = genai.Client(api_key=api_key)
        gemini_file = client.files.upload(file=local_path)

        now = datetime.now(timezone.utc)
        target["gemini_name"] = gemini_file.name
        target["gemini_uri"] = gemini_file.uri
        target["uploaded_at"] = now.isoformat()
        target["expires_at"] = (now + timedelta(hours=EXPIRY_HOURS)).isoformat()

        with self._lock:
            records = self._load_meta()
            for i, r in enumerate(records):
                if r["id"] == file_id:
                    records[i] = target
                    break
            self._save_meta(records)

        logger.info("Re-uploaded %s -> %s", file_id, gemini_file.name)
        return target

    def get_enabled_files(self) -> list[dict]:
        """Return Gemini refs (name, uri, mime_type) for enabled, non-expired files."""
        records = self._load_meta()
        result = []
        for r in records:
            if r.get("enabled") and not self._is_expired(r):
                name = r.get("gemini_name", "")
                uri = r.get("gemini_uri", "")
                if name and uri:
                    result.append({
                        "name": name,
                        "uri": uri,
                        "mime_type": r.get("mime_type", "application/octet-stream"),
                    })
        return result

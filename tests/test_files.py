import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest
from vlm_service.files import FileManager


@pytest.fixture
def tmp_data_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def file_manager(tmp_data_dir):
    return FileManager(data_dir=tmp_data_dir)


class TestFileManagerUpload:

    def test_upload_stores_metadata(self, file_manager):
        with patch("vlm_service.files.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_gemini_file = MagicMock()
            mock_gemini_file.name = "files/abc123"
            mock_gemini_file.uri = "https://gemini.googleapis.com/files/abc123"
            mock_client.files.upload.return_value = mock_gemini_file

            record = file_manager.upload(
                file_bytes=b"test content",
                filename="test.pdf",
                mime_type="application/pdf",
                api_key="test-key",
            )

            assert record["filename"] == "test.pdf"
            assert record["gemini_name"] == "files/abc123"
            assert record["gemini_uri"] == "https://gemini.googleapis.com/files/abc123"
            assert record["enabled"] is True
            assert "id" in record
            assert "expires_at" in record

    def test_upload_saves_local_file(self, file_manager):
        with patch("vlm_service.files.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_file = MagicMock()
            mock_file.name = "files/x"
            mock_file.uri = "uri://x"
            mock_client.files.upload.return_value = mock_file

            record = file_manager.upload(b"hello", "test.txt", "text/plain", "key")
            assert os.path.exists(record["local_path"])
            with open(record["local_path"], "rb") as f:
                assert f.read() == b"hello"


class TestFileManagerList:

    def test_list_empty(self, file_manager):
        assert file_manager.list_files() == []

    def test_list_with_files(self, file_manager):
        with patch("vlm_service.files.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_file = MagicMock()
            mock_file.name = "files/a"
            mock_file.uri = "uri://a"
            mock_client.files.upload.return_value = mock_file

            file_manager.upload(b"data", "a.pdf", "application/pdf", "key")
            files = file_manager.list_files()
            assert len(files) == 1
            assert "expired" in files[0]


class TestFileManagerToggle:

    def test_toggle_disable(self, file_manager):
        with patch("vlm_service.files.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_file = MagicMock()
            mock_file.name = "files/a"
            mock_file.uri = "uri://a"
            mock_client.files.upload.return_value = mock_file

            record = file_manager.upload(b"data", "a.pdf", "application/pdf", "key")
            assert file_manager.toggle(record["id"], enabled=False) is True
            files = file_manager.list_files()
            assert files[0]["enabled"] is False

    def test_toggle_not_found(self, file_manager):
        assert file_manager.toggle("nonexistent", enabled=True) is False


class TestFileManagerGetEnabled:

    def test_get_enabled_filters_disabled(self, file_manager):
        with patch("vlm_service.files.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_file = MagicMock()
            mock_file.name = "files/a"
            mock_file.uri = "uri://a"
            mock_client.files.upload.return_value = mock_file

            record = file_manager.upload(b"data", "a.pdf", "application/pdf", "key")
            file_manager.toggle(record["id"], enabled=False)
            enabled = file_manager.get_enabled_files()
            assert len(enabled) == 0


class TestFileManagerDelete:

    def test_delete_removes_record(self, file_manager):
        with patch("vlm_service.files.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_file = MagicMock()
            mock_file.name = "files/a"
            mock_file.uri = "uri://a"
            mock_client.files.upload.return_value = mock_file

            record = file_manager.upload(b"data", "a.pdf", "application/pdf", "key")
            assert file_manager.delete(record["id"], api_key="key") is True
            assert file_manager.list_files() == []

    def test_delete_not_found(self, file_manager):
        assert file_manager.delete("nonexistent") is False

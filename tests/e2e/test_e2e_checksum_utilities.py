"""
RED Phase Tests for Checksum Utilities.

Tests for calculate_md5 and verify_checksums functions.
TASK-003: File checksum utility implementation.
"""

import pytest
import tempfile
from pathlib import Path


class TestChecksumUtilities:
    """Tests for checksum utility functions."""

    def test_calculate_md5_valid_file(self, tmp_path: Path):
        """Test MD5 calculation for a valid file."""
        from voice_man.services.e2e_test_service import calculate_md5

        # Create a test file with known content
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        checksum = calculate_md5(test_file)

        # MD5 of "Hello, World!" is known
        assert checksum == "65a8e27d8879283831b664bd8b7f0ad4"
        assert len(checksum) == 32
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_calculate_md5_empty_file(self, tmp_path: Path):
        """Test MD5 calculation for an empty file."""
        from voice_man.services.e2e_test_service import calculate_md5

        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        checksum = calculate_md5(test_file)

        # MD5 of empty string is known
        assert checksum == "d41d8cd98f00b204e9800998ecf8427e"

    def test_calculate_md5_binary_file(self, tmp_path: Path):
        """Test MD5 calculation for a binary file."""
        from voice_man.services.e2e_test_service import calculate_md5

        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(bytes([0x00, 0xFF, 0x01, 0xFE]))

        checksum = calculate_md5(test_file)

        assert len(checksum) == 32
        assert isinstance(checksum, str)

    def test_calculate_md5_nonexistent_file(self, tmp_path: Path):
        """Test MD5 calculation for nonexistent file raises error."""
        from voice_man.services.e2e_test_service import calculate_md5

        nonexistent = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            calculate_md5(nonexistent)

    def test_calculate_md5_large_file(self, tmp_path: Path):
        """Test MD5 calculation for a large file (chunked reading)."""
        from voice_man.services.e2e_test_service import calculate_md5

        # Create a file larger than the chunk size (8192 bytes)
        test_file = tmp_path / "large.bin"
        content = b"A" * 100000  # 100KB
        test_file.write_bytes(content)

        checksum = calculate_md5(test_file)

        assert len(checksum) == 32
        # Verify consistency - same content should give same hash
        checksum2 = calculate_md5(test_file)
        assert checksum == checksum2

    def test_verify_checksums_matching(self):
        """Test verify_checksums returns True for matching checksums."""
        from voice_man.services.e2e_test_service import verify_checksums

        original = {
            "/path/file1.m4a": "abc123",
            "/path/file2.m4a": "def456",
        }
        current = {
            "/path/file1.m4a": "abc123",
            "/path/file2.m4a": "def456",
        }

        assert verify_checksums(original, current) is True

    def test_verify_checksums_mismatch(self):
        """Test verify_checksums returns False for mismatched checksums."""
        from voice_man.services.e2e_test_service import verify_checksums

        original = {
            "/path/file1.m4a": "abc123",
            "/path/file2.m4a": "def456",
        }
        current = {
            "/path/file1.m4a": "abc123",
            "/path/file2.m4a": "changed_hash",  # Modified!
        }

        assert verify_checksums(original, current) is False

    def test_verify_checksums_missing_file(self):
        """Test verify_checksums returns False when file is missing."""
        from voice_man.services.e2e_test_service import verify_checksums

        original = {
            "/path/file1.m4a": "abc123",
            "/path/file2.m4a": "def456",
        }
        current = {
            "/path/file1.m4a": "abc123",
            # file2 missing!
        }

        assert verify_checksums(original, current) is False

    def test_verify_checksums_extra_file(self):
        """Test verify_checksums returns False when extra file present."""
        from voice_man.services.e2e_test_service import verify_checksums

        original = {
            "/path/file1.m4a": "abc123",
        }
        current = {
            "/path/file1.m4a": "abc123",
            "/path/file2.m4a": "extra_file",  # Extra file!
        }

        assert verify_checksums(original, current) is False

    def test_verify_checksums_empty_sets(self):
        """Test verify_checksums with empty checksum dictionaries."""
        from voice_man.services.e2e_test_service import verify_checksums

        assert verify_checksums({}, {}) is True

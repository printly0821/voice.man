"""
RFC 3161 Timestamp Service for Forensic Evidence.

Implements RFC 3161 compliant timestamping with actual TSA server integration.
Uses rfc3161ng library for communication with RFC 3161 compliant TSA servers.

TAG: [FORENSIC-EVIDENCE-001]
Reference: SPEC-FORENSIC-EVIDENCE-001
Compliance: RFC 3161 (Internet X.509 Public Key Infrastructure Time-Stamp Protocol)
"""

import base64
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import rfc3161ng

logger = logging.getLogger(__name__)


class TimestampService:
    """
    RFC 3161 timestamp service for forensic evidence authentication.

    Provides RFC 3161 compliant timestamp token generation and verification
    with automatic fallback handling for legal evidence timestamping.

    Attributes:
        tsa_url: Primary TSA server URL
        fallback_tsa_url: Fallback TSA server URL
        hash_algorithm: Hash algorithm to use (default: SHA-256)
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        tsa_url: Optional[str] = None,
        fallback_tsa_url: Optional[str] = None,
        hash_algorithm: str = "sha256",
        timeout: int = 30,
    ):
        """
        Initialize the timestamp service.

        Args:
            tsa_url: Primary TSA server URL (default: FreeTSA)
            fallback_tsa_url: Fallback TSA server URL (default: None)
            hash_algorithm: Hash algorithm (sha256, sha384, sha512)
            timeout: Request timeout in seconds
        """
        # FreeTSA (free for development/testing)
        self.tsa_url = tsa_url or "https://freetsa.org/tsr"
        self.fallback_tsa_url = fallback_tsa_url
        self.hash_algorithm = hash_algorithm
        self.timeout = timeout

        # Map algorithm name to hashlib
        self._hash_func = getattr(hashlib, hash_algorithm)

        logger.info(
            f"TimestampService initialized: TSA={self.tsa_url}, "
            f"algorithm={hash_algorithm}, timeout={timeout}s"
        )

    def generate_timestamp(self, file_hash: str) -> Dict[str, Any]:
        """
        Generate RFC 3161 timestamp token for file hash.

        Attempts primary TSA server, then fallback, then local timestamp.

        Args:
            file_hash: SHA-256 hash of the file (hex string)

        Returns:
            Dict containing:
                - timestamp_token: Base64-encoded RFC 3161 token
                - timestamp_iso8601: ISO 8601 timestamp
                - tsa_url: TSA server URL used
                - serial_number: TSA serial number
                - tsa_certificate: Base64-encoded TSA certificate
                - policy_oid: TSA policy OID
                - is_rfc3161_compliant: Boolean
                - is_local: Boolean (True if fallback to local)
                - warning: Optional warning message

        Raises:
            Exception: If all timestamp methods fail
        """
        try:
            return self._generate_rfc3161_timestamp(file_hash, self.tsa_url)
        except Exception as e:
            logger.warning(f"Primary TSA failed ({self.tsa_url}): {e}")

            # Try fallback TSA if configured
            if self.fallback_tsa_url:
                try:
                    return self._generate_rfc3161_timestamp(file_hash, self.fallback_tsa_url)
                except Exception as fallback_error:
                    logger.warning(
                        f"Fallback TSA failed ({self.fallback_tsa_url}): {fallback_error}"
                    )

            # Final fallback to local timestamp
            logger.warning("All TSA services failed, using local timestamp")
            return self._generate_local_timestamp(file_hash)

    def _generate_rfc3161_timestamp(self, file_hash: str, tsa_url: str) -> Dict[str, Any]:
        """
        Generate RFC 3161 timestamp from TSA server using rfc3161ng.

        Args:
            file_hash: SHA-256 hash of the file (hex string)
            tsa_url: TSA server URL

        Returns:
            Dict containing RFC 3161 timestamp token and metadata

        Raises:
            Exception: If TSA service is unavailable or request fails
        """
        try:
            # Convert hex hash to bytes
            hash_bytes = bytes.fromhex(file_hash)

            # Create RemoteTimestamper instance
            timestamper = rfc3161ng.RemoteTimestamper(
                url=tsa_url,
                hashname=self.hash_algorithm,
                include_tsa_certificate=True,
                timeout=self.timeout,
            )

            # Send request to TSA server (returns DER-encoded bytes)
            logger.info(f"Sending RFC 3161 request to TSA: {tsa_url}")
            timestamp_bytes = timestamper.timestamp(digest=hash_bytes)

            # Decode timestamp response using asn1crypto.cms directly
            # FreeTSA returns ContentInfo directly, not TimeStampResp wrapper
            from asn1crypto import cms

            content_info = cms.ContentInfo.load(timestamp_bytes)
            encap_content = content_info["content"]["encap_content_info"]["content"]
            tst_info_bytes = encap_content.contents

            from asn1crypto import tsp

            tst_info_obj = tsp.TSTInfo.load(tst_info_bytes)
            tst_info = tst_info_obj.native

            # Extract metadata
            serial_number = tst_info.get("serial_number")
            gen_time = tst_info.get("gen_time")
            policy_id = tst_info.get("policy")

            # Encode token as base64 for storage
            timestamp_token_b64 = base64.b64encode(timestamp_bytes).decode("ascii")

            # Get TSA certificate if available
            tsa_certificate_b64 = None
            signed_data = content_info["content"]
            if "certificates" in signed_data and signed_data["certificates"]:
                tsa_certificate_b64 = base64.b64encode(
                    signed_data["certificates"][0].dump()
                ).decode("ascii")

            result = {
                "timestamp_token": timestamp_token_b64,
                "timestamp": gen_time.isoformat() if gen_time else None,  # For backward compatibility
                "timestamp_iso8601": gen_time.isoformat() if gen_time else None,
                "tsa_url": tsa_url,
                "serial_number": str(serial_number) if serial_number else None,
                "tsa_certificate": tsa_certificate_b64,
                "policy_oid": str(policy_id) if policy_id else None,
                "is_rfc3161_compliant": True,
                "is_local": False,
                "hash_algorithm": self.hash_algorithm,
                "file_hash": file_hash,
                "hash": file_hash,  # For backward compatibility with tests
                "algorithm": self.hash_algorithm,  # For backward compatibility
                "source": "rfc3161",
            }

            logger.info(
                f"RFC 3161 timestamp generated successfully: "
                f"serial={serial_number}, time={gen_time}"
            )

            return result

        except Exception as e:
            logger.error(f"RFC 3161 timestamp generation failed: {e}")
            raise

    def _generate_local_timestamp(self, file_hash: str) -> Dict[str, Any]:
        """
        Generate local timestamp as final fallback.

        Used only when all TSA services are unavailable.
        Records warning for legal compliance tracking.

        Args:
            file_hash: SHA-256 hash of the file

        Returns:
            Dict containing local timestamp with warning
        """
        now = datetime.now(timezone.utc)

        result = {
            "timestamp_token": None,  # No RFC 3161 token available
            "timestamp": now.isoformat(),  # For backward compatibility
            "timestamp_iso8601": now.isoformat(),
            "tsa_url": None,
            "serial_number": None,
            "tsa_certificate": None,
            "policy_oid": None,
            "is_rfc3161_compliant": False,
            "is_local": True,
            "hash_algorithm": self.hash_algorithm,
            "file_hash": file_hash,
            "hash": file_hash,  # For backward compatibility with tests
            "algorithm": self.hash_algorithm,  # For backward compatibility
            "source": "local",
            "warning": (
                "TSA service unavailable - using local timestamp. "
                "This may not be acceptable for legal evidence submission."
            ),
        }

        logger.warning(
            f"Local timestamp generated for hash {file_hash[:16]}... "
            "- RFC 3161 compliance compromised"
        )

        return result

    def verify_timestamp(
        self, file_hash: str, timestamp_token_b64: Optional[str], expected_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify RFC 3161 timestamp token.

        Args:
            file_hash: Original file hash (hex string) to verify against
            timestamp_token_b64: Base64-encoded RFC 3161 timestamp token (None for local timestamps)
            expected_hash: Expected hash value (optional, uses file_hash if not provided)

        Returns:
            Dict containing:
                - valid: Boolean indicating if timestamp is valid
                - hash_match: Boolean indicating if hash matches
                - ts_certificates: TSA certificates used
                - gen_time: Timestamp generation time
                - serial_number: TSA serial number
                - error: Error message if verification failed
        """
        # Handle local timestamp (None token)
        if timestamp_token_b64 is None:
            return {
                "valid": True,  # Local timestamp is considered "valid" but not RFC 3161 compliant
                "hash_match": True,
                "is_local": True,
                "warning": "Local timestamp - not RFC 3161 compliant",
            }

        try:
            # Handle if a dict is passed instead of timestamp_token_b64 (for backward compatibility)
            if isinstance(timestamp_token_b64, dict):
                token_dict = timestamp_token_b64
                timestamp_token_b64 = token_dict.get("timestamp_token")

            # If still None after extraction, return valid for local timestamp
            if timestamp_token_b64 is None:
                return {
                    "valid": True,
                    "hash_match": True,
                    "is_local": True,
                    "warning": "Local timestamp - not RFC 3161 compliant",
                }

            # Decode base64 token
            timestamp_token_bytes = base64.b64decode(timestamp_token_b64)

            # Load timestamp token
            from asn1crypto import cms, tsp

            timestamp_token = cms.ContentInfo.load(timestamp_token_bytes)

            # Extract TSTInfo from ParsableOctetString
            encap_content = timestamp_token["content"]["encap_content_info"]["content"]
            tst_info_bytes = encap_content.contents

            # Parse TSTInfo
            tst_info_obj = tsp.TSTInfo.load(tst_info_bytes)
            tst_info = tst_info_obj.native

            # Get generation time
            gen_time = tst_info.get("gen_time")

            # Get message imprint (hash)
            message_imprint = tst_info.get("message_imprint", {})
            hashed_algorithm = message_imprint.get("hash_algorithm", {}).get("algorithm")
            hashed_message = message_imprint.get("hashed_message", b"").hex() if message_imprint.get("hashed_message") else ""

            # Verify hash matches
            hash_to_verify = expected_hash or file_hash
            hash_match = hashed_message == hash_to_verify

            result = {
                "valid": hash_match,
                "hash_match": hash_match,
                "hash_algorithm": hashed_algorithm,
                "hashed_message": hashed_message,
                "expected_hash": hash_to_verify,
                "gen_time": gen_time.isoformat() if gen_time else None,
                "serial_number": str(tst_info.get("serial_number", "")) if tst_info.get("serial_number") else None,
                "policy_id": str(tst_info.get("policy", "")) if tst_info.get("policy") else None,
                "is_local": False,
                "error": None if hash_match else "Hash mismatch",
            }

            logger.info(
                f"Timestamp verification: valid={hash_match}, "
                f"time={gen_time}, serial={result['serial_number']}"
            )

            return result

        except Exception as e:
            logger.error(f"Timestamp verification failed: {e}")
            return {
                "valid": False,
                "hash_match": False,
                "error": str(e),
            }

    def get_timestamp_info(self, timestamp_token_b64: str) -> Dict[str, Any]:
        """
        Extract information from RFC 3161 timestamp token without verification.

        Args:
            timestamp_token_b64: Base64-encoded RFC 3161 timestamp token

        Returns:
            Dict containing timestamp information
        """
        try:
            timestamp_token_bytes = base64.b64decode(timestamp_token_b64)

            from asn1crypto import cms

            timestamp_token = cms.ContentInfo.load(timestamp_token_bytes)
            tst_info = timestamp_token["content"]["encap_content_info"]["content"]

            return {
                "gen_time": tst_info["gen_time"].native.isoformat(),
                "serial_number": str(tst_info["serial_number"].native),
                "policy_id": str(tst_info["policy_id"].native)
                if tst_info["policy_id"].native
                else None,
                "hash_algorithm": tst_info["message_imprint"]["hash_algorithm"]["algorithm"].native,
                "tsa_name": tst_info["tsa"].native,
                "accuracy": tst_info["accuracy"].native if "accuracy" in tst_info else None,
                "ordering": tst_info["ordering"].native if "ordering" in tst_info else None,
                "nonce": tst_info["nonce"].native if "nonce" in tst_info else None,
            }

        except Exception as e:
            logger.error(f"Failed to extract timestamp info: {e}")
            return {"error": str(e)}

    def save_timestamp_metadata(
        self, file_hash: str, token: Dict[str, Any], metadata_path: str
    ) -> None:
        """
        Save timestamp metadata to JSON file.

        Args:
            file_hash: SHA-256 hash of the file
            token: Timestamp token
            metadata_path: Path to save metadata JSON
        """
        metadata = {"file_hash": file_hash, "token": token}

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def verify_from_metadata(self, file_hash: str, metadata_path: str) -> bool:
        """
        Load timestamp metadata and verify.

        Args:
            file_hash: SHA-256 hash of the file to verify
            metadata_path: Path to metadata JSON

        Returns:
            bool: True if timestamp is valid, False otherwise
        """
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        token = metadata["token"]

        # If token has timestamp_token_b64, verify it
        if token.get("timestamp_token"):
            return self.verify_timestamp(file_hash=file_hash, timestamp_token_b64=token["timestamp_token"]).get(
                "valid", False
            )

        # Otherwise, just check hash match
        return token.get("file_hash") == file_hash


# Convenience function for quick timestamp generation
def generate_forensic_timestamp(file_hash: str) -> Dict[str, Any]:
    """
    Generate RFC 3161 timestamp for forensic evidence.

    Convenience function that creates a TimestampService instance
    and generates a timestamp.

    Args:
        file_hash: SHA-256 hash of the file (hex string)

    Returns:
        Dict containing timestamp token and metadata
    """
    service = TimestampService()
    return service.generate_timestamp(file_hash)


# Convenience function for quick timestamp verification
def verify_forensic_timestamp(file_hash: str, timestamp_token_b64: str) -> Dict[str, Any]:
    """
    Verify RFC 3161 timestamp for forensic evidence.

    Convenience function that creates a TimestampService instance
    and verifies a timestamp.

    Args:
        file_hash: Original file hash (hex string)
        timestamp_token_b64: Base64-encoded RFC 3161 timestamp token

    Returns:
        Dict containing verification result
    """
    service = TimestampService()
    return service.verify_timestamp(file_hash, timestamp_token_b64)

"""
Digital Signature Service for Forensic Evidence Authentication.

Implements RSA 2048-bit digital signatures for evidence integrity
according to Korean Criminal Procedure Law Article 313(2)(3).

TAG: [FORENSIC-EVIDENCE-001]
"""

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from datetime import datetime, timezone
import base64
import json
from typing import Tuple


class DigitalSignatureService:
    """
    Digital signature service for forensic evidence authentication.

    Provides RSA 2048-bit key generation, signature creation/verification,
    and signature metadata storage for legal evidence.
    """

    def __init__(self):
        """Initialize the digital signature service."""
        self.key_size = 2048
        self.algorithm = "RSA-2048-PSS-SHA256"

    def generate_key_pair(self) -> Tuple[str, str]:
        """
        Generate RSA 2048-bit key pair.

        Returns:
            Tuple[str, str]: (private_key_pem, public_key_pem)
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=self.key_size, backend=default_backend()
        )

        public_key = private_key.public_key()

        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        # Serialize public key
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

        return private_pem, public_pem

    def sign_hash(self, file_hash: str, private_key_pem: str) -> str:
        """
        Sign a file hash with private key.

        Args:
            file_hash: SHA-256 hash of the file (hex string)
            private_key_pem: Private key in PEM format

        Returns:
            str: Base64-encoded signature
        """
        # Load private key
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode("utf-8"), password=None, backend=default_backend()
        )

        # Sign the hash (convert hex to bytes first)
        hash_bytes = bytes.fromhex(file_hash)

        signature = private_key.sign(
            hash_bytes,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )

        # Return base64-encoded signature
        return base64.b64encode(signature).decode("utf-8")

    def verify_signature(self, file_hash: str, signature_b64: str, public_key_pem: str) -> bool:
        """
        Verify a signature with public key.

        Args:
            file_hash: SHA-256 hash of the file (hex string)
            signature_b64: Base64-encoded signature
            public_key_pem: Public key in PEM format

        Returns:
            bool: True if signature is valid, False otherwise
        """
        try:
            # Load public key
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode("utf-8"), backend=default_backend()
            )

            # Decode signature
            signature = base64.b64decode(signature_b64)

            # Convert hash to bytes
            hash_bytes = bytes.fromhex(file_hash)

            # Verify signature
            public_key.verify(
                signature,
                hash_bytes,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )

            return True

        except Exception:
            return False

    def save_signature_metadata(
        self,
        file_path: str,
        file_hash: str,
        signature: str,
        public_key: str,
        metadata_path: str,
    ) -> None:
        """
        Save signature metadata to JSON file.

        Args:
            file_path: Path to the evidence file
            file_hash: SHA-256 hash of the file
            signature: Base64-encoded signature
            public_key: Public key in PEM format
            metadata_path: Path to save metadata JSON
        """
        metadata = {
            "file_path": file_path,
            "file_hash": file_hash,
            "signature": signature,
            "public_key": public_key,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "algorithm": self.algorithm,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def verify_from_metadata(self, file_hash: str, metadata_path: str) -> bool:
        """
        Load signature metadata and verify.

        Args:
            file_hash: SHA-256 hash of the file to verify
            metadata_path: Path to metadata JSON

        Returns:
            bool: True if signature is valid, False otherwise
        """
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return self.verify_signature(
            file_hash=file_hash,
            signature_b64=metadata["signature"],
            public_key_pem=metadata["public_key"],
        )

    def get_key_size(self, public_key_pem: str) -> int:
        """
        Get the key size of a public key.

        Args:
            public_key_pem: Public key in PEM format

        Returns:
            int: Key size in bits
        """
        public_key = serialization.load_pem_public_key(
            public_key_pem.encode("utf-8"), backend=default_backend()
        )

        return public_key.key_size

#!/usr/bin/env python3
"""
E2E Test Runner with PyTorch 2.8+ Compatibility Patch

This wrapper script patches torch.load to work with weights_only=False
for WhisperX/pyannote-audio compatibility on PyTorch 2.6+.
"""

import sys

# Apply torch.load patch BEFORE any other imports
import torch

original_load = torch.load


def patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_load(*args, **kwargs)


torch.load = patched_load
import torch.serialization

torch.serialization.load = patched_load

# Patch lightning_fabric if available
try:
    import lightning_fabric.utilities.cloud_io as cloud_io

    cloud_io.torch.load = patched_load
except ImportError:
    pass

print("torch.load patched for PyTorch 2.6+ compatibility")

# Now run the actual e2e batch test
if __name__ == "__main__":
    import asyncio

    # Remove this script from argv
    sys.argv[0] = "scripts/e2e_batch_test.py"

    # Import and run the main function from e2e_batch_test
    from e2e_batch_test import main

    asyncio.run(main())

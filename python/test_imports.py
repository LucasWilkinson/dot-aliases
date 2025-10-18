#!/usr/bin/env python3
"""Test that all infrastructure modules can be imported."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test all imports work."""
    print("Testing imports...")
    
    # Test individual modules
    from python.vllm_test_infra import utils
    print("✓ utils")
    
    from python.vllm_test_infra import logging
    print("✓ logging")
    
    from python.vllm_test_infra import process
    print("✓ process")
    
    from python.vllm_test_infra import server
    print("✓ server")
    
    from python.vllm_test_infra import git
    print("✓ git")
    
    from python.vllm_test_infra import config
    print("✓ config")
    
    from python.vllm_test_infra import ui
    print("✓ ui")
    
    # Test package imports
    from python.vllm_test_infra import (
        Config,
        GitManager,
        LogManager,
        ProcessManager,
        VLLMServer,
        UIManager,
        note,
    )
    print("✓ All package imports")
    
    # Test basic functionality
    note("Test message")
    print("✓ Basic functionality")
    
    print("\n✅ All imports successful!")
    return 0

if __name__ == "__main__":
    sys.exit(test_imports())


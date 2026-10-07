"""
pytest configuration and shared fixtures for tests.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Regenerate golden baseline files in tests/golden/",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

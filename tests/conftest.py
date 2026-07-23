"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from oncoemotion.factory import build_default_mapper
from oncoemotion.terminology.pro_ctcae import load_pro_ctcae
from oncoemotion.terminology.ctcae import load_ctcae


@pytest.fixture(scope="session")
def pro_library():
    return load_pro_ctcae()


@pytest.fixture(scope="session")
def ctcae_dict():
    return load_ctcae(allow_synthetic=True)


@pytest.fixture(scope="session")
def mapper():
    return build_default_mapper()

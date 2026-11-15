"""Opt-in sparse decode geometry, using the existing FP32-reference tests."""

import pytest
import torch

if not torch.cuda.is_available() or not torch.version.hip:
    pytest.skip("requires ROCm", allow_module_level=True)

from flash_mla.amd.ops import decode_head_tile


@pytest.mark.parametrize("value", ["0", "64", "invalid"])
def test_invalid_head_tile_fails_closed(monkeypatch, value):
    monkeypatch.setenv("FLASH_MLA_AMD_DECODE_HEAD_TILE", value)
    with pytest.raises(ValueError):
        decode_head_tile()


def test_default_preserves_original_tile(monkeypatch):
    monkeypatch.delenv("FLASH_MLA_AMD_DECODE_HEAD_TILE", raising=False)
    assert decode_head_tile() == 16
    monkeypatch.setenv("FLASH_MLA_AMD_DECODE_HEAD_TILE", "32")
    assert decode_head_tile() == 32

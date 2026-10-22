"""CPU-only package loading tests for CUDA, Ascend and ROCm isolation."""

import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPORTS = (
    "get_mla_metadata", "flash_mla_with_kvcache", "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func", "flash_attn_varlen_kvpacked_func",
    "flash_mla_sparse_fwd",
)


@pytest.mark.parametrize("backend", ["cuda", "npu", "hip"])
def test_native_extension_selection(monkeypatch, backend):
    for name in tuple(sys.modules):
        if name == "flash_mla" or name.startswith("flash_mla."):
            monkeypatch.delitem(sys.modules, name)
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip="7.2" if backend == "hip" else None)
    torch.ops = types.SimpleNamespace(load_library=Mock())
    monkeypatch.setitem(sys.modules, "torch", torch)
    build_info = types.ModuleType("flash_mla._build_info")
    build_info.BACKEND = backend
    build_info.EXTENSION = f"test_{backend}.so"
    if backend != "hip":
        monkeypatch.setitem(sys.modules, "flash_mla._build_info", build_info)
    interface = types.ModuleType("flash_mla.flash_mla_interface")
    for name in EXPORTS:
        setattr(interface, name, Mock(name=name))
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    fused = types.ModuleType("flash_mla.fused_norm_rope_attn_rope_cast")
    monkeypatch.setitem(sys.modules, fused.__name__, fused)
    if backend == "npu":
        monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    extension_path = ROOT / "flash_mla" / build_info.EXTENSION
    real_is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda path: path == extension_path or real_is_file(path))
    spec = importlib.util.spec_from_file_location(
        "flash_mla", ROOT / "flash_mla/__init__.py",
        submodule_search_locations=[str(ROOT / "flash_mla")],
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "flash_mla", package)
    spec.loader.exec_module(package)
    assert package.BACKEND == backend
    assert package.__all__ == [*EXPORTS, "fused_norm_rope_attn_rope_cast"]
    for name in EXPORTS:
        assert getattr(package, name) is getattr(interface, name)
    if backend == "hip":
        torch.ops.load_library.assert_not_called()
    else:
        torch.ops.load_library.assert_called_once_with(extension_path)

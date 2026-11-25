"""Tests for ledger.profile build prerequisite checks and failure diagnostics."""

import subprocess

import pytest

from ledger import profile


def _fake_nvcc(tmp_path):
    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\n")
    nvcc.chmod(0o755)
    return nvcc


def test_build_missing_cupti_headers_gives_clear_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("CUPTI_INC", raising=False)
    monkeypatch.delenv("CUPTI_LIB", raising=False)
    nvcc = _fake_nvcc(tmp_path)

    with pytest.raises(SystemExit) as exc:
        profile.build_capture(nvcc)
    msg = str(exc.value)
    assert "cupti.h" in msg
    assert "CUPTI_INC" in msg


def test_build_missing_cupti_lib_gives_clear_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("CUPTI_INC", raising=False)
    monkeypatch.delenv("CUPTI_LIB", raising=False)
    nvcc = _fake_nvcc(tmp_path)
    inc = tmp_path / "cuda" / "extras" / "CUPTI" / "include"
    inc.mkdir(parents=True)
    (inc / "cupti.h").write_text("// stub\n")

    with pytest.raises(SystemExit) as exc:
        profile.build_capture(nvcc)
    msg = str(exc.value)
    assert "libcupti" in msg
    assert "CUPTI_LIB" in msg


def test_build_missing_sanitizer_gives_clear_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("SANITIZER", raising=False)
    nvcc = _fake_nvcc(tmp_path)

    with pytest.raises(SystemExit) as exc:
        profile.build_library("access", profile.ACCESS_SOURCES, ("libledger_access.so", "patches.fatbin"), nvcc, "ARCH=121")
    msg = str(exc.value)
    assert "Compute Sanitizer" in msg
    assert "SANITIZER" in msg


def test_build_make_failure_summarizes_error(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    nvcc = _fake_nvcc(tmp_path)
    inc = tmp_path / "cuda" / "extras" / "CUPTI" / "include"
    lib = tmp_path / "cuda" / "extras" / "CUPTI" / "lib64"
    inc.mkdir(parents=True)
    lib.mkdir(parents=True)
    (inc / "cupti.h").write_text("// stub\n")
    (lib / "libcupti.so").write_text("")

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            ["make"], 2, stdout="", stderr="/usr/bin/ld: cannot find -lcuda\ncollect2: error: ld returned 1 exit status\n"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        profile.build_capture(nvcc)
    msg = str(exc.value)
    assert "libcuda" in msg
    assert "LIBRARY_PATH" in msg

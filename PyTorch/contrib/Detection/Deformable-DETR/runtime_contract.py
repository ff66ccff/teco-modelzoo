# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
# Copyright (c) 2026 Sichuan University contributors.

"""Capability contract shared by the Deformable-DETR delivery entry points.

The delivery entries must run on the SDAA stack: vendor Python 3.12 / Torch-SDAA
for the local evidence, official ModelZoo Python 3.11 / PyTorch 2.7.1 for the
committee environment.  This module replaces the historical
``interpreter path == /home/py312/bin/python`` assertions with checks on the
*real* capabilities of the running interpreter:

* Python major.minor is a supported release (3.11 official or 3.12 vendor),
* ``torch`` and ``torchvision`` import and report usable versions,
* ``torch_sdaa`` imports and exposes at least one SDAA device -- CPU fallback is
  never accepted,
* the paired ``tecoops`` wheel is installed and importable, exposes
  ``ms_deform_attn`` / ``ms_deform_attn_forward`` / ``ms_deform_attn_backward``
  and ships its native objects (``_torch_ext*.so``, ``libteco_ops*.so``) next to
  the package.  ``TECOOPS_API_ROOT`` is an optional *development* override for a
  source-tree package; when it is set, ``tecoops`` must resolve under it so an
  unrelated wheel can never be used silently.

Every failure prints an actionable hint and stops the entry; nothing here
silently degrades, and the interpreter may be selected with ``PYTHON``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

VENDOR_PYTHON = "/home/py312/bin/python"
SUPPORTED_PYTHON_MINORS = ((3, 11), (3, 12))
MINIMUM_TORCH_MINOR = (2, 0)
REQUIRED_TECOOPS_ATTRS = ("ms_deform_attn_forward", "ms_deform_attn_backward")


class CapabilityError(RuntimeError):
    """Raised when the running interpreter cannot honour the delivery contract."""

    def __init__(self, reason: str, hints: tuple[str, ...] = ()):
        super().__init__(reason)
        self.reason = reason
        self.hints = tuple(hints)


def select_python() -> str:
    """Return the interpreter the shell entries should use (``PYTHON`` wins)."""
    return os.environ.get("PYTHON", VENDOR_PYTHON)


def _python_hints() -> tuple[str, ...]:
    return (
        "official ModelZoo environment: Python 3.11 + PyTorch 2.7.1",
        f"local vendor environment: PYTHON={VENDOR_PYTHON} (Python 3.12 + Torch-SDAA)",
        "select an interpreter explicitly with PYTHON=/path/to/python and re-run the entry",
    )


def _sdaa_hints() -> tuple[str, ...]:
    return (
        "source /opt/tecoai/setvars.sh (the SDAA runtime) before running the entry",
        f"or run through the delivery entry with PYTHON={VENDOR_PYTHON}",
        "CPU fallback is not a supported delivery mode; the SDAA stack is mandatory",
    )


def _check_python() -> str:
    if sys.version_info[:2] not in SUPPORTED_PYTHON_MINORS:
        supported = ", ".join(f"{major}.{minor}" for major, minor in SUPPORTED_PYTHON_MINORS)
        raise CapabilityError(
            f"interpreter {sys.executable} reports Python {sys.version.split()[0]}; "
            f"supported releases are {supported}",
            _python_hints(),
        )
    return sys.version.split()[0]


def _check_torch() -> tuple[str, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on the host stack
        raise CapabilityError(
            f"cannot import torch from {sys.executable}: {exc!r}",
            _python_hints()
            + ("never pip-install or upgrade torch/torchvision over the vendor stack",),
        ) from exc
    raw_version = getattr(torch, "__version__", "") or "unknown"
    release = raw_version.split("+")[0]
    parts = release.split(".")
    try:
        minor = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        minor = MINIMUM_TORCH_MINOR
    if minor < MINIMUM_TORCH_MINOR:
        raise CapabilityError(
            f"torch {raw_version} is older than the supported 2.x stack",
            _python_hints(),
        )
    try:
        import torchvision
    except Exception as exc:  # pragma: no cover - depends on the host stack
        raise CapabilityError(
            f"cannot import torchvision: {exc!r}",
            _python_hints() + ("torchvision must match the installed torch build",),
        ) from exc
    return raw_version, getattr(torchvision, "__version__", "unknown")


def _check_sdaa() -> tuple[int, str]:
    try:
        import torch
        import torch_sdaa  # noqa: F401  (registers torch.sdaa)
    except Exception as exc:  # pragma: no cover - depends on the host stack
        raise CapabilityError(
            f"cannot import torch_sdaa: {exc!r}",
            _sdaa_hints(),
        ) from exc
    sdaa = getattr(torch, "sdaa", None)
    if sdaa is None:  # pragma: no cover - defensive
        raise CapabilityError("torch_sdaa imported but torch.sdaa is unavailable", _sdaa_hints())
    try:
        available = bool(sdaa.is_available())
        device_count = int(sdaa.device_count())
    except Exception as exc:  # pragma: no cover - depends on the host stack
        raise CapabilityError(f"SDAA device query failed: {exc!r}", _sdaa_hints()) from exc
    if not available or device_count < 1:
        raise CapabilityError(
            f"the SDAA stack reports no usable device (available={available}, count={device_count})",
            _sdaa_hints(),
        )
    return device_count, getattr(torch_sdaa, "__version__", "unknown")


def _check_tecoops(root: str, require_root: bool) -> dict:
    """Bind the MSDA operator to the *installed* wheel.

    Supported form: ``pip install tecoops-*.whl`` and let ``import tecoops``
    resolve from site-packages.  ``TECOOPS_API_ROOT`` remains an explicit
    development override for a source-tree package; when it is set, ``tecoops``
    must resolve underneath it.  A missing wheel, missing native objects or a
    missing entry point all fail closed -- there is no torch fallback, and no
    source-tree path is consulted by default.
    """
    root = root or os.environ.get("TECOOPS_API_ROOT", "")
    try:
        import tecoops
    except Exception as exc:
        raise CapabilityError(
            f"cannot import tecoops: {exc!r}",
            (
                "install the paired wheel for this interpreter:",
                "  pip install tecoops-<version>-cp3xx-cp3xx-linux_loongarch64.whl",
                "the wheel is built from Teco-Ops PR #40 (branch op/deformable-msda-forward);",
                "TECOOPS_API_ROOT is only a development override for a source-tree package",
            ),
        ) from exc
    tecoops_file = Path(tecoops.__file__).resolve()
    if root:
        expected = Path(root).resolve() / "tecoops" / "__init__.py"
        if not expected.is_file():
            raise CapabilityError(
                f"TECOOPS_API_ROOT={Path(root).resolve()} does not contain tecoops/__init__.py",
                ("point TECOOPS_API_ROOT at the parent directory that holds the paired tecoops package",),
            )
        try:
            tecoops_file.relative_to(Path(root).resolve())
        except ValueError:
            raise CapabilityError(
                f"tecoops resolved to {tecoops_file}, which is NOT under "
                f"TECOOPS_API_ROOT={Path(root).resolve()} (an older wheel is shadowing it)",
                (
                    "put TECOOPS_API_ROOT first on PYTHONPATH (the shell entries already do),",
                    "or remove the unrelated tecoops installation from site-packages",
                ),
            ) from None
    missing = [name for name in REQUIRED_TECOOPS_ATTRS if not hasattr(tecoops, name)]
    if missing:
        raise CapabilityError(
            f"tecoops at {tecoops_file} is missing {', '.join(missing)}",
            ("use the package built from the paired Teco-Ops PR #40 native forward and first-order backward",),
        )
    package_dir = tecoops_file.parent
    extensions = sorted(package_dir.glob("_torch_ext*.so"))
    if not extensions:
        raise CapabilityError(
            f"no _torch_ext*.so next to {tecoops_file}",
            ("the tecoops package is incomplete; rebuild and redeploy the paired native extension",),
        )
    cores = sorted(package_dir.glob("libteco_ops*.so"))
    return {
        "tecoops_file": str(tecoops_file),
        "tecoops_root": str(Path(root).resolve()) if root else "",
        "tecoops_source": "source-tree-override" if root else "installed-wheel",
        "extension": str(extensions[0]),
        "core": str(cores[0]) if cores else "",
    }


_RECEIPT_CACHE: dict[tuple[str, bool], dict] = {}


def require_capabilities(*, role: str, tecoops_root: str = "", require_tecoops_root: bool = False) -> dict:
    """Validate the runtime capabilities; raise :class:`CapabilityError` on failure.

    The expensive checks run once per process; later callers (for example
    ``verify_native_voc.py`` importing ``train_sdaa``) re-report the same receipt
    under their own role instead of re-importing the stack.
    """
    key = (tecoops_root or os.environ.get("TECOOPS_API_ROOT", ""), require_tecoops_root)
    cached = _RECEIPT_CACHE.get(key)
    if cached is not None:
        receipt = dict(cached, role=role)
        _report_success(receipt)
        return receipt
    try:
        receipt = {
            "role": role,
            "python_version": _check_python(),
            # Keep the interpreter exactly as invoked: resolving symlinks would strip a
            # virtualenv prefix (/home/py311/bin/python -> /usr/bin/python3.11), and the
            # shell entries re-exec this path, which then loses the venv site-packages
            # and fails with ModuleNotFoundError: torch_sdaa.
            "executable": sys.executable,
        }
        receipt["torch"], receipt["torchvision"] = _check_torch()
        receipt["sdaa_devices"], receipt["torch_sdaa"] = _check_sdaa()
        receipt.update(_check_tecoops(tecoops_root, require_tecoops_root))
    except CapabilityError as exc:
        _report_failure(role, exc)
        raise
    _RECEIPT_CACHE[key] = dict(receipt)
    _report_success(receipt)
    return receipt


def _report_failure(role: str, exc: CapabilityError) -> None:
    print(f"[runtime_contract] FAIL role={role}", file=sys.stderr)
    print(f"[runtime_contract] reason: {exc.reason}", file=sys.stderr)
    for hint in exc.hints:
        print(f"[runtime_contract] hint: {hint}", file=sys.stderr)


def _report_success(receipt: dict) -> None:
    print(
        "[runtime_contract] OK "
        f"role={receipt['role']} python={receipt['python_version']} "
        f"torch={receipt['torch']} torchvision={receipt['torchvision']} "
        f"sdaa_devices={receipt['sdaa_devices']} tecoops={receipt['tecoops_file']} "
        f"root={receipt['tecoops_root'] or 'unset'}",
        file=sys.stderr,
    )
    print(f"[runtime_contract] tecoops.__file__={receipt['tecoops_file']}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="run the capability contract")
    parser.add_argument("--tecoops-root", default="", help="expected parent directory of the paired tecoops package")
    parser.add_argument("--require-tecoops-root", action="store_true",
                        help="when TECOOPS_API_ROOT is set, require tecoops to resolve under it (the installed wheel is the default and needs no root)")
    parser.add_argument("--role", default="runtime_contract.py", help="calling entry point, for messages")
    parser.add_argument("--json", action="store_true", help="print the receipt as JSON on success")
    arguments = parser.parse_args(argv)
    if not arguments.check:
        parser.error("nothing to do: pass --check")
    try:
        receipt = require_capabilities(
            role=arguments.role,
            tecoops_root=arguments.tecoops_root,
            require_tecoops_root=arguments.require_tecoops_root,
        )
    except CapabilityError:
        return 1
    if arguments.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

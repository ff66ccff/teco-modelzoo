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
# Adapted to tecorigin hardware
# Copyright (c) 2026 Sichuan University contributors.
"""Single import-time binding to the installed ``tecoops`` wheel.

The model consumes the custom multi-scale deformable attention operator through
the **installed wheel only**:

    pip install tecoops-<version>-cp3xx-cp3xx-linux_loongarch64.whl
    import tecoops                      # resolved from site-packages

Nothing in this module reads an environment variable, resolves a source-tree
path, or falls back to a ``torch``/CUDA implementation: if the wheel is missing or
does not expose the paired native entry, importing this module fails closed with
an actionable message.  Binding happens once, at import time, so the hot path has
no branch and no ``getenv`` (AGENTS.md rule 6).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

#: The paired native surface the model needs (forward + first-order backward).
REQUIRED_TECOOPS_ATTRS = ("ms_deform_attn", "ms_deform_attn_forward", "ms_deform_attn_backward")

_INSTALL_HINT = (
    "install the paired wheel for this interpreter:",
    "  pip install tecoops-<version>-cp3xx-cp3xx-linux_loongarch64.whl",
    "the wheel is built from Teco-Ops PR #40 (branch op/deformable-msda-forward);",
    "there is deliberately no fallback to a torch or local CUDA implementation",
)


def _fail(reason: str) -> None:
    raise ImportError("[deformable-detr] " + reason + "\n  " + "\n  ".join(_INSTALL_HINT))


try:
    import tecoops  # noqa: F401  (fail-closed: the wheel must be installed)
except ImportError as exc:  # pragma: no cover - depends on the host environment
    _fail("cannot import tecoops ({!r}).".format(exc))

_missing = [name for name in REQUIRED_TECOOPS_ATTRS if not hasattr(tecoops, name)]
if _missing:
    _fail(
        "tecoops at {} is missing {}.".format(
            getattr(tecoops, "__file__", "<unknown>"), ", ".join(_missing)
        )
    )

_PACKAGE_DIR = Path(tecoops.__file__).resolve().parent
_NATIVE_OBJECTS = sorted(p.name for p in _PACKAGE_DIR.glob("*.so"))
if not _NATIVE_OBJECTS:
    _fail("tecoops at {} ships no native library (*.so).".format(_PACKAGE_DIR))

#: The paired entry, bound once at import time.
ms_deform_attn = tecoops.ms_deform_attn


def native_core_sha256() -> dict:
    """Provenance receipt for the native objects this binding actually loaded."""
    return {
        name: hashlib.sha256((_PACKAGE_DIR / name).read_bytes()).hexdigest()
        for name in _NATIVE_OBJECTS
    }


#: Import-time binding receipt (path + native object hashes); printed by the
#: reference entry so a run can self-certify which wheel it used.
BINDING = {
    "tecoops_file": str(Path(tecoops.__file__).resolve()),
    "package_dir": str(_PACKAGE_DIR),
    "native_objects": _NATIVE_OBJECTS,
    "native_core_sha256": native_core_sha256(),
    "required_attrs": list(REQUIRED_TECOOPS_ATTRS),
    "source": "installed-wheel",
}

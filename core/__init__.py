"""
core — shared library for DD-RUO (TM/DM/DC + quantize + cross_eval).

Provides the TensorPool-based image synthesis / compression backbone
(`core.ts`) plus shared `utils` and `networks`.

Backward-compatibility shim
---------------------------
Before the repo reorganization, the compression backbone lived at the
top-level package `ts/` (e.g. `ts.tensor_data_func_v6`). Checkpoints saved
under that layout pickle class references as `ts.<mod>.<Class>`. After the
move to `core/ts/`, unpickling those checkpoints raises
`ModuleNotFoundError: No module named 'ts'`.

To load legacy checkpoints unchanged, we register `ts` and its submodules
as aliases pointing at `core.ts`'s submodule tree. New checkpoints pickle
references as `core.ts.<mod>.<Class>` and load natively. Both directions
work with a single copy of the code (no duplication).

If a future legacy checkpoint references a `ts.*` path not covered here,
the unpickler error message will name it — extend the alias map below.
"""

import importlib
import sys
import pkgutil


def _register_ts_aliases() -> None:
    # Avoid re-running: only register once, and only when core.ts is importable.
    if "ts" in sys.modules and getattr(sys.modules["ts"], "__ddruo_alias__", False):
        return
    try:
        from core import ts as ts_pkg
    except Exception:
        return

    # Map the top-level package.
    sys.modules["ts"] = ts_pkg
    setattr(sys.modules["ts"], "__ddruo_alias__", True)

    prefix = "core.ts."
    # Mirror every already-loaded core.ts.* submodule under the ts. namespace.
    for modname in list(sys.modules):
        if modname.startswith(prefix):
            alias = "ts." + modname[len(prefix):]
            sys.modules.setdefault(alias, sys.modules[modname])

    # Proactively import + alias every submodule on disk, so references to
    # not-yet-imported modules (e.g. ts.core.quantizer) resolve at unpickle time.
    for _finder, submod_name, _ispkg in pkgutil.walk_packages(
        ts_pkg.__path__, prefix="core.ts."
    ):
        try:
            mod = importlib.import_module(submod_name)
        except Exception:
            continue
        alias = "ts." + submod_name[len(prefix):]
        sys.modules.setdefault(alias, mod)


_register_ts_aliases()

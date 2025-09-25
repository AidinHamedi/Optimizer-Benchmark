import importlib
import pkgutil
from typing import Any, Dict, Tuple

DEBUG = False
PACKAGE_NAME = __name__
IGNORE_MODULES = {"__init__", "norm"}
IGNORE_FUNCTIONS = {"normalize"}


def load_functions() -> Dict[str, Dict[str, Any]]:
    func_dict: Dict[str, Dict[str, Any]] = {}

    package = importlib.import_module(PACKAGE_NAME)

    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        if module_name in IGNORE_MODULES:
            if DEBUG:
                print(f"⚠️  Skipping module: {module_name}")
            continue

        full_module_name = f"{PACKAGE_NAME}.{module_name}"
        module = importlib.import_module(full_module_name)

        if not hasattr(module, "FUNCTION_NAME"):
            if DEBUG:
                print(f"⚠️  {full_module_name} has no FUNCTION_NAME → skipped")
            continue

        name = getattr(module, "FUNCTION_NAME")

        candidates = [
            (fname, fval)
            for fname, fval in module.__dict__.items()
            if (
                callable(fval)
                and not fname.startswith("_")
                and fname not in IGNORE_FUNCTIONS
            )  # Exclude ignored function names
        ]

        if not candidates:
            if DEBUG:
                print(f"⚠️  {full_module_name} has no public functions → skipped")
            continue

        func = None
        for fname, fval in candidates:
            if (
                fname.lower() in module_name.lower()
                or fname.lower() in name.lower().replace("-", "").replace(" ", "")
            ):
                func = fval
                break
        if func is None:
            func = candidates[0][1]

        func_dict[name] = {
            "func": func,
            "size": getattr(module, "EVAL_SIZE", None),
            "pos": getattr(module, "START_POS", None),
            "gm_pos": getattr(module, "GLOBAL_MINIMUM_LOC", None),
        }

        if DEBUG:
            print(f"✅ Loaded: {name}")
            print(f"   ├─ Module : {full_module_name}")
            print(f"   ├─ Func   : {func.__name__}()")
            print(f"   ├─ Size   : {func_dict[name]['size']}")
            print(f"   ├─ Start  : {func_dict[name]['pos']}")
            print(f"   └─ GM Pos : {func_dict[name]['gm_pos']}\n")

    return func_dict


FUNC_DICT = load_functions()


def scale_eval_size(
    eval_size: Tuple[Tuple[float, float], Tuple[float, float]], scale: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Scale the evaluation size by a given factor."""

    def adjust_pair(first: float, second: float, factor: float) -> Tuple[float, float]:
        if first == -second:
            return first * factor, second * factor

        sign = 1 if first >= 0 else -1
        abs_first = abs(first)
        abs_second_scaled = abs(second * factor)

        if factor > 1:
            new_mag = abs_first / factor
        else:
            new_mag = min(abs_first / factor, abs_second_scaled)

        return sign * new_mag, second * factor

    return (
        adjust_pair(eval_size[0][0], eval_size[0][1], scale),
        adjust_pair(eval_size[1][0], eval_size[1][1], scale),
    )

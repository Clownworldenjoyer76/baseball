#!/usr/bin/env python3
"""Run the shared MLB selector with the actual morning configuration.

Importing the shared selector as a module is intentional: function global lookups
must refer to the same namespace where morning settings are assigned. Updating a
runpy.run_path() result dictionary does NOT update those function globals.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import yaml

SELECTOR_PATH = Path(__file__).with_name("baseball_select_bets.py")
OUTPUT_DIR = Path("docs/win/baseball/mlb/04_select/morning")
CONFIG_PATH = Path("docs/win/baseball/mlb/config/markets_AM.yaml")
ERROR_DIR = Path("docs/win/baseball/mlb/errors/04_select")


def load_morning_selector():
    spec = spec_from_file_location("mlb_morning_selector_shared", SELECTOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import MLB selector: {SELECTOR_PATH}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)["markets"]["mlb"]

    module.OUTPUT_DIR = OUTPUT_DIR
    module.CONFIG_PATH = CONFIG_PATH
    module.AUDIT_DIR = OUTPUT_DIR / "audit"
    module.LOCKED_DIR = OUTPUT_DIR / "locked"
    module.ERROR_DIR = ERROR_DIR
    module.LOG_FILE = ERROR_DIR / "select_bets_AM.txt"
    module.CONFIG = config
    module.FILTERS = config

    for directory in (module.OUTPUT_DIR, module.AUDIT_DIR,
                      module.LOCKED_DIR, module.ERROR_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    return module


if __name__ == "__main__":
    load_morning_selector().main()

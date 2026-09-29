"""Where the benchmark and demo scripts save their plots and flight logs.

The scripts in benchmarks/ and demos/ write into outputs/ at the repository
root, which git ignores. The folder is found from the calling script's own
file, not from the current working directory, so the result does not depend on
where the script is started from, and a copy of the scripts (as the tests run
them) writes into its own outputs/.
"""

from pathlib import Path


def output_file(script_file, name):
    """Return the path of ``name`` in outputs/, creating the folder if missing.

    ``script_file`` is the calling script's ``__file__``. The scripts sit one
    folder below the repository root (benchmarks/ or demos/), so outputs/ is
    next to that folder. The path is returned as a string: save_flight_log
    derives its second file name from it with str.replace.
    """
    folder = Path(script_file).resolve().parents[1] / 'outputs'
    folder.mkdir(exist_ok=True)
    return str(folder / name)

"""CPU checks of the setting A job list: every job names each option once (argparse keeps only the last copy of a
repeated option, which silently drops inputs), and the selections keep their methods.

Run from the project root: CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:experiments torch-python tests/test_setting_a.py
"""

import setting_a


def test_every_option_appears_once() -> None:
    """No job of any stage repeats an option."""
    for jobs in (setting_a.plant_jobs("root"), setting_a.method_jobs("root"), setting_a.evaluate_jobs("root")):
        for marker, argv in jobs:
            options = [part for part in argv if part.startswith("--")]
            assert len(options) == len(set(options)), f"{marker}: repeated option in {options}"


def test_recovery_reads_every_ranking() -> None:
    """The recovery job of a plant reads the one-shot and the sufficiency rankings and every learned run."""
    recovery = next(argv for marker, argv in setting_a.evaluate_jobs("root") if "recovery_all" in marker)
    scores = recovery[recovery.index("--scores") + 1:]
    assert scores[0].startswith("stats=") and scores[1].startswith("sufficiency=")
    runs = recovery[recovery.index("--runs") + 1:]
    assert {entry.split("=")[0] for entry in runs if "=" in entry} >= set(setting_a.METHODS)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")

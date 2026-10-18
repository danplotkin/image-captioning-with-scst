import copy
import importlib.util
from pathlib import Path

import pytest


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reporting = load_script("report_final_models")
evaluation = load_script("evaluate_final_models")


def test_continuation_history_does_not_double_count_or_hide_changes():
    original = [{"epoch": 1, "validation_reward": 0.4}, {"epoch": 2, "validation_reward": 0.5}]
    continued = [*original, {"epoch": 3, "validation_reward": 0.6}]
    assert reporting.continued_history(original, continued) == continued
    changed = copy.deepcopy(continued)
    changed[0]["validation_reward"] = 0.3
    with pytest.raises(ValueError, match="prefix"):
        reporting.continued_history(original, changed)
    with pytest.raises(ValueError, match="epochs"):
        reporting.continued_history(original, [*continued, continued[-1]])


def test_examples_are_reproducible_and_independent_of_input_order():
    ids = [f"{i}.jpg" for i in range(1000)]
    selected = reporting.select_examples(ids)
    assert len(selected) == len(set(selected)) == 12
    assert selected == reporting.select_examples(list(reversed(ids)))
    assert set(selected) <= set(ids)


@pytest.mark.parametrize("failure", ["image_ids", "checkpoint", "precision", "references", "cache"])
def test_final_evaluation_rejects_incomparable_cells(failure):
    payload = {
        "provenance": {
            "context": {
                "checkpoint_sha256": "abc",
                "decoding": "beam",
                "inference_precision": "fp32",
                "use_kv_cache": True,
            },
            "num_images": 2,
            "reference_counts": [5],
        }
    }
    predictions = [{"image_id": "a"}, {"image_id": "b"}]
    evaluation.validate_cell(payload, predictions, ["a", "b"], "abc", "beam", False)
    if failure == "image_ids":
        predictions.reverse()
    elif failure == "checkpoint":
        payload["provenance"]["context"]["checkpoint_sha256"] = "other"
    elif failure == "precision":
        payload["provenance"]["context"]["inference_precision"] = "cuda_amp_fp16"
    elif failure == "references":
        payload["provenance"]["reference_counts"] = [1]
    else:
        payload["provenance"]["context"]["use_kv_cache"] = False
    with pytest.raises(ValueError):
        evaluation.validate_cell(payload, predictions, ["a", "b"], "abc", "beam", False)

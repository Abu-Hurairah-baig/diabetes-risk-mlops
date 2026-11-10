import json

PASS_MARK = 0.75  # minimum accuracy allowed


def test_accuracy_is_above_pass_mark():
    with open("metrics.json") as f:
        acc = json.load(f)["accuracy"]
    assert acc >= PASS_MARK, f"Accuracy {acc:.2f} is below the pass mark {PASS_MARK}"
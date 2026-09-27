"""The demo may recommend retraining without silently launching a cloud deployment."""

from pathlib import Path

import yaml


def test_monitor_dispatch_is_opt_in_and_failures_are_not_hidden() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / ".github/workflows/model-monitoring-drift-monitor.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    # PyYAML's YAML 1.1 loader treats the Actions key `on` as True.
    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    assert inputs["dispatch_retraining"]["type"] == "boolean"
    assert inputs["dispatch_retraining"]["default"] is False
    steps = workflow["jobs"]["monitor"]["steps"]
    dispatch = next(step for step in steps if step.get("name") == "Trigger retraining pipeline")
    assert "inputs.dispatch_retraining" in dispatch["if"]
    assert "steps.run.outputs.code == '3'" in dispatch["if"]
    assert "TRAINING_WORKFLOW" in dispatch["env"]
    assert "sagemaker-byoc-deploy-cd.yml" in dispatch["env"]["TRAINING_WORKFLOW"]
    assert (path.parent / "sagemaker-byoc-deploy-cd.yml").is_file()
    assert "-f deploy_production=false" in dispatch["run"]
    assert "||" not in dispatch["run"]
    assert "continue-on-error" not in dispatch
    report_only = next(
        step
        for step in steps
        if step.get("name") == "Record retraining recommendation without dispatch"
    )
    assert "!inputs.dispatch_retraining" in report_only["if"]
    assert "GITHUB_STEP_SUMMARY" in report_only["run"]

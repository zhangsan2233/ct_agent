import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chestct_agent.evaluation import patient_id_from_case_id


def main() -> None:
    training = pd.read_csv("artifacts/prepared/case_index.csv").fillna("")
    manifest = pd.read_csv("artifacts/evaluation/multimodal_manifest.csv").fillna("")
    splits = pd.read_csv("artifacts/evaluation/patient_splits.csv").fillna("")
    train_cases = set(training["case_id"].astype(str))
    valid_cases = set(manifest["case_id"].astype(str))
    train_patients = {patient_id_from_case_id(case) for case in train_cases}
    valid_patients = {patient_id_from_case_id(case) for case in valid_cases}
    calibration = set(
        splits.loc[splits["evaluation_split"].eq("calibration"), "patient_id"].astype(str)
    )
    test = set(splits.loc[splits["evaluation_split"].eq("test"), "patient_id"].astype(str))
    payload = {
        "training_cases": len(train_cases),
        "local_validation_cases": len(valid_cases),
        "training_case_prefixes": sorted({case.split("_", 1)[0] for case in train_cases}),
        "validation_case_prefixes": sorted({case.split("_", 1)[0] for case in valid_cases}),
        "train_validation_case_overlap": len(train_cases & valid_cases),
        "train_validation_patient_overlap": len(train_patients & valid_patients),
        "calibration_patients": len(calibration),
        "test_patients": len(test),
        "calibration_test_patient_overlap": len(calibration & test),
        "passed": not (train_cases & valid_cases)
        and not (train_patients & valid_patients)
        and not (calibration & test),
    }
    output = Path("artifacts/evaluation/split_leakage_audit.json")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit("Split leakage audit failed.")


if __name__ == "__main__":
    main()

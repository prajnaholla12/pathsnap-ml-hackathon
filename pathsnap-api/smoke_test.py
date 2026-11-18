"""API integration checks using only student_test from the supplied student ZIP.

Run from repo root: python pathsnap-api/smoke_test.py --student-zip "Student folder.zip"
Requires requirements-test.txt. Does not train or overwrite experiment results.
"""
import argparse
import json
import re
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fastapi.testclient import TestClient
from threadpoolctl import threadpool_limits

import main


def run(student_zip):
    checks, results = [], []
    with tempfile.TemporaryDirectory(prefix="pathsnap-student-smoke-") as tmp:
        test_dir = Path(tmp)
        with ZipFile(student_zip) as archive:
            for member in archive.namelist():
                match = re.fullmatch(r"[^/]+/student_test/([0-9]+)/\1\.(track|nodes|arcs|route)", member)
                if match:
                    trip_id, suffix = match.groups()
                    folder = test_dir / trip_id
                    folder.mkdir(exist_ok=True)
                    (folder / f"{trip_id}.{suffix}").write_bytes(archive.read(member))
        ids = sorted(p.name for p in test_dir.iterdir())
        assert len(ids) == 10, f"Expected 10 student-test trips, found {len(ids)}"
        main.EVAL_DIR = test_dir
        with TestClient(main.app) as client:
            health = client.get("/health")
            assert health.status_code == 200 and health.json()["trips"] == 10
            listing = client.get("/trips")
            assert listing.status_code == 200
            assert [t["trip_id"] for t in listing.json()["trips"]] == ids
            assert client.post("/predict", json={"trip_id": "../unknown"}).status_code == 404
            assert client.post("/predict", json={}).status_code == 422
            checks.extend(["health", "trip listing", "unknown ID rejected", "missing ID rejected"])
            with np.load(main.ROOT / "pathsnap-hmm/results/student_test_hmm_predictions.npz") as recorded:
                for trip_id in ids:
                    response = client.post("/predict", json={"trip_id": trip_id})
                    assert response.status_code == 200, response.text
                    value = response.json()
                    json.dumps(value, allow_nan=False)
                    assert value["n_points"] == len(value["gps"]) == len(value["predicted_arcs_per_point"])
                    assert value["predicted_route"] and value["metrics"]["has_reference"]
                    assert not any(k in value for k in ("reference_route", "ground_truth", "labels"))
                    assert 0 <= value["metrics"]["hmm_top1_accuracy"] <= 1
                    differences = int(np.count_nonzero(np.asarray(value["predicted_arcs_per_point"]) != recorded[trip_id]))
                    results.append({"trip_id": trip_id, "n_points": value["n_points"],
                                    "hmm_top1_accuracy": value["metrics"]["hmm_top1_accuracy"],
                                    "differences_from_supplied_predictions": differences})
                    print(f"PASS {trip_id}: {value['n_points']} points; saved prediction differences={differences}", flush=True)
            checks.append("all 10 student-test trips return valid prediction JSON")
            trip_id = "00000052"
            before = client.post("/predict", json={"trip_id": trip_id}).json()
            # Delete only a temporary reference copy, never the original dataset.
            (test_dir / trip_id / f"{trip_id}.route").unlink()
            response = client.post("/predict", json={"trip_id": trip_id})
            assert response.status_code == 200, response.text
            after = response.json()
            json.dumps(after, allow_nan=False)
            assert not after["metrics"]["has_reference"]
            assert after["metrics"]["hmm_top1_accuracy"] is None
            assert before["predicted_arcs_per_point"] == after["predicted_arcs_per_point"]
            assert before["predicted_route"] == after["predicted_route"]
            checks.append("missing reference returns null scores and identical prediction")
            (test_dir / trip_id / f"{trip_id}.arcs").write_text("999999999 0\n", encoding="utf-8")
            assert client.post("/predict", json={"trip_id": trip_id}).status_code == 422
            checks.append("invalid network rejected")
    return {"partition": "student_test", "checks": checks, "trips": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-zip", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        result = run(args.student_zip)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(result['checks'])} checks, 10 student-test trips", flush=True)

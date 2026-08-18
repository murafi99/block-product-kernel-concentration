import json
import pathlib
import sys
import textwrap

import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from theory import c_b


def test_c_b_table_shape_and_values():
    t = c_b.table(6)
    assert list(t.keys()) == [1, 2, 3, 4, 5, 6]
    for b, v in t.items():
        assert v == pytest.approx(c_b.FROZEN_CONTRACT_VALUES[b], abs=1e-9)


def test_exponent_sweep_writes_well_formed_table(tmp_path):
    from experiments import exponent_sweep

    seeds_path = tmp_path / "seeds.yaml"
    seeds_path.write_text("master_seed: 1\n")

    cfg = dict(
        experiment="exponent_sweep",
        block_sizes=[2],
        depth=3,
        n_pairs=40,
        seeds_per_block={2: 1},
        domain=[-3.14159, 3.14159],
        purity_check_samples=30,
        run_regression_crosscheck=False,
        regression_block_repeats=[1, 2],
        output_json=str(tmp_path / "expA.json"),
        output_table=str(tmp_path / "table_A.md"),
        b1_structural=dict(ry_only_depth=1, n_pairs=40, digitizing_bins=2),
    )
    cfg_path = tmp_path / "exponent_sweep.yaml"
    cfg_path.write_text(yaml.dump(cfg))

    results = exponent_sweep.main(str(cfg_path))

    assert "b1_structural" in results
    assert "b=2" in results
    table_text = pathlib.Path(cfg["output_table"]).read_text()
    lines = [l for l in table_text.splitlines() if l.strip()]
    assert lines[0].startswith("| b | c(b) theory")
    # header + separator + b=1 row + b=2 row = 4 lines
    assert len(lines) == 4

    j = json.loads(pathlib.Path(cfg["output_json"]).read_text())
    assert j["b=2"]["b"] == 2
    assert j["b=2"]["c_theory"] == pytest.approx(c_b.c_of_b(2))


def test_stress_test_table_lists_all_ensembles(tmp_path):
    from experiments import construction_stress_test as cst

    cfg = dict(
        experiment="construction_stress_test",
        qubit_dim=2,
        n_pairs=500,
        ensembles=["digitizing", "sic_povm"],
        output_json=str(tmp_path / "stress.json"),
        output_table=str(tmp_path / "stress.md"),
    )
    cfg_path = tmp_path / "stress_test.yaml"
    cfg_path.write_text(yaml.dump(cfg))

    results = cst.main(str(cfg_path))
    assert results["_summary"]["all_within_nakata_interval"] is True

    table_text = pathlib.Path(cfg["output_table"]).read_text()
    assert "digitizing" in table_text
    assert "sic_povm" in table_text
    rows = [l for l in table_text.splitlines() if l.startswith("| digitizing") or l.startswith("| sic_povm")]
    assert len(rows) == 2


def test_interpolation_table_has_both_constructions(tmp_path):
    from experiments import interpolation

    out_json = tmp_path / "expD.json"
    out_table = tmp_path / "table_D.md"
    interpolation.main(b=2, depth=3, n_pairs=40, p_values=(0.0, 1.0),
                        out_json=str(out_json), out_table=str(out_table))
    text = out_table.read_text()
    assert "C1 exact" in text
    assert "C2 exact" in text
    j = json.loads(out_json.read_text())
    assert len(j["construction1"]) == 2
    assert len(j["construction2_analytic_only"]) == 2
    # endpoints must match the closed forms exactly (no simulation noise there)
    assert j["construction2_analytic_only"][0]["c_exact"] == pytest.approx(c_b.c_of_b(2), abs=1e-9)
    assert j["construction2_analytic_only"][1]["c_exact"] == pytest.approx(1.0, abs=1e-9)

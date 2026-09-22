# SPDX-FileCopyrightText: : 2017-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: CC0-1.0

"""
Tests for scripts/prepare_sclopf.py's outage/non-outage line splitting.

Run with: pytest test/test_prepare_sclopf.py
(uses the pypsa-eur pixi env, e.g. via
 `pixi run --manifest-path submodules/pypsa-eur/pixi.toml pytest test`)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import prepare_sclopf as ps  # noqa: E402


def _make_network(lines):
    """Build a minimal pypsa.Network with one Line per (s_nom, num_parallel) pair
    in `lines`, each between its own pair of buses."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2013-01-01", periods=2, freq="h"))
    for i, (s_nom, num_parallel) in enumerate(lines):
        b0, b1 = f"bus{i}_0", f"bus{i}_1"
        n.add("Bus", [b0, b1])
        n.add(
            "Line",
            f"L{i}",
            bus0=b0,
            bus1=b1,
            x=0.1,
            r=0.01,
            s_nom=s_nom,
            num_parallel=num_parallel,
        )
    return n


class TestGetPartition:
    def test_single_voltage_classes_decompose_exactly(self):
        reachable, w_int = ps._build_reachability(ps.TYPE_WEIGHTS, max_target=10)
        for weight in ps.TYPE_WEIGHTS:
            counts = ps.get_partition(weight, reachable, w_int)
            assert (np.array(counts) * ps.TYPE_WEIGHTS).sum() == pytest.approx(weight, abs=1e-6)
            assert counts.sum() == 1

    def test_combination_of_two_voltages_decomposes(self):
        # 2x220kV + 1x300kV
        target = 2 * ps.TYPE_WEIGHTS[0] + ps.TYPE_WEIGHTS[1]
        reachable, w_int = ps._build_reachability(ps.TYPE_WEIGHTS, max_target=10)
        counts = ps.get_partition(target, reachable, w_int)
        assert (np.array(counts) * ps.TYPE_WEIGHTS).sum() == pytest.approx(target, abs=1e-6)

    def test_floating_point_noise_within_tolerance_still_decomposes(self):
        # e.g. 2x380kV picked up ~2.8e-6 of clustering noise, as seen in practice
        target = 2 * 1.0 + 2.8e-6
        reachable, w_int = ps._build_reachability(ps.TYPE_WEIGHTS, max_target=10)
        counts = ps.get_partition(target, reachable, w_int)
        assert counts.sum() == 2

    def test_non_decomposable_value_raises(self):
        reachable, w_int = ps._build_reachability(ps.TYPE_WEIGHTS, max_target=10)
        with pytest.raises(ValueError):
            # smaller than the smallest TYPE_WEIGHTS entry - can't be any combination
            ps.get_partition(0.15, reachable, w_int)

    def test_htls_project_lines_decompose(self):
        # regression test: two NEP "HTLS 4-bundle 380.0" project lines merged by
        # cluster_network's plain num_parallel sum, as seen in practice (was raising
        # "No valid partition found for the given total weight 6.201545140553104"
        # before HTLS was added to _VOLTAGE_TYPE)
        reachable, w_int = ps._build_reachability(ps.TYPE_WEIGHTS, max_target=10)
        counts = ps.get_partition(6.201545140553104, reachable, w_int)
        # accumulated rounding across 4 circuits (see _TOL's comment) - not 1e-6
        assert (np.array(counts) * ps.TYPE_WEIGHTS).sum() == pytest.approx(
            6.201545140553104, abs=1e-5
        )
        assert counts.sum() == 4


class TestAssertCapacityConserved:
    def test_passes_when_conserved(self):
        original = pd.DataFrame({"s_nom": [1000.0], "num_parallel": [1.171053]}, index=["L0"])
        split = pd.DataFrame(
            {
                "s_nom": [247.191011, 505.617978, 247.191011],
                "num_parallel": [0.289474, 0.592105, 0.289474],
            },
            index=["L0_outagetype0", "L0_outagetype1", "L0"],
        )
        ps._assert_capacity_conserved(original, split)  # should not raise

    def test_raises_when_capacity_inflated(self):
        original = pd.DataFrame({"s_nom": [1000.0], "num_parallel": [1.171053]}, index=["L0"])
        inflated = pd.DataFrame(
            {
                "s_nom": [247.19, 505.62, 1000.0],  # non-outage line kept full s_nom too
                "num_parallel": [0.289474, 0.592105, 1.171053],
            },
            index=["L0_outagetype0", "L0_outagetype1", "L0"],
        )
        with pytest.raises(AssertionError):
            ps._assert_capacity_conserved(original, inflated)


class TestSplitOutageLines:
    @pytest.mark.parametrize(
        "s_nom,num_parallel",
        [
            (1000.0, ps.TYPE_WEIGHTS[0]),  # single 220kV circuit
            (1000.0, 2 * ps.TYPE_WEIGHTS[0] + ps.TYPE_WEIGHTS[1]),  # 2x220kV + 1x300kV
            (1000.0, ps.TYPE_WEIGHTS[2]),  # single 380kV circuit
            (1000.0, ps.TYPE_WEIGHTS[0] + ps.TYPE_WEIGHTS[2]),  # 1x220kV + 1x380kV
        ],
    )
    def test_conserves_total_capacity(self, s_nom, num_parallel):
        n = _make_network([(s_nom, num_parallel)])
        original = n.lines[["s_nom", "num_parallel"]].copy()

        ps.fix_capacities(n, overdim_extendables=1.0)
        n = ps.split_outage_lines(n)

        base_name = n.lines.index.to_series().str.replace(r"_outagetype\d+$", "", regex=True)
        totals = n.lines.groupby(base_name)[["s_nom", "num_parallel"]].sum()

        assert totals.loc["L0", "s_nom"] == pytest.approx(original.loc["L0", "s_nom"], abs=1e-6)
        assert totals.loc["L0", "num_parallel"] == pytest.approx(
            original.loc["L0", "num_parallel"], abs=1e-6
        )

    def test_conserves_capacity_across_multiple_corridors(self):
        n = _make_network(
            [
                (1000.0, ps.TYPE_WEIGHTS[0]),
                (2000.0, 2 * ps.TYPE_WEIGHTS[0] + ps.TYPE_WEIGHTS[1]),
                (500.0, 3 * ps.TYPE_WEIGHTS[1]),  # 3x300kV circuits
            ]
        )
        original = n.lines[["s_nom", "num_parallel"]].copy()

        ps.fix_capacities(n, overdim_extendables=1.0)
        n = ps.split_outage_lines(n)

        base_name = n.lines.index.to_series().str.replace(r"_outagetype\d+$", "", regex=True)
        totals = n.lines.groupby(base_name)[["s_nom", "num_parallel"]].sum()

        for name in original.index:
            assert totals.loc[name, "s_nom"] == pytest.approx(original.loc[name, "s_nom"], abs=1e-6)
            assert totals.loc[name, "num_parallel"] == pytest.approx(
                original.loc[name, "num_parallel"], abs=1e-6
            )

    def test_outage_line_index_alignment_preserved(self):
        # regression test: remove_linetype must keep the original line index,
        # otherwise lines_nonout['num_parallel'] silently becomes NaN.
        n = _make_network([(1000.0, 2 * ps.TYPE_WEIGHTS[0] + ps.TYPE_WEIGHTS[1])])
        ps.fix_capacities(n, overdim_extendables=1.0)
        n = ps.split_outage_lines(n)
        assert not n.lines[["s_nom", "num_parallel"]].isna().any().any()

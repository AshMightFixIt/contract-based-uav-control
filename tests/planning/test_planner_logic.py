"""The planner's logic bugs (W4-09, T2; K10).

- The replanner checked H-inf at wind - 1 m/s, so it approved H-inf at 16 m/s,
  above the 15 m/s its contract assumes. It now checks at the real wind.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

STATE = {'tracking_error': 0.5, 'position_est_error': 0.3, 'velocity_est_error': 0.1}


def replan(wind, state=STATE):
    from contract_uav_core.planning.horizon_planner import AdaptiveRePlanner, HorizonPlanner

    replanner = AdaptiveRePlanner(HorizonPlanner())
    return replanner.find_feasible_plan(dict(state), np.zeros(3), {'wind_speed': wind})


def test_hinf_is_planned_up_to_its_wind_limit():
    plan, mods = replan(15.0)
    assert plan.is_safe() and plan.recommended_controller == 'hinf'
    assert plan.horizon_length == 7          # shorter horizons keep the margin at 15 m/s
    assert mods == {'emergency_mode': False}


@pytest.mark.parametrize('wind', [15.5, 16.0])
def test_hinf_is_not_certified_above_its_wind_limit(wind):
    """Before W4-09: FEASIBLE H-inf plan ('target_modified', checked at wind - 1 = 15 m/s)."""
    plan, mods = replan(wind)
    assert not plan.is_safe()
    assert plan.status.value == 'infeasible' and plan.recommended_controller == 'hinf'
    assert mods == {'emergency_mode': True}


def test_no_wind_discount_on_the_margin():
    """At 10 m/s and a 4.2 m error, H-inf's margin is 0.458 m < 0.5 even at the shortest horizon (3).

    Before W4-09 the replanner retried it at 9 m/s (margin 0.553 m) and accepted it.
    """
    plan, mods = replan(10.0, dict(STATE, tracking_error=4.2))
    assert not plan.is_safe() and mods == {'emergency_mode': True}
    plan, _ = replan(10.0, dict(STATE, tracking_error=4.0))   # margin 0.62 m at horizon 3: still planned
    assert plan.is_safe() and plan.recommended_controller == 'hinf' and plan.horizon_length == 3

"""
Per-step bookkeeping and flight-log export for AdaptiveDroneController (core.py).

_record_step holds steps 11-14 of control_step (runtime monitors, contract
monitoring, the flight-log entry) and save_flight_log writes the log; both are
moved from adaptive_control_system.py. The telemetry dict that control_step
returns is still built in core.py.

The flight log (W4-07, A-R4) is AdaptiveDroneController(flight_log_len=...):
- None (the default): a list with one entry per step, as before W4-07. It
  grows by about 2 KB per step (6 MB per simulated minute at 50 Hz), so the
  benchmark and demo scripts keep and save whole runs; a long run, or the ROS
  node, should pass a length.
- N > 0: a collections.deque(maxlen=N) of the newest N entries.
- 0: no flight log; _record_step does not build the entry.
"""

from collections import deque
import numbers

import numpy as np
from typing import Deque, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class TelemetryMixin:
    """Runtime and contract monitor updates, the flight log and its export."""

    @staticmethod
    def _new_flight_log(flight_log_len: Optional[int]) -> Union[List[Dict], Deque[Dict]]:
        """The empty flight log for AdaptiveDroneController(flight_log_len=...); see the module docstring."""
        if flight_log_len is None:
            return []
        if isinstance(flight_log_len, bool) or not isinstance(flight_log_len, numbers.Integral) \
                or flight_log_len < 0:
            raise ValueError(f"flight_log_len must be None or an integer >= 0, got {flight_log_len!r}")
        return deque(maxlen=int(flight_log_len))

    def _record_step(self,
                     raw_sensors: Dict[str, any],
                     state_dict: Dict[str, np.ndarray],
                     setpoint: Dict,
                     control: Dict[str, np.ndarray],
                     cbf_intervened: bool,
                     estimation_ok: bool,
                     planner_info: Dict) -> Tuple[str, str]:
        """Steps 11-14 of control_step. Returns (active_controller, flight_mode)."""
        # 11. Update runtime monitors
        self.runtime_monitor.update_wind_estimate(
            state_dict['velocity'],
            setpoint.get('velocity', np.zeros(3))
        )
        if 'gps' in raw_sensors:
            gps = raw_sensors['gps']
            self.runtime_monitor.update_gps_quality(
                gps.get('satellites', 0), gps.get('hdop', 100.0)
            )

        # 12. Contract monitoring
        active_controller = self.controller_switcher.get_active_controller()
        # Re-estimate conditions with actual control output
        system_conditions = self.estimate_system_conditions(state_dict, setpoint, control)
        self.contract_monitor.monitor_runtime(
            f"controller_{active_controller}",
            system_conditions,
            self.time
        )

        # 13. Actuator contract monitoring
        actuator_state = {
            'control_effort': control['thrust'],
            'battery_voltage': raw_sensors.get('battery', {}).get('voltage', 12.0),
            'motor_temperature': raw_sensors.get('motors', {}).get('temperature', 25.0),
        }
        self.contract_monitor.monitor_runtime('actuators', actuator_state, self.time)

        # 14. Log data (not with flight_log_len=0: a deque that keeps nothing)
        flight_mode = self.supervisor.get_mode().value
        if getattr(self.flight_log, 'maxlen', None) == 0:
            return active_controller, flight_mode
        log_entry = {
            'time': self.time,
            'state': state_dict.copy(),
            'control': control.copy(),
            'controller': active_controller,
            'flight_mode': flight_mode,
            'cbf_intervened': cbf_intervened,
            'system_conditions': system_conditions.copy(),
            'estimation_ok': estimation_ok,
            'planner_info': planner_info.copy() if planner_info else {}
        }
        self.flight_log.append(log_entry)

        return active_controller, flight_mode

    def save_flight_log(self, filename: str):
        """Save the flight log as JSON to filename, and the contract metrics next to it.

        The contract metrics (contract_monitor.export_metrics: its newest HISTORY_LEN
        metrics) go to filename with '_contracts' before a '.json' extension
        (flight.json -> flight_contracts.json), or with '_contracts.json' appended to
        any other name (flight -> flight_contracts.json), so they never overwrite the log.
        The arrays are converted to lists in copies: the flight log in memory keeps them.
        (W4-07: before, the conversion wrote into the log itself, and a name without
        '.json' was overwritten by the contract metrics.)
        """
        import json
        import os

        filename = os.fspath(filename)
        log_serializable = []
        for entry in self.flight_log:
            entry_copy = entry.copy()
            for key in ['state', 'control', 'system_conditions']:
                if key in entry_copy:
                    entry_copy[key] = {subkey: val.tolist() if isinstance(val, np.ndarray) else val
                                       for subkey, val in entry_copy[key].items()}
            log_serializable.append(entry_copy)

        with open(filename, 'w') as f:
            json.dump(log_serializable, f, indent=2)

        logger.info(f"Flight log saved to {filename}")

        root, extension = os.path.splitext(filename)
        contract_log = (root if extension.lower() == '.json' else filename) + '_contracts.json'
        self.contract_monitor.export_metrics(contract_log)

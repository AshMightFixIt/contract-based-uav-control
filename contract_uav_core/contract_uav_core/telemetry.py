"""
Per-step bookkeeping and flight-log export for AdaptiveDroneController (core.py).

_record_step holds steps 11-14 of control_step (runtime monitors, contract
monitoring, the flight-log entry) and save_flight_log writes the log; both are
moved unchanged from adaptive_control_system.py. The telemetry dict that
control_step returns is still built in core.py.
"""

import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TelemetryMixin:
    """Runtime and contract monitor updates, the flight log and its export."""

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

        # 14. Log data
        flight_mode = self.supervisor.get_mode().value
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
        """Save flight log for analysis."""
        import json

        log_serializable = []
        for entry in self.flight_log:
            entry_copy = entry.copy()
            for key in ['state', 'control', 'system_conditions']:
                if key in entry_copy:
                    for subkey, val in entry_copy[key].items():
                        if isinstance(val, np.ndarray):
                            entry_copy[key][subkey] = val.tolist()
            log_serializable.append(entry_copy)

        with open(filename, 'w') as f:
            json.dump(log_serializable, f, indent=2)

        logger.info(f"Flight log saved to {filename}")

        contract_log = filename.replace('.json', '_contracts.json')
        self.contract_monitor.export_metrics(contract_log)

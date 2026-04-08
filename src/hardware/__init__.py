"""
Hardware integration layer for DEXI-5 drone.

Provides real hardware interfaces to replace simulation:
    MSPInterface   — serial communication with Betaflight FC via MSP v1
    SensorReader   — converts FC telemetry to the sensor dict format
    ActuatorInterface — converts control output to Betaflight RC channels
"""

from .msp_interface import MSPInterface
from .sensor_reader import SensorReader
from .actuator_interface import ActuatorInterface
from . import hardware_config as config

__all__ = ['MSPInterface', 'SensorReader', 'ActuatorInterface', 'config']

"""
Airframe files: schema, validation and loader (W4-04, P0a.10).

The files are config/airframes/<name>.yaml, installed with the package. Nothing
in the legacy scripts or the control path reads them yet; the reference plant
(sim/quadrotor.py) can be built from one.

    from contract_uav_core.config.airframe import load_airframe
    x500 = load_airframe('x500_sitl')
    x500.hover_thrust                  # 0.6
    x500.provenance['hover_thrust']    # Param(value=0.6, unit='norm', status='px4_airframe', ...)

File layout: top-level keys schema_version (1), name, description,
px4 {version, airframe, airframe_url} and three sections of numbers:
vehicle, battery (PX4- or model-sourced) and plant (reference-plant parameters,
which have no PX4 source). Every number is an entry
{value, unit, status, source} with optional note and param (the PX4 parameter):

- status is one of STATUSES. px4_airframe and px4_param_default need param and
  a source URL; model_sdf needs a source URL; assumed is allowed, and required,
  in the plant section only; unknown means value null.
- value is null exactly when status is unknown, and only for fields the schema
  marks nullable. assumed and unknown need a note (why, and the effect).
- unit must equal the schema's unit; the value must have the schema's type and
  lie in its range. Unknown, duplicate and non-string keys are errors.

load_airframe returns a frozen Airframe; any violation raises AirframeError
naming the file and the key.
"""

import math
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple, Union

import yaml

SCHEMA_VERSION = 1
STATUSES = ('px4_airframe', 'px4_param_default', 'model_sdf', 'assumed', 'unknown')
_PX4_STATUSES = ('px4_airframe', 'px4_param_default')
_SOURCED_STATUSES = _PX4_STATUSES + ('model_sdf',)
_SOURCE_PREFIX = 'https://raw.githubusercontent.com/'

AIRFRAME_DIR = Path(__file__).resolve().parent / 'airframes'

# attribute -> (section, key in the file, unit, type, min, max, nullable)
FieldSpec = namedtuple('FieldSpec', 'section key unit kind lo hi nullable')
FIELDS = MappingProxyType({
    # vehicle: PX4 parameters and the Gazebo model
    'mass': FieldSpec('vehicle', 'mass', 'kg', float, 0.05, 50.0, True),
    'hover_thrust': FieldSpec('vehicle', 'hover_thrust', 'norm', float, 0.01, 1.0, False),
    'max_thrust': FieldSpec('vehicle', 'max_thrust', 'norm', float, 0.01, 1.0, False),
    'min_thrust': FieldSpec('vehicle', 'min_thrust', 'norm', float, 0.0, 1.0, False),
    'motor_constant': FieldSpec('vehicle', 'motor_constant', 'N s^2/rad^2', float, 1e-9, 1e-2, True),
    'max_rot_velocity': FieldSpec('vehicle', 'max_rot_velocity', 'rad/s', float, 1.0, 1e5, True),
    'moment_constant': FieldSpec('vehicle', 'moment_constant', 'm', float, 0.0, 1.0, True),
    # battery: PX4 BAT1_* parameters (voltages per cell)
    'battery_n_cells': FieldSpec('battery', 'n_cells', 'cells', int, 1, 16, False),
    'battery_v_empty_cell': FieldSpec('battery', 'v_empty_cell', 'V', float, 2.5, 4.5, False),
    'battery_v_charged_cell': FieldSpec('battery', 'v_charged_cell', 'V', float, 2.5, 4.5, False),
    'battery_capacity': FieldSpec('battery', 'capacity', 'mAh', float, 1.0, 100000.0, True),
    # plant: reference-plant parameters, always assumed
    'attitude_time_constant': FieldSpec('plant', 'attitude_time_constant', 's', float, 0.01, 2.0, False),
    'drag_coefficient': FieldSpec('plant', 'drag_coefficient', 'kg/m', float, 0.0, 10.0, False),
})
SECTIONS = ('vehicle', 'battery', 'plant')
_TOP_KEYS = ('schema_version', 'name', 'description', 'px4') + SECTIONS
_PX4_KEYS = ('version', 'airframe', 'airframe_url')
_ENTRY_REQUIRED = ('value', 'unit', 'status', 'source')
_ENTRY_OPTIONAL = ('note', 'param')


class AirframeError(ValueError):
    """An airframe file that does not match the schema."""


@dataclass(frozen=True)
class Param:
    """One number of an airframe file with its provenance."""
    value: Optional[Union[float, int]]
    unit: str
    status: str
    source: Optional[str]
    note: Optional[str] = None
    param: Optional[str] = None


@dataclass(frozen=True)
class Airframe:
    """A validated airframe. None marks an unknown value; provenance maps each field to its Param."""
    name: str
    description: str
    px4_version: str
    px4_airframe: str
    px4_airframe_url: str
    mass: Optional[float]
    hover_thrust: float
    max_thrust: float
    min_thrust: float
    motor_constant: Optional[float]
    max_rot_velocity: Optional[float]
    moment_constant: Optional[float]
    battery_n_cells: int
    battery_v_empty_cell: float
    battery_v_charged_cell: float
    battery_capacity: Optional[float]
    attitude_time_constant: float
    drag_coefficient: float
    provenance: Mapping[str, Param]


def available_airframes() -> Tuple[str, ...]:
    """Names of the packaged airframe files."""
    return tuple(sorted(p.stem for p in AIRFRAME_DIR.glob('*.yaml')))


def airframe_path(name: str) -> Path:
    """Path of the packaged airframe file <name>.yaml."""
    names = available_airframes()
    if name not in names:
        raise AirframeError(f"unknown airframe {name!r}; available: {', '.join(names) or 'none'}")
    return AIRFRAME_DIR / f'{name}.yaml'


def load_airframe(name: str) -> Airframe:
    """Load and validate a packaged airframe by name, e.g. 'x500_sitl' or 'dexi'."""
    airframe = load_airframe_file(airframe_path(name))
    if airframe.name != name:
        raise AirframeError(f"{name}.yaml: name: {airframe.name!r} does not match the file name")
    return airframe


def load_airframe_file(path: Union[str, Path]) -> Airframe:
    """Load and validate an airframe file from any path."""
    path = Path(path)
    try:
        with open(path, encoding='utf-8') as f:
            data = yaml.load(f, Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, AirframeError) as exc:
        raise AirframeError(f"{path.name}: not a valid airframe file: {exc}") from None
    return parse_airframe(data, origin=path.name)


def parse_airframe(data: Any, origin: str = '<data>') -> Airframe:
    """Validate a parsed airframe file (a mapping) and return the Airframe."""
    def fail(where, message):
        raise AirframeError(f"{origin}: {where}: {message}")

    _check_keys(data, _TOP_KEYS, (), 'top level', fail)
    if type(data['schema_version']) is not int or data['schema_version'] != SCHEMA_VERSION:
        fail('schema_version', f"expected {SCHEMA_VERSION}, got {data['schema_version']!r}")
    for key in ('name', 'description'):
        _check_text(data[key], key, fail)
    _check_keys(data['px4'], _PX4_KEYS, (), 'px4', fail)
    for key in _PX4_KEYS:
        _check_text(data['px4'][key], f'px4.{key}', fail)
    if not data['px4']['airframe_url'].startswith(_SOURCE_PREFIX):
        fail('px4.airframe_url', f"expected a URL starting with {_SOURCE_PREFIX}")

    values, provenance = {}, {}
    for section in SECTIONS:
        keys = tuple(spec.key for spec in FIELDS.values() if spec.section == section)
        _check_keys(data[section], keys, (), section, fail)
    for attr, spec in FIELDS.items():
        where = f'{spec.section}.{spec.key}'
        param = _parse_entry(data[spec.section][spec.key], spec, where, fail)
        values[attr] = param.value
        provenance[attr] = param

    if not values['min_thrust'] <= values['hover_thrust'] <= values['max_thrust']:
        fail('vehicle', f"need min_thrust <= hover_thrust <= max_thrust, got {values['min_thrust']}, "
                        f"{values['hover_thrust']}, {values['max_thrust']}")
    if not values['battery_v_empty_cell'] < values['battery_v_charged_cell']:
        fail('battery', f"need v_empty_cell < v_charged_cell, got {values['battery_v_empty_cell']} "
                        f"and {values['battery_v_charged_cell']}")

    return Airframe(
        name=data['name'], description=data['description'],
        px4_version=data['px4']['version'], px4_airframe=data['px4']['airframe'],
        px4_airframe_url=data['px4']['airframe_url'],
        provenance=MappingProxyType(provenance), **values)


def _parse_entry(entry, spec, where, fail) -> Param:
    _check_keys(entry, _ENTRY_REQUIRED, _ENTRY_OPTIONAL, where, fail)
    status, value, unit = entry['status'], entry['value'], entry['unit']
    source, note, param = entry['source'], entry.get('note'), entry.get('param')

    if status not in STATUSES:
        fail(f'{where}.status', f"{status!r} is not one of {', '.join(STATUSES)}")
    if spec.section == 'plant' and status != 'assumed':
        fail(f'{where}.status', "plant parameters have no PX4 source and must be 'assumed'")
    if spec.section != 'plant' and status == 'assumed':
        fail(f'{where}.status', "'assumed' is allowed only in the plant section")
    if unit != spec.unit:
        fail(f'{where}.unit', f"expected {spec.unit!r}, got {unit!r}")
    for key, text in (('note', note), ('param', param), ('source', source)):
        if text is not None and (not isinstance(text, str) or not text.strip()):
            fail(f'{where}.{key}', f"expected a non-empty string or null, got {text!r}")
    if status in ('assumed', 'unknown') and note is None:
        fail(f'{where}.note', f"a note is required when status is {status!r}")
    if status in _PX4_STATUSES and param is None:
        fail(f'{where}.param', f"the PX4 parameter name is required when status is {status!r}")
    if status in _SOURCED_STATUSES and (source is None or not source.startswith(_SOURCE_PREFIX)):
        fail(f'{where}.source', f"status {status!r} needs a source URL starting with {_SOURCE_PREFIX}")

    if value is None:
        if status != 'unknown':
            fail(f'{where}.value', f"null is allowed only with status 'unknown', not {status!r}")
        if not spec.nullable:
            fail(f'{where}.value', "this field may not be unknown (null)")
    else:
        if status == 'unknown':
            fail(f'{where}.value', f"status 'unknown' needs value null, got {value!r}")
        value = _check_number(value, spec, f'{where}.value', fail)
    return Param(value=value, unit=unit, status=status, source=source, note=note, param=param)


def _check_number(value, spec, where, fail):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(where, f"expected a number in {spec.unit}, got {type(value).__name__} {value!r}")
    if spec.kind is int and not isinstance(value, int):
        fail(where, f"expected an integer, got {value!r}")
    value = spec.kind(value)
    if not math.isfinite(value) or not spec.lo <= value <= spec.hi:
        fail(where, f"{value!r} {spec.unit} is outside [{spec.lo}, {spec.hi}]")
    return value


def _check_keys(mapping, required, optional, where, fail):
    if not isinstance(mapping, dict):
        fail(where, f"expected a mapping, got {type(mapping).__name__}")
    missing = [k for k in required if k not in mapping]
    unknown = [k for k in mapping if k not in required and k not in optional]
    if missing:
        fail(where, f"missing key(s): {', '.join(map(str, missing))}")
    if unknown:
        fail(where, f"unknown key(s): {', '.join(map(str, unknown))}")


def _check_text(value, where, fail):
    if not isinstance(value, str) or not value.strip():
        fail(where, f"expected a non-empty string, got {value!r}")


class _UniqueKeyLoader(yaml.SafeLoader):
    """yaml.SafeLoader that rejects duplicate and non-string keys (instead of keeping the last
    duplicate, or failing with a bare TypeError on an unhashable key)."""


def _construct_unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)  # deep: a list key is built in full
        if not isinstance(key, str):  # e.g. a number, null, or an unhashable list or mapping
            raise AirframeError(f"key {key!r} on line {key_node.start_mark.line + 1} is not a string")
        if key in seen:
            raise AirframeError(f"duplicate key {key!r} on line {key_node.start_mark.line + 1}")
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)

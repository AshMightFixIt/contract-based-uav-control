"""P1.3 (W4-05): the airframe fields added for the controllers and the contracts.

- battery.v_full_cell, status datasheet (no PX4 parameter holds the 4.2 V full
  charge of a LiPo cell), and the derived Airframe.battery_pack_voltage_range;
- the section unavailable: contract variables with no PX4 source
  (motor_temperature on both airframes);
- the schema stays additive: the batch-2 fields, names and order are unchanged;
- schema version 2 (fix round, review nit N3): a version-1 file fails with a
  message naming the missing keys and pointing to x500_sitl.yaml.
"""
from __future__ import annotations

import dataclasses

import pytest

pytestmark = pytest.mark.usefixtures("core_package")

# The Airframe fields as batch 2 froze them, in order (W4-04, P0a.10).
BATCH2_FIELDS = (
    'name', 'description', 'px4_version', 'px4_airframe', 'px4_airframe_url', 'mass',
    'hover_thrust', 'max_thrust', 'min_thrust', 'motor_constant', 'max_rot_velocity',
    'moment_constant', 'battery_n_cells', 'battery_v_empty_cell', 'battery_v_charged_cell',
    'battery_capacity', 'attitude_time_constant', 'drag_coefficient', 'provenance')


@pytest.fixture(scope='module')
def airframe():
    from contract_uav_core.config import airframe as module
    return module


def raw(module, name):
    import yaml
    with open(module.airframe_path(name), encoding='utf-8') as f:
        return yaml.safe_load(f)


def expect_error(module, data, *fragments):
    with pytest.raises(module.AirframeError) as info:
        module.parse_airframe(data, origin='test.yaml')
    message = str(info.value)
    for fragment in fragments:
        assert fragment in message, message


@pytest.mark.parametrize('name, n_cells, pack', [('dexi', 6, (19.8, 25.2)), ('x500_sitl', 4, (14.4, 16.8))])
def test_pack_voltage_range(airframe, name, n_cells, pack):
    af = airframe.load_airframe(name)
    assert af.battery_n_cells == n_cells
    assert af.battery_v_full_cell == 4.2
    assert af.battery_pack_voltage_range == pack     # n x v_empty_cell .. n x v_full_cell
    param = af.provenance['battery_v_full_cell']
    assert (param.status, param.unit, param.source, param.param) == ('datasheet', 'V', None, None)
    assert 'LiPo' in param.note and 'no PX4 parameter' in param.note


def test_dexi_notes_still_ask_for_check_5(airframe):
    dexi = airframe.load_airframe('dexi')
    for field, param in dexi.provenance.items():
        assert '(check 5)' in param.note, field
    for variable, note in dexi.unavailable.items():
        assert '(check 5)' in note, variable


@pytest.mark.parametrize('name', ['dexi', 'x500_sitl'])
def test_motor_temperature_is_unavailable(airframe, name):
    af = airframe.load_airframe(name)
    assert set(af.unavailable) == {'motor_temperature'}
    assert 'no motor temperature' in af.unavailable['motor_temperature']
    with pytest.raises(TypeError):
        af.unavailable['motor_temperature'] = 'x'      # read-only


def test_schema_is_additive(airframe):
    names = [f.name for f in dataclasses.fields(airframe.Airframe)]
    assert tuple(names[:len(BATCH2_FIELDS)]) == BATCH2_FIELDS
    assert names[len(BATCH2_FIELDS):] == ['battery_v_full_cell', 'unavailable']
    assert airframe.SCHEMA_VERSION == 2      # review N3: the new keys are required, so the version moved
    assert airframe.STATUSES[:5] == ('px4_airframe', 'px4_param_default', 'model_sdf', 'assumed', 'unknown')
    assert 'datasheet' in airframe.STATUSES


def test_datasheet_status_rules(airframe):
    # allowed outside the plant section, e.g. on a vehicle number, with a note
    data = raw(airframe, 'x500_sitl')
    data['vehicle']['max_thrust'].update(status='datasheet', note='from the ESC datasheet', param=None)
    assert airframe.parse_airframe(data).provenance['max_thrust'].status == 'datasheet'
    # needs a note
    data = raw(airframe, 'x500_sitl')
    del data['battery']['v_full_cell']['note']
    expect_error(airframe, data, 'battery.v_full_cell.note', "a note is required when status is 'datasheet'")
    # not in the plant section
    data = raw(airframe, 'x500_sitl')
    data['plant']['drag_coefficient']['status'] = 'datasheet'
    expect_error(airframe, data, 'plant.drag_coefficient.status', "must be 'assumed'")
    # a value, never null
    data = raw(airframe, 'x500_sitl')
    data['battery']['v_full_cell']['value'] = None
    expect_error(airframe, data, 'battery.v_full_cell.value', "null is allowed only with status 'unknown'")


def test_full_cell_voltage_checks(airframe):
    data = raw(airframe, 'dexi')
    del data['battery']['v_full_cell']
    expect_error(airframe, data, 'battery', 'missing key(s): v_full_cell')
    data = raw(airframe, 'dexi')
    data['battery']['v_full_cell']['value'] = 4.0          # below BAT1_V_CHARGED 4.05
    expect_error(airframe, data, 'battery', 'v_charged_cell <= v_full_cell')
    data = raw(airframe, 'dexi')
    data['battery']['v_full_cell']['unit'] = 'mV'
    expect_error(airframe, data, 'battery.v_full_cell.unit', "expected 'V'")


def test_unavailable_section_rules(airframe):
    data = raw(airframe, 'x500_sitl')
    del data['unavailable']
    expect_error(airframe, data, 'top level', 'missing key(s): unavailable')
    data = raw(airframe, 'x500_sitl')
    data['unavailable'] = None
    expect_error(airframe, data, 'unavailable', 'expected a mapping')
    data = raw(airframe, 'x500_sitl')
    data['unavailable'] = {}
    assert dict(airframe.parse_airframe(data).unavailable) == {}
    data = raw(airframe, 'x500_sitl')
    data['unavailable']['motor_temperature']['status'] = 'unknown'
    expect_error(airframe, data, 'unavailable.motor_temperature.status', "expected 'unavailable'")
    data = raw(airframe, 'x500_sitl')
    del data['unavailable']['motor_temperature']['note']
    expect_error(airframe, data, 'unavailable.motor_temperature', 'missing key(s): note')
    data = raw(airframe, 'x500_sitl')
    data['unavailable']['motor_temperature']['note'] = ' '
    expect_error(airframe, data, 'unavailable.motor_temperature.note', 'non-empty string')
    data = raw(airframe, 'x500_sitl')
    data['unavailable']['motor_temperature']['value'] = 25.0
    expect_error(airframe, data, 'unavailable.motor_temperature', 'unknown key(s): value')
    data = raw(airframe, 'x500_sitl')
    data['unavailable']['Motor Temp'] = {'status': 'unavailable', 'note': 'x'}
    expect_error(airframe, data, 'unavailable.Motor Temp', 'lower case')
    data = raw(airframe, 'x500_sitl')
    data['unavailable']['example_input'] = {'status': 'unavailable', 'note': 'x', 'param': 'EXAMPLE_PARAM'}
    assert set(airframe.parse_airframe(data).unavailable) == {'motor_temperature', 'example_input'}


def version_1(module, name='x500_sitl', drop=('v_full_cell', 'unavailable')):
    """The packaged file turned back into the batch-2 (version 1) format."""
    data = raw(module, name)
    data['schema_version'] = 1
    if 'v_full_cell' in drop:
        del data['battery']['v_full_cell']
    if 'unavailable' in drop:
        del data['unavailable']
    return data


def test_version_1_file_fails_with_a_clear_message(airframe, tmp_path):
    import yaml
    path = tmp_path / 'old_x500.yaml'
    path.write_text(yaml.safe_dump(version_1(airframe)), encoding='utf-8')
    with pytest.raises(airframe.AirframeError) as info:
        airframe.load_airframe_file(path)
    message = str(info.value)
    assert message.startswith('old_x500.yaml: schema_version: version 1 is the airframe format before P1.3'), message
    assert 'missing here: battery.v_full_cell, unavailable (top-level section)' in message, message
    assert 'set schema_version: 2' in message and 'see config/airframes/x500_sitl.yaml for an example' in message
    # it names only what is actually missing
    expect_error(airframe, version_1(airframe, drop=('unavailable',)),
                 'schema_version', 'missing here: unavailable (top-level section).')
    expect_error(airframe, version_1(airframe, 'dexi', drop=('v_full_cell',)),
                 'schema_version', 'missing here: battery.v_full_cell.')
    expect_error(airframe, version_1(airframe, drop=()), 'schema_version', 'missing here: none.')
    # other versions keep the plain message
    data = raw(airframe, 'x500_sitl')
    data['schema_version'] = 0
    expect_error(airframe, data, 'schema_version', 'expected 2, got 0')
    assert raw(airframe, 'dexi')['schema_version'] == raw(airframe, 'x500_sitl')['schema_version'] == 2

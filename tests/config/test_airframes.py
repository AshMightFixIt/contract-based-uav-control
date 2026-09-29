"""The airframe files (config/airframes/*.yaml) and their schema and loader (config/airframe.py)."""
from __future__ import annotations

import copy
import dataclasses

import pytest

pytestmark = pytest.mark.usefixtures("core_package")

PX4_TAG = 'https://raw.githubusercontent.com/PX4/PX4-Autopilot/v1.16.2/'
GZ_MODELS = 'https://raw.githubusercontent.com/PX4/PX4-gazebo-models/e05f4312d3f28aa621157610584a4870406cb6d3/'
DEXI_NOTE = 'PX4 default, confirm on the vehicle (check 5)'


@pytest.fixture(scope='module')
def airframe():
    from contract_uav_core.config import airframe as module
    return module


def raw(module, name):
    """The packaged file as plain data, for mutation."""
    import yaml
    with open(module.airframe_path(name), encoding='utf-8') as f:
        return yaml.safe_load(f)


def expect_error(module, data, *fragments):
    with pytest.raises(module.AirframeError) as info:
        module.parse_airframe(data, origin='test.yaml')
    message = str(info.value)
    assert message.startswith('test.yaml: '), message
    for fragment in fragments:
        assert fragment in message, message
    return message


def test_packaged_airframes(airframe):
    assert airframe.available_airframes() == ('dexi', 'x500_sitl')


def test_x500_sitl_loads(airframe):
    x500 = airframe.load_airframe('x500_sitl')
    assert (x500.name, x500.px4_version, x500.px4_airframe) == ('x500_sitl', 'v1.16.2', '4001_gz_x500')
    assert x500.mass == 2.0643 and x500.provenance['mass'].status == 'model_sdf'
    assert x500.hover_thrust == 0.60 and x500.provenance['hover_thrust'].status == 'px4_airframe'
    assert x500.provenance['hover_thrust'].param == 'MPC_THR_HOVER'
    assert (x500.max_thrust, x500.min_thrust) == (1.0, 0.12)
    assert (x500.motor_constant, x500.max_rot_velocity, x500.moment_constant) == (8.54858e-06, 1000.0, 0.016)
    assert isinstance(x500.battery_n_cells, int) and x500.battery_n_cells == 4
    assert (x500.battery_v_empty_cell, x500.battery_v_charged_cell) == (3.6, 4.05)
    assert x500.battery_capacity is None and x500.provenance['battery_capacity'].status == 'unknown'
    assert (x500.attitude_time_constant, x500.drag_coefficient) == (0.15, 0.05)


def test_dexi_loads_with_px4_defaults_and_unknowns(airframe):
    dexi = airframe.load_airframe('dexi')
    assert dexi.px4_airframe == '4601_droneblocks_dexi_5'
    assert (dexi.hover_thrust, dexi.max_thrust, dexi.min_thrust) == (0.22, 0.5, 0.025)
    assert (dexi.battery_n_cells, dexi.battery_v_empty_cell, dexi.battery_capacity) == (6, 3.3, 4000.0)
    assert dexi.battery_v_charged_cell == 4.05
    assert dexi.provenance['battery_v_charged_cell'].status == 'px4_param_default'
    for name in ('mass', 'motor_constant', 'max_rot_velocity', 'moment_constant'):
        assert getattr(dexi, name) is None and dexi.provenance[name].status == 'unknown'
        assert dexi.provenance[name].note
    for name, param in dexi.provenance.items():
        if param.status in ('px4_airframe', 'px4_param_default'):
            assert param.note.startswith(DEXI_NOTE), name
        assert '(check 5)' in param.note, name   # every DEXI value needs the vehicle


def test_provenance_rules_hold_in_both_files(airframe):
    for name in airframe.available_airframes():
        af = airframe.load_airframe(name)
        assert af.px4_airframe_url.startswith(PX4_TAG)
        assert set(af.provenance) == set(airframe.FIELDS)
        for field, param in af.provenance.items():
            spec = airframe.FIELDS[field]
            assert param.unit == spec.unit
            if spec.section == 'plant':
                assert param.status == 'assumed' and param.note, (name, field)
            else:
                assert param.status != 'assumed', (name, field)
            if param.status in ('px4_airframe', 'px4_param_default'):
                assert param.source.startswith(PX4_TAG) and param.param, (name, field)
            if param.status == 'model_sdf':
                assert param.source.startswith(GZ_MODELS), (name, field)


def test_airframe_is_frozen(airframe):
    af = airframe.load_airframe('x500_sitl')
    with pytest.raises(dataclasses.FrozenInstanceError):
        af.hover_thrust = 0.5
    with pytest.raises(TypeError):
        af.provenance['hover_thrust'] = None
    with pytest.raises(dataclasses.FrozenInstanceError):
        af.provenance['mass'].value = 1.0


def test_unknown_airframe_name(airframe):
    with pytest.raises(airframe.AirframeError, match="unknown airframe 'x600'; available: dexi, x500_sitl"):
        airframe.load_airframe('x600')


@pytest.mark.parametrize('where, key', [
    ('top', 'px4'), ('vehicle', 'hover_thrust'), ('battery', 'n_cells'), ('entry', 'unit')])
def test_missing_key_fails(airframe, where, key):
    data = raw(airframe, 'x500_sitl')
    if where == 'top':
        del data[key]
        expect_error(airframe, data, 'top level', f'missing key(s): {key}')
    elif where == 'entry':
        del data['vehicle']['mass'][key]
        expect_error(airframe, data, 'vehicle.mass', f'missing key(s): {key}')
    else:
        del data[where][key]
        expect_error(airframe, data, where, f'missing key(s): {key}')


@pytest.mark.parametrize('where', ['top', 'px4', 'section', 'entry'])
def test_unknown_key_fails(airframe, where):
    data = raw(airframe, 'x500_sitl')
    target = {'top': data, 'px4': data['px4'], 'section': data['vehicle'],
              'entry': data['vehicle']['mass']}[where]
    target['thrust_to_weight'] = 2.0
    expect_error(airframe, data, 'unknown key(s): thrust_to_weight')


@pytest.mark.parametrize('value, fragment', [
    ('0.6', 'expected a number in norm, got str'),
    (True, 'expected a number in norm, got bool'),
    ([0.6], 'expected a number in norm, got list'),
    (1.5, 'outside [0.01, 1.0]'),
    (float('nan'), 'outside'),
])
def test_wrong_type_or_range_fails(airframe, value, fragment):
    data = raw(airframe, 'x500_sitl')
    data['vehicle']['hover_thrust']['value'] = value
    expect_error(airframe, data, 'vehicle.hover_thrust.value', fragment)


def test_integer_field_rejects_a_float(airframe):
    data = raw(airframe, 'dexi')
    data['battery']['n_cells']['value'] = 6.0
    expect_error(airframe, data, 'battery.n_cells.value', 'expected an integer')


def test_wrong_unit_fails(airframe):
    data = raw(airframe, 'x500_sitl')
    data['vehicle']['mass']['unit'] = 'g'
    expect_error(airframe, data, 'vehicle.mass.unit', "expected 'kg', got 'g'")


def test_null_unknown_only_where_the_schema_allows_it(airframe):
    # allowed: the DEXI mass is unknown
    assert airframe.load_airframe('dexi').mass is None
    # not allowed: hover thrust may never be unknown
    data = raw(airframe, 'dexi')
    data['vehicle']['hover_thrust'].update(value=None, status='unknown')
    expect_error(airframe, data, 'vehicle.hover_thrust.value', 'may not be unknown')
    # null needs status unknown
    data = raw(airframe, 'x500_sitl')
    data['vehicle']['mass']['value'] = None
    expect_error(airframe, data, 'vehicle.mass.value', "null is allowed only with status 'unknown'")
    # unknown needs null and a note
    data = raw(airframe, 'x500_sitl')
    data['vehicle']['mass']['status'] = 'unknown'
    expect_error(airframe, data, 'vehicle.mass.value', "status 'unknown' needs value null")
    data = raw(airframe, 'dexi')
    del data['vehicle']['mass']['note']
    expect_error(airframe, data, 'vehicle.mass.note', 'a note is required')


def test_status_rules(airframe):
    data = raw(airframe, 'x500_sitl')
    data['vehicle']['mass']['status'] = 'measured'
    expect_error(airframe, data, 'vehicle.mass.status', "'measured' is not one of")
    data = raw(airframe, 'x500_sitl')
    data['plant']['drag_coefficient'].update(status='px4_param_default', param='X', source=PX4_TAG + 'x')
    expect_error(airframe, data, 'plant.drag_coefficient.status', "must be 'assumed'")
    data = raw(airframe, 'x500_sitl')
    data['vehicle']['max_thrust'].update(status='assumed')
    expect_error(airframe, data, 'vehicle.max_thrust.status', 'only in the plant section')
    data = raw(airframe, 'x500_sitl')
    del data['vehicle']['max_thrust']['param']
    expect_error(airframe, data, 'vehicle.max_thrust.param', 'PX4 parameter name is required')
    data = raw(airframe, 'x500_sitl')
    data['vehicle']['mass']['source'] = 'https://example.com/model.sdf'
    expect_error(airframe, data, 'vehicle.mass.source', 'needs a source URL')


def test_cross_field_checks(airframe):
    data = raw(airframe, 'dexi')
    data['vehicle']['hover_thrust']['value'] = 0.6   # above MPC_THR_MAX 0.5
    expect_error(airframe, data, 'vehicle', 'min_thrust <= hover_thrust <= max_thrust')
    data = raw(airframe, 'dexi')
    data['battery']['v_empty_cell']['value'] = 4.1
    expect_error(airframe, data, 'battery', 'v_empty_cell < v_charged_cell')


def test_schema_version_and_name(airframe):
    data = raw(airframe, 'x500_sitl')
    data['schema_version'] = 2
    expect_error(airframe, data, 'schema_version', 'expected 1, got 2')
    data = raw(airframe, 'x500_sitl')
    data['name'] = ''
    expect_error(airframe, data, 'name', 'non-empty string')


def test_file_errors(airframe, tmp_path):
    text = airframe.airframe_path('x500_sitl').read_text(encoding='utf-8')
    # A copy under another name loads by path; the name check is load_airframe's.
    other = tmp_path / 'renamed.yaml'
    other.write_text(text, encoding='utf-8')
    assert airframe.load_airframe_file(other).name == 'x500_sitl'
    # Duplicate keys are rejected instead of silently keeping the last one.
    dup = tmp_path / 'dup.yaml'
    dup.write_text(text.replace('schema_version: 1\n', 'schema_version: 1\nschema_version: 1\n'),
                   encoding='utf-8')
    with pytest.raises(airframe.AirframeError, match=r"dup\.yaml: .*duplicate key 'schema_version'"):
        airframe.load_airframe_file(dup)
    broken = tmp_path / 'broken.yaml'
    broken.write_text('vehicle: [unclosed\n', encoding='utf-8')
    with pytest.raises(airframe.AirframeError, match=r'broken\.yaml: not a valid airframe file'):
        airframe.load_airframe_file(broken)
    empty = tmp_path / 'empty.yaml'
    empty.write_text('', encoding='utf-8')
    with pytest.raises(airframe.AirframeError, match=r'empty\.yaml: top level: expected a mapping'):
        airframe.load_airframe_file(empty)


@pytest.mark.parametrize('text, shown', [
    ('? [a, b]\n: 1\n', "['a', 'b']"),                   # unhashable list key
    ('? {a: 1}\n: 1\n', "{'a': 1}"),                     # unhashable mapping key
    ('schema_version: 1\n2: two\n', '2'),                # number key
    ('vehicle:\n  null: 1\n', 'None'),                   # null key, nested
])
def test_non_string_keys_are_rejected(airframe, tmp_path, text, shown):
    path = tmp_path / 'keys.yaml'
    path.write_text(text, encoding='utf-8')
    with pytest.raises(airframe.AirframeError) as info:
        airframe.load_airframe_file(path)
    message = str(info.value)
    assert message.startswith('keys.yaml: not a valid airframe file: '), message
    assert f'key {shown} on line ' in message and 'is not a string' in message, message


def test_unchanged_copy_passes(airframe):
    data = raw(airframe, 'dexi')
    assert airframe.parse_airframe(copy.deepcopy(data), origin='x').hover_thrust == 0.22

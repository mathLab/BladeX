import numpy as np
import pandas as pd
import pytest
from bladex import DesignTable


R = 0.2                                # propeller radius [m]
R_DIM = np.array([0.5, 1.0]) * R       # dimensional radii [m] -> [0.1, 0.2]
THETA_DEG = np.degrees(0.1)            # 5.7295779...°, i.e. 0.1 rad


def pitch_for_angle(angle_deg):
    """Pitch length [m] at each section giving a pitch angle of angle_deg"""
    return 2 * np.pi * R_DIM * np.tan(np.radians(angle_deg))


@pytest.fixture
def df():
    """Skew 0.01 m and 0.02 m at zero pitch -> theta = 0.1 rad in both sections."""
    return pd.DataFrame({
        'radii': [0.5, 1.0],
        'chord': [0.05, 0.04],
        'pitch': [0.0, 0.0],      # pitch length in meters
        'skew': [0.01, 0.02],
    })


# ---------- skew conversion in __init__ ----------

def test_deg_is_default_and_left_unchanged(df):
    df['skew'] = [3.0, 5.0]
    dt = DesignTable(df)
    np.testing.assert_allclose(dt.table['skew'], [3.0, 5.0])


def test_deg_explicit_left_unchanged(df):
    df['skew'] = [3.0, 5.0]
    dt = DesignTable(df, skew_unit='deg')
    np.testing.assert_allclose(dt.table['skew'], [3.0, 5.0])


def test_deg_ignores_pitch_and_radius(df):
    df['skew'] = [3.0, 5.0]
    df['pitch'] = pitch_for_angle(60)
    dt = DesignTable(df, skew_unit='deg')
    np.testing.assert_allclose(dt.table['skew'], [3.0, 5.0])


def test_meters_to_degrees_zero_pitch(df):
    dt = DesignTable(df, skew_unit='m', propeller_radius=R)
    np.testing.assert_allclose(dt.table['skew'], [THETA_DEG, THETA_DEG])


def test_meters_to_degrees_pitch_angle_45(df):
    df['pitch'] = pitch_for_angle(45)       # cos = sqrt(2)/2
    dt = DesignTable(df, skew_unit='m', propeller_radius=R)
    np.testing.assert_allclose(dt.table['skew'], [THETA_DEG * np.sqrt(2) / 2] * 2)


def test_meters_to_degrees_pitch_angle_60(df):
    df['pitch'] = pitch_for_angle(60)       # cos = 0.5
    dt = DesignTable(df, skew_unit='m', propeller_radius=R)
    np.testing.assert_allclose(dt.table['skew'], [THETA_DEG / 2] * 2)


def test_meters_to_degrees_varying_pitch(df):
    df['pitch'] = [0.0, pitch_for_angle(60)[1]]
    dt = DesignTable(df, skew_unit='m', propeller_radius=R)
    np.testing.assert_allclose(dt.table['skew'], [THETA_DEG, THETA_DEG / 2])


def test_pitch_ratio_is_not_used_as_an_angle(df):
    """Regression: with P = 2*pi*r the ratio P/(2*pi*r) is 1, which is tan(phi), not phi.
    The factor must be cos(arctan(1)) = 0.7071, not cos(1) = 0.5403."""
    df['pitch'] = 2 * np.pi * R_DIM
    dt = DesignTable(df, skew_unit='m', propeller_radius=R)
    wrong_factor = np.cos(1.0)
    assert not np.allclose(dt.table['skew'], THETA_DEG * wrong_factor)
    np.testing.assert_allclose(dt.table['skew'], [THETA_DEG * np.sqrt(2) / 2] * 2)


def test_very_large_pitch_gives_zero_skew(df):
    df['pitch'] = [1e9, 1e9]                # phi -> 90°, cos -> 0
    dt = DesignTable(df, skew_unit='m', propeller_radius=R)
    np.testing.assert_allclose(dt.table['skew'], [0.0, 0.0], atol=1e-6)


def test_zero_skew_stays_zero(df):
    df['skew'] = [0.0, 0.0]
    df['pitch'] = pitch_for_angle(45)
    dt = DesignTable(df, skew_unit='m', propeller_radius=R)
    np.testing.assert_allclose(dt.table['skew'], [0.0, 0.0])


def test_conversion_scales_with_propeller_radius(df):
    """Zero pitch: same skew in meters on a propeller twice as big means half the angle"""
    small = DesignTable(df, skew_unit='m', propeller_radius=R)
    big = DesignTable(df, skew_unit='m', propeller_radius=2 * R)
    np.testing.assert_allclose(big.table['skew'], small.table['skew'] / 2)


def test_roundtrip_degrees_back_to_meters(df):
    """Inverse formula written independently: SK = theta_rad * r / cos(phi),
    with phi = arctan(P / (2 pi r))"""
    pitch = np.array([0.3, 0.5])
    df['pitch'] = pitch
    df['skew'] = [0.012, 0.025]
    dt = DesignTable(df, skew_unit='m', propeller_radius=R)
    phi = np.arctan(pitch / (2 * np.pi * R_DIM))
    skew_back = np.radians(dt.table['skew'].to_numpy()) * R_DIM / np.cos(phi)
    np.testing.assert_allclose(skew_back, [0.012, 0.025])


def test_table_without_skew_column_is_accepted(df):
    dt = DesignTable(df.drop(columns='skew'), skew_unit='m', propeller_radius=R)
    assert 'skew' not in dt.table.columns


# ---------- validation ----------

def test_meters_without_radius_raises(df):
    with pytest.raises(ValueError, match="propeller_radius"):
        DesignTable(df, skew_unit='m')


def test_deg_without_radius_is_fine(df):
    DesignTable(df, skew_unit='deg')  # must not raise


def test_meters_without_pitch_raises(df):
    with pytest.raises(ValueError, match="pitch"):
        DesignTable(df.drop(columns='pitch'), skew_unit='m', propeller_radius=R)


def test_unknown_unit_raises(df):
    with pytest.raises(ValueError, match="Unknown skew unit"):
        DesignTable(df, skew_unit='rad', propeller_radius=R)


def test_input_dataframe_not_mutated(df):
    original = df.copy()
    DesignTable(df, skew_unit='m', propeller_radius=R)
    pd.testing.assert_frame_equal(df, original)


def test_internal_table_is_a_copy(df):
    dt = DesignTable(df)
    df.loc[0, 'chord'] = 999.0
    assert dt.table.loc[0, 'chord'] == 0.05


# ---------- from_csv ----------

def test_from_csv_meters(tmp_path, df):
    path = tmp_path / "table.csv"
    df.to_csv(path, index=False)
    dt = DesignTable.from_csv(str(path), skew_unit='m', propeller_radius=R)
    np.testing.assert_allclose(dt.table['skew'], [THETA_DEG, THETA_DEG])


def test_from_csv_meters_with_pitch(tmp_path, df):
    df['pitch'] = pitch_for_angle(60)
    path = tmp_path / "table.csv"
    df.to_csv(path, index=False)
    dt = DesignTable.from_csv(str(path), skew_unit='m', propeller_radius=R)
    np.testing.assert_allclose(dt.table['skew'], [THETA_DEG / 2] * 2)


def test_from_csv_default_is_degrees(tmp_path, df):
    path = tmp_path / "table.csv"
    df.to_csv(path, index=False)
    dt = DesignTable.from_csv(str(path))
    np.testing.assert_allclose(dt.table['skew'], [0.01, 0.02])


def test_from_csv_meters_without_radius_raises(tmp_path, df):
    path = tmp_path / "table.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="propeller_radius"):
        DesignTable.from_csv(str(path), skew_unit='m')


# ---------- accessors ----------

def test_radii(df):
    assert DesignTable(df).radii() == [0.5, 1.0]


def test_properties_excludes_radii(df):
    props = DesignTable(df).properties()
    assert 'radii' not in props.columns
    assert set(props.columns) == {'chord', 'pitch', 'skew'}


# ---------- add_property ----------

def test_add_regular_property(df):
    dt = DesignTable(df)
    dt.add_property('rake', [0.1, 0.2])
    np.testing.assert_allclose(dt.table['rake'], [0.1, 0.2])


def test_add_property_ignores_skew_unit_when_not_skew(df):
    dt = DesignTable(df)
    dt.add_property('rake', [0.1, 0.2], skew_unit='m')
    np.testing.assert_allclose(dt.table['rake'], [0.1, 0.2])


def test_add_skew_in_degrees_default(df):
    dt = DesignTable(df.drop(columns='skew'))
    dt.add_property('skew', [3.0, 5.0])
    np.testing.assert_allclose(dt.table['skew'], [3.0, 5.0])


def test_add_skew_in_meters_is_converted(df):
    dt = DesignTable(df.drop(columns='skew'), propeller_radius=R)
    dt.add_property('skew', [0.01, 0.02], skew_unit='m')
    np.testing.assert_allclose(dt.table['skew'], [THETA_DEG, THETA_DEG])


def test_add_skew_in_meters_uses_pitch(df):
    df['pitch'] = pitch_for_angle(60)
    dt = DesignTable(df.drop(columns='skew'), propeller_radius=R)
    dt.add_property('skew', [0.01, 0.02], skew_unit='m')
    np.testing.assert_allclose(dt.table['skew'], [THETA_DEG / 2] * 2)


def test_add_skew_in_meters_before_pitch_raises(df):
    dt = DesignTable(df.drop(columns=['skew', 'pitch']), propeller_radius=R)
    with pytest.raises(ValueError, match="pitch"):
        dt.add_property('skew', [0.01, 0.02], skew_unit='m')
    assert 'skew' not in dt.table.columns  # nothing half-written


def test_add_skew_in_meters_without_radius_raises(df):
    dt = DesignTable(df.drop(columns='skew'))
    with pytest.raises(ValueError, match="propeller_radius"):
        dt.add_property('skew', [0.01, 0.02], skew_unit='m')


def test_add_skew_unknown_unit_raises(df):
    dt = DesignTable(df.drop(columns='skew'), propeller_radius=R)
    with pytest.raises(ValueError, match="Unknown skew unit"):
        dt.add_property('skew', [0.01, 0.02], skew_unit='rad')


def test_add_not_admissible_raises(df):
    with pytest.raises(ValueError, match="not among the admissible"):
        DesignTable(df).add_property('foo', [1, 2])


def test_add_existing_property_raises(df):
    with pytest.raises(ValueError, match="already in the design table"):
        DesignTable(df).add_property('chord', [1, 2])


def test_add_property_wrong_length_raises(df):
    dt = DesignTable(df)
    with pytest.raises(ValueError):
        dt.add_property('rake', [0.1, 0.2, 0.3])


def test_add_skew_in_meters_wrong_length_raises(df):
    dt = DesignTable(df.drop(columns='skew'), propeller_radius=R)
    with pytest.raises(ValueError):
        dt.add_property('skew', [0.01, 0.02, 0.03], skew_unit='m')


# ---------- change_property ----------

def test_change_regular_property(df):
    dt = DesignTable(df)
    dt.change_property('chord', [0.5, 0.6])
    np.testing.assert_allclose(dt.table['chord'], [0.5, 0.6])


def test_change_skew_in_degrees_uses_new_values(df):
    dt = DesignTable(df)
    dt.change_property('skew', [9.0, 8.0])
    np.testing.assert_allclose(dt.table['skew'], [9.0, 8.0])


def test_change_skew_in_meters_uses_new_values(df):
    dt = DesignTable(df, propeller_radius=R)  # initial skew is [0.01, 0.02] deg
    dt.change_property('skew', [0.02, 0.04], skew_unit='m')
    np.testing.assert_allclose(dt.table['skew'], [2 * THETA_DEG, 2 * THETA_DEG])


def test_change_skew_in_meters_uses_current_pitch(df):
    dt = DesignTable(df, propeller_radius=R)
    dt.change_property('pitch', pitch_for_angle(60))
    dt.change_property('skew', [0.01, 0.02], skew_unit='m')
    np.testing.assert_allclose(dt.table['skew'], [THETA_DEG / 2] * 2)


def test_change_skew_does_not_touch_other_columns(df):
    dt = DesignTable(df, propeller_radius=R)
    dt.change_property('skew', [0.01, 0.02], skew_unit='m')
    np.testing.assert_allclose(dt.table['chord'], [0.05, 0.04])
    np.testing.assert_allclose(dt.table['pitch'], [0.0, 0.0])


def test_change_not_admissible_raises(df):
    with pytest.raises(ValueError, match="not among the admissible"):
        DesignTable(df).change_property('foo', [1, 2])


def test_change_missing_property_raises(df):
    with pytest.raises(ValueError, match="not in the design table"):
        DesignTable(df).change_property('rake', [1, 2])


def test_change_skew_unknown_unit_raises(df):
    dt = DesignTable(df, propeller_radius=R)
    with pytest.raises(ValueError, match="Unknown skew unit"):
        dt.change_property('skew', [0.01, 0.02], skew_unit='rad')


# ---------- internals ----------

def test_skew_conversion_returns_ndarray(df):
    dt = DesignTable(df, propeller_radius=R)
    assert isinstance(dt._skew_to_degrees([0.01, 0.02], 'm'), np.ndarray)
    assert isinstance(dt._skew_to_degrees([0.01, 0.02], 'deg'), np.ndarray)
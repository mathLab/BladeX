from unittest import TestCase
import bladex.profiles as pr
import bladex.blade as bl
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_sample_blade_NACA():
    sections = np.asarray([pr.NacaProfile(digits='0012') for i in range(10)])
    radii = np.arange(0.4, 1.31, 0.1)
    chord_lengths = np.concatenate((np.arange(0.55, 1.1, 0.15),
                                    np.arange(1.03, 0.9, -0.03),
                                    np.array([0.3])))
    pitch = np.append(np.arange(3.0, 4., 0.2), np.arange(4.1, 3.2, -0.2))
    rake = np.append(np.arange(5e-3, 0.08, 1e-2), np.arange(0.075, 0.02, -3e-2))
    skew_angles = np.append(np.arange(-4., -9., -3.), np.arange(-7., 15., 3.))
    return bl.Blade(
        sections=sections,
        radii=radii,
        chord_lengths=chord_lengths,
        pitch=pitch,
        rake=rake,
        skew_angles=skew_angles)


def create_sample_blade_NACA_10():
    sections = np.asarray(
        [pr.NacaProfile(digits='0012', n_points=10) for i in range(2)])
    radii = np.array([0.4, 0.5])
    chord_lengths = np.array([0.55, 0.7])
    pitch = np.array([3.0, 3.2])
    rake = np.array([5e-3, 0.015])
    skew_angles = np.array([-4., -7])
    return bl.Blade(
        sections=sections,
        radii=radii,
        chord_lengths=chord_lengths,
        pitch=pitch,
        rake=rake,
        skew_angles=skew_angles)


def create_sample_blade_custom():
    xup = np.linspace(-1.0, 1.0, 5)
    yup = np.array([0.0, 0.75, 1.0, 0.75, 0.0])
    xdown = np.linspace(-1.0, 1.0, 5)
    ydown = np.zeros(5)
    sections = np.asarray([
        pr.CustomProfile(xup=xup, yup=yup, xdown=xdown, ydown=ydown)
        for i in range(10)
    ])
    radii = np.arange(0.4, 1.31, 0.1)
    chord_lengths = np.concatenate((np.arange(0.55, 1.1, 0.15),
                                    np.arange(1.03, 0.9, -0.03),
                                    np.array([0.3])))
    pitch = np.append(np.arange(3.0, 4., 0.2), np.arange(4.1, 3.2, -0.2))
    rake = np.append(np.arange(5e-3, 0.08, 1e-2), np.arange(0.075, 0.02, -3e-2))
    skew_angles = np.append(np.arange(-4., -9., -3.), np.arange(-7., 15., 3.))
    return bl.Blade(
        sections=sections,
        radii=radii,
        chord_lengths=chord_lengths,
        pitch=pitch,
        rake=rake,
        skew_angles=skew_angles)


class TestBlade(TestCase):
    """
    Test case for the blade module.
    We first test using NACA and custom profiles then we proceed only with NACA
    to avoid WET code.
    """

    def test_sections_inheritance_naca(self):
        blade = create_sample_blade_NACA()
        self.assertIsInstance(blade.sections[0], pr.NacaProfile)

    def test_sections_inheritance_custom(self):
        blade = create_sample_blade_custom()
        self.assertIsInstance(blade.sections[0], pr.CustomProfile)

    def test_sections_1_naca(self):
        blade = create_sample_blade_NACA()
        self.assertIsInstance(blade.sections, np.ndarray)

    def test_sections_1_custom(self):
        blade = create_sample_blade_custom()
        self.assertIsInstance(blade.sections, np.ndarray)

    def test_sections_2_naca(self):
        blade = create_sample_blade_NACA()
        self.assertIsInstance(blade.sections[0].xup_coordinates, np.ndarray)

    def test_sections_2_custom(self):
        blade = create_sample_blade_custom()
        self.assertIsInstance(blade.sections[0].xup_coordinates, np.ndarray)

    def test_radii_naca(self):
        blade = create_sample_blade_NACA()
        np.testing.assert_equal(blade.radii, np.arange(0.4, 1.31, 0.1))

    def test_radii_custom(self):
        blade = create_sample_blade_custom()
        np.testing.assert_equal(blade.radii, np.arange(0.4, 1.31, 0.1))

    def test_chord_naca(self):
        blade = create_sample_blade_NACA()
        np.testing.assert_equal(blade.chord_lengths,
                                np.concatenate((np.arange(0.55, 1.1, 0.15),
                                                np.arange(1.03, 0.9, -0.03),
                                                np.array([0.3]))))

    def test_chord_custom(self):
        blade = create_sample_blade_custom()
        np.testing.assert_equal(blade.chord_lengths,
                                np.concatenate((np.arange(0.55, 1.1, 0.15),
                                                np.arange(1.03, 0.9, -0.03),
                                                np.array([0.3]))))

    def test_pitch_naca(self):
        blade = create_sample_blade_NACA()
        np.testing.assert_equal(blade.pitch,
                                np.append(
                                    np.arange(3.0, 4., 0.2),
                                    np.arange(4.1, 3.2, -0.2)))

    def test_pitch_custom(self):
        blade = create_sample_blade_custom()
        np.testing.assert_equal(blade.pitch,
                                np.append(
                                    np.arange(3.0, 4., 0.2),
                                    np.arange(4.1, 3.2, -0.2)))

    def test_rake_naca(self):
        blade = create_sample_blade_NACA()
        np.testing.assert_equal(blade.rake,
                                np.append(
                                    np.arange(5e-3, 0.08, 1e-2),
                                    np.arange(0.075, 0.02, -3e-2)))

    def test_rake_custom(self):
        blade = create_sample_blade_custom()
        np.testing.assert_equal(blade.rake,
                                np.append(
                                    np.arange(5e-3, 0.08, 1e-2),
                                    np.arange(0.075, 0.02, -3e-2)))

    def test_skew_naca(self):
        blade = create_sample_blade_NACA()
        np.testing.assert_equal(blade.skew_angles,
                                np.append(
                                    np.arange(-4., -9., -3.),
                                    np.arange(-7., 15., 3.)))

    def test_skew_custom(self):
        blade = create_sample_blade_custom()
        np.testing.assert_equal(blade.skew_angles,
                                np.append(
                                    np.arange(-4., -9., -3.),
                                    np.arange(-7., 15., 3.)))

    def test_sections_list_to_ndarray(self):
        blade = create_sample_blade_NACA()
        blade.sections = [pr.NacaProfile(digits='0012') for i in range(10)]
        blade._check_params()
        self.assertIsInstance(blade.sections, np.ndarray)

    def test_radii_list_to_ndarray(self):
        blade = create_sample_blade_NACA()
        blade.radii = list(range(10))
        blade._check_params()
        self.assertIsInstance(blade.radii, np.ndarray)

    def test_chord_list_to_ndarray(self):
        blade = create_sample_blade_NACA()
        blade.chord_lengths = list(range(10))
        blade._check_params()
        self.assertIsInstance(blade.chord_lengths, np.ndarray)

    def test_pitch_list_to_ndarray(self):
        blade = create_sample_blade_NACA()
        blade.pitch = list(range(10))
        blade._check_params()
        self.assertIsInstance(blade.pitch, np.ndarray)

    def test_rake_list_to_ndarray(self):
        blade = create_sample_blade_NACA()
        blade.rake = list(range(10))
        blade._check_params()
        self.assertIsInstance(blade.rake, np.ndarray)

    def test_skew_list_to_ndarray(self):
        blade = create_sample_blade_NACA()
        blade.skew_angles = list(range(10))
        blade._check_params()
        self.assertIsInstance(blade.skew_angles, np.ndarray)

    def test_sections_array_different_length(self):
        blade = create_sample_blade_NACA()
        blade.sections = np.arange(9)
        with self.assertRaises(ValueError):
            blade._check_params()

    def test_radii_array_different_length(self):
        blade = create_sample_blade_NACA()
        blade.radii = np.arange(9)
        with self.assertRaises(ValueError):
            blade._check_params()

    def test_chord_array_different_length(self):
        blade = create_sample_blade_NACA()
        blade.chord_lengths = np.arange(9)
        with self.assertRaises(ValueError):
            blade._check_params()

    def test_pitch_array_different_length(self):
        blade = create_sample_blade_NACA()
        blade.pitch = np.arange(9)
        with self.assertRaises(ValueError):
            blade._check_params()

    def test_rake_array_different_length(self):
        blade = create_sample_blade_NACA()
        blade.rake = np.arange(9)
        with self.assertRaises(ValueError):
            blade._check_params()

    def test_skew_array_different_length(self):
        blade = create_sample_blade_NACA()
        blade.skew_angles = np.arange(9)
        with self.assertRaises(ValueError):
            blade._check_params()

    def test_compute_pitch_angle(self):
        blade = create_sample_blade_NACA()
        blade.radii[1] = 1.
        blade.pitch[1] = 2.0 * np.pi
        blade.pitch_angles = blade._compute_pitch_angle()
        assert blade.pitch_angles[1] == (np.pi / 4.0)

    def test_pitch_angles_array_length(self):
        blade = create_sample_blade_NACA()
        assert blade.pitch_angles.size == 10

    def test_induced_rake_from_skew(self):
        blade = create_sample_blade_NACA()
        blade.radii[1] = 1.
        blade.skew_angles[1] = 45.
        blade.pitch_angles[1] = np.pi / 4.
        blade.induced_rake = blade._induced_rake_from_skew()
        np.testing.assert_almost_equal(blade.induced_rake[1], np.pi / 4.)

    def test_induced_rake_array_length(self):
        blade = create_sample_blade_NACA()
        assert blade.induced_rake.size == 10

    def test_blade_coordinates_up_init(self):
        blade = create_sample_blade_NACA()
        assert len(blade.blade_coordinates_up) == 0

    def test_blade_coordinates_down_init(self):
        blade = create_sample_blade_NACA()
        assert len(blade.blade_coordinates_down) == 0

    def test_planar_to_cylindrical_blade_up(self):
        blade = create_sample_blade_NACA()
        blade._planar_to_cylindrical()
        blade_coordinates_up_expected = np.load(
            'tests/test_datasets/planar_to_cylindrical_blade_up.npy')
        np.testing.assert_almost_equal(blade.blade_coordinates_up,
                                       blade_coordinates_up_expected)

    def test_planar_to_cylindrical_blade_down(self):
        blade = create_sample_blade_NACA()
        blade._planar_to_cylindrical()
        blade_coordinates_down_expected = np.load(
            'tests/test_datasets/planar_to_cylindrical_blade_down.npy')
        np.testing.assert_almost_equal(blade.blade_coordinates_down,
                                       blade_coordinates_down_expected)

    def test_transformations_reflect_blade_up(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations(reflect=True)
        blade_coordinates_up_expected = np.load(
            'tests/test_datasets/blade_up_after_transformation_reflect.npy')
        np.testing.assert_almost_equal(blade.blade_coordinates_up,
                                       blade_coordinates_up_expected)

    def test_transformations_reflect_blade_down(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations(reflect=True)
        blade_coordinates_down_expected = np.load(
            'tests/test_datasets/blade_down_after_transformation_reflect.npy')
        np.testing.assert_almost_equal(blade.blade_coordinates_down,
                                       blade_coordinates_down_expected)

    def test_transformations_no_reflect_blade_up(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations(reflect=False)
        blade_coordinates_up_expected = np.load(
            'tests/test_datasets/blade_up_after_transformation_no_reflect.npy')
        np.testing.assert_almost_equal(blade.blade_coordinates_up,
                                       blade_coordinates_up_expected)

    def test_transformations_no_reflect_blade_down(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations(reflect=False)
        blade_coordinates_down_expected = np.load(
            'tests/test_datasets/blade_down_after_transformation_no_reflect.npy'
        )
        np.testing.assert_almost_equal(blade.blade_coordinates_down,
                                       blade_coordinates_down_expected)

    def test_blade_rotate_exceptions(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations()
        with self.assertRaises(ValueError):
            blade.rotate(rad_angle=None, deg_angle=None)

    def test_blade_rotate_exceptions_2(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations()
        with self.assertRaises(ValueError):
            blade.rotate(rad_angle=np.pi, deg_angle=180)

    def test_blade_rotate_exceptions_no_transformation(self):
        blade = create_sample_blade_NACA()
        with self.assertRaises(ValueError):
            blade.rotate(rad_angle=80, deg_angle=None)

    def test_rotate_deg_section_0_xup(self):
        blade = create_sample_blade_NACA_10()
        blade.apply_transformations()
        blade.rotate(deg_angle=90)
        rotated_coordinates = np.array([
            0.23913475, 0.20945559, 0.16609993, 0.11970761, 0.07154874,
            0.02221577, -0.02796314, -0.07881877, -0.13030229, -0.18246808
        ])
        np.testing.assert_almost_equal(blade.blade_coordinates_up[0][0],
                                       rotated_coordinates)

    def test_rotate_deg_section_0_yup(self):
        blade = create_sample_blade_NACA_10()
        blade.apply_transformations()
        blade.rotate(deg_angle=90)
        rotated_coordinates = np.array([
            0.3488408, 0.37407923, 0.38722075, 0.39526658, 0.39928492,
            0.39980927, 0.39716902, 0.39160916, 0.38335976, 0.3726862
        ])
        np.testing.assert_almost_equal(blade.blade_coordinates_up[0][1],
                                       rotated_coordinates)

    def test_rotate_deg_section_0_zup(self):
        blade = create_sample_blade_NACA_10()
        blade.apply_transformations()
        blade.rotate(deg_angle=90)
        rotated_coordinates = np.array([
            0.19572966, 0.14165003, 0.1003, 0.06135417, 0.02390711, -0.01235116,
            -0.04750545, -0.08150009, -0.11417222, -0.14527558
        ])
        np.testing.assert_almost_equal(blade.blade_coordinates_up[0][2],
                                       rotated_coordinates)

    def test_rotate_rad_section_1_xdown(self):
        blade = create_sample_blade_NACA_10()
        blade.apply_transformations()
        blade.rotate(rad_angle=np.pi / 2.0)
        rotated_coordinates = np.array([
            0.29697841, 0.2176438, 0.15729805, 0.10116849, 0.04749167,
            -0.00455499, -0.05542713, -0.10535969, -0.15442047, -0.20253397
        ])
        np.testing.assert_almost_equal(blade.blade_coordinates_down[1][0],
                                       rotated_coordinates)

    def test_rotate_rad_section_1_ydown(self):
        blade = create_sample_blade_NACA_10()
        blade.apply_transformations()
        blade.rotate(rad_angle=np.pi / 2.0)
        rotated_coordinates = np.array([
            0.40908705, 0.42570092, 0.44956113, 0.47048031, 0.48652991,
            0.49660315, 0.49999921, 0.49627767, 0.48516614, 0.4664844
        ])
        np.testing.assert_almost_equal(blade.blade_coordinates_down[1][1],
                                       rotated_coordinates)

    def test_rotate_rad_section_1_zdown(self):
        blade = create_sample_blade_NACA_10()
        blade.apply_transformations()
        blade.rotate(rad_angle=np.pi / 2.0)
        rotated_coordinates = np.array([
            0.28748529, 0.26225699, 0.21884879, 0.16925801, 0.11527639,
            0.05818345, -0.00088808, -0.0608972, -0.1208876, -0.17997863
        ])
        np.testing.assert_almost_equal(blade.blade_coordinates_down[1][2],
                                       rotated_coordinates)

    def test_plot_view_elev_init(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations()
        blade.plot(elev=None)
        plt.close()

    def test_plot_view_elev(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations()
        blade.plot(elev=45)
        plt.close()

    def test_plot_view_azim_init(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations()
        blade.plot(azim=None)
        plt.close()

    def test_plot_view_azim(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations()
        blade.plot(azim=-90)
        plt.close()

    def test_plot_ax_single(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations()
        blade.plot(ax=None)
        plt.close()

    def test_plot_ax_multi(self):
        blade_1 = create_sample_blade_NACA()
        blade_1.apply_transformations()
        blade_2 = create_sample_blade_custom()
        blade_2.apply_transformations()
        fig = plt.figure()
        ax = fig.gca(projection=Axes3D.name)
        blade_1.plot(ax=ax)
        blade_2.plot(ax=ax)
        plt.close()

    def test_plot_save(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations()
        blade.plot(outfile='tests/test_datasets/test_plot.png')
        plt.close()
        self.assertTrue(os.path.isfile('tests/test_datasets/test_plot.png'))
        self.addCleanup(os.remove, 'tests/test_datasets/test_plot.png')

    def test_plot_exceptions(self):
        blade = create_sample_blade_NACA()
        with self.assertRaises(ValueError):
            blade.plot()

    def test_iges_upper_blade_not_string(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations()
        upper = 1
        with self.assertRaises(Exception):
            blade.generate_iges(
                upper_face=upper,
                lower_face=None,
                tip=None,
                display=False,
                errors=None)

    def test_iges_lower_blade_not_string(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations()
        lower = 1
        with self.assertRaises(Exception):
            blade.generate_iges(
                upper_face=None,
                lower_face=lower,
                tip=None,
                display=False,
                errors=None)

    def test_iges_tip_not_string(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations()
        tip = 1
        with self.assertRaises(Exception):
            blade.generate_iges(
                upper_face=None,
                lower_face=None,
                tip=tip,
                display=False,
                errors=None)

    def test_iges_blade_tip_generate(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations()
        blade.generate_iges(
            upper_face=None,
            lower_face=None,
            tip='tests/test_datasets/tip',
            display=False,
            errors=None)
        self.assertTrue(os.path.isfile('tests/test_datasets/tip.iges'))
        self.addCleanup(os.remove, 'tests/test_datasets/tip.iges')

    def test_iges_blade_maxDeg_exception(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations()
        with self.assertRaises(ValueError):
            blade.generate_iges(
                upper_face=None,
                lower_face=None,
                tip=None,
                maxDeg=-1,
                display=False,
                errors=None)

    def test_iges_errors_exception(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations()
        with self.assertRaises(ValueError):
            blade.generate_iges(
                upper_face=None,
                lower_face=None,
                tip=None,
                display=False,
                errors='tests/test_datasets/errors')

    def test_iges_generate_errors_upper(self):
        blade = create_sample_blade_NACA_10()
        blade.apply_transformations()
        blade.generate_iges(
            upper_face='tests/test_datasets/upper',
            lower_face=None,
            tip=None,
            display=False,
            errors='tests/test_datasets/errors')
        self.assertTrue(os.path.isfile('tests/test_datasets/upper.iges'))
        self.addCleanup(os.remove, 'tests/test_datasets/upper.iges')
        self.assertTrue(os.path.isfile('tests/test_datasets/errors.txt'))
        self.addCleanup(os.remove, 'tests/test_datasets/errors.txt')

    def test_iges_generate_errors_lower(self):
        blade = create_sample_blade_NACA_10()
        blade.apply_transformations()
        blade.generate_iges(
            upper_face=None,
            lower_face='tests/test_datasets/lower',
            tip=None,
            display=False,
            errors='tests/test_datasets/errors')
        self.assertTrue(os.path.isfile('tests/test_datasets/lower.iges'))
        self.addCleanup(os.remove, 'tests/test_datasets/lower.iges')
        self.assertTrue(os.path.isfile('tests/test_datasets/errors.txt'))
        self.addCleanup(os.remove, 'tests/test_datasets/errors.txt')

    def test_abs_to_norm_radii(self):
        blade = create_sample_blade_NACA()
        blade.radii[0] = 1.
        blade._abs_to_norm(D_prop=1.)
        assert blade.radii[0] == 2.

    def test_abs_to_norm_chord(self):
        blade = create_sample_blade_NACA()
        blade.chord_lengths[0] = 1.
        blade._abs_to_norm(D_prop=2.)
        assert blade.chord_lengths[0] == 0.5

    def test_abs_to_norm_pitch(self):
        blade = create_sample_blade_NACA()
        blade.pitch[0] = 1.
        blade._abs_to_norm(D_prop=2.)
        assert blade.pitch[0] == 0.5

    def test_abs_to_norm_rake(self):
        blade = create_sample_blade_NACA()
        blade.rake[0] = 1.
        blade._abs_to_norm(D_prop=2.)
        assert blade.rake[0] == 0.5

    def test_norm_to_abs_radii(self):
        blade = create_sample_blade_NACA()
        blade.radii[0] = 1.
        blade._norm_to_abs(D_prop=1.)
        assert blade.radii[0] == 0.5

    def test_norm_to_abs_chord(self):
        blade = create_sample_blade_NACA()
        blade.chord_lengths[0] = 1.5
        blade._norm_to_abs(D_prop=2.)
        assert blade.chord_lengths[0] == 3.

    def test_norm_to_abs_pitch(self):
        blade = create_sample_blade_NACA()
        blade.pitch[0] = 1.5
        blade._norm_to_abs(D_prop=2.)
        assert blade.pitch[0] == 3.

    def test_norm_to_abs_rake(self):
        blade = create_sample_blade_NACA()
        blade.rake[0] = 1.5
        blade._norm_to_abs(D_prop=2.)
        assert blade.rake[0] == 3.

    def test_export_ppg(self):
        blade = create_sample_blade_NACA()
        blade.export_ppg(
            filename='tests/test_datasets/data_out.ppg',
            D_prop=0.25,
            D_hub=0.075,
            n_blades=5)
        self.assertTrue(os.path.isfile('tests/test_datasets/data_out.ppg'))
        self.addCleanup(os.remove, 'tests/test_datasets/data_out.ppg')

    def test_blade_str_method(self):
        blade = create_sample_blade_NACA()
        string = ''
        string += 'Blade number of sections = {}'.format(blade.n_sections)
        string += '\nBlade radii sections = {}'.format(blade.radii)
        string += '\nChord lengths of the sectional profiles'\
                  ' = {}'.format(blade.chord_lengths)
        string += '\nRadial distribution of the pitch (in unit lengths)'\
                  ' = {}'.format(blade.pitch)
        string += '\nRadial distribution of the rake (in unit length)'\
                  ' = {}'.format(blade.rake)
        string += '\nRadial distribution of the skew angles'\
                  ' (in degrees) = {}'.format(blade.skew_angles)
        string += '\nPitch angles (in radians) for the'\
                  ' sections = {}'.format(blade.pitch_angles)
        string += '\nInduced rake from skew (in unit length)'\
                  ' for the sections = {}'.format(blade.induced_rake)
        assert blade.__str__() == string

from unittest import TestCase
import bladex.profiles as pr
import bladex.blade as bl
import numpy as np
import os
import matplotlib.pyplot as plt


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

    def test_plot(self):
        blade = create_sample_blade_NACA()
        blade.apply_transformations()
        blade.plot()
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

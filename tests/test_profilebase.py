import os
from unittest import TestCase
import unittest
import bladex.profilebase as pb
import numpy as np
import matplotlib.pyplot as plt


def create_sample_profile():
    """
    xup_coordinates, yup_coordinates, xdown_coordinates, ydown_coordinates
    """
    profile = pb.ProfileBase()
    profile.xup_coordinates = np.linspace(-1.0, 1.0, 5)
    profile.yup_coordinates = np.array([0.0, 0.75, 1.0, 0.75, 0.0])
    profile.xdown_coordinates = np.linspace(-1.0, 1.0, 5)
    profile.ydown_coordinates = np.zeros(5)
    return profile


def create_sample_profile_2():
    """
    xup_coordinates != xdown_coordinates elementwise
    """
    profile = pb.ProfileBase()
    profile.xup_coordinates = np.linspace(-1.0, 1.0, 5)
    profile.yup_coordinates = np.array([0.0, 0.75, 1.0, 0.75, 0.0])
    profile.xdown_coordinates = np.linspace(-1.0, 1.0, 5)
    profile.xdown_coordinates[2] = 0.1
    profile.ydown_coordinates = np.zeros(5)
    return profile


def create_sample_profile_3():
    """
    Coordinates of unit circle (i.e. x^2+y^2=1)
    """
    profile = pb.ProfileBase()
    profile.xup_coordinates = np.linspace(-1.0, 1.0, num=250)
    profile.yup_coordinates = np.sqrt(1 - np.power(profile.xup_coordinates, 2))
    profile.xdown_coordinates = np.linspace(-1.0, 1.0, num=250)
    profile.ydown_coordinates = -np.sqrt(
        1 - np.power(profile.xup_coordinates, 2))
    return profile


class TestProfileBase(TestCase):
    def test_xup_init(self):
        profile = pb.ProfileBase()
        assert profile.xup_coordinates is None

    def test_yup_init(self):
        profile = pb.ProfileBase()
        assert profile.yup_coordinates is None

    def test_xdown_init(self):
        profile = pb.ProfileBase()
        assert profile.xdown_coordinates is None

    def test_ydown_init(self):
        profile = pb.ProfileBase()
        assert profile.ydown_coordinates is None

    def test_chord_init(self):
        profile = pb.ProfileBase()
        assert profile.chord_line is None

    def test_camber_init(self):
        profile = pb.ProfileBase()
        assert profile.camber_line is None

    def test_leading_edge_size(self):
        profile = pb.ProfileBase()
        assert profile.leading_edge.size == 2

    def test_trailing_edge_size(self):
        profile = pb.ProfileBase()
        assert profile.trailing_edge.size == 2

    def test_leading_edge_init(self):
        profile = pb.ProfileBase()
        np.testing.assert_equal(profile.leading_edge, np.zeros(2))

    def test_trailing_edge_init(self):
        profile = pb.ProfileBase()
        np.testing.assert_equal(profile.trailing_edge, np.zeros(2))

    def test_update_edges_exceptions(self):
        profile = create_sample_profile()
        profile.xdown_coordinates[0] = 0.1
        with self.assertRaises(ValueError):
            profile._update_edges()

    def test_update_edges_exceptions_2(self):
        profile = create_sample_profile()
        profile.xdown_coordinates[-1] = 0.1
        with self.assertRaises(ValueError):
            profile._update_edges()

    def test_update_edges_leading_edge(self):
        profile = create_sample_profile()
        profile.yup_coordinates[-1] = 1.0
        profile._update_edges()
        np.testing.assert_equal(profile.leading_edge, np.array([-1.0, 0.0]))

    def test_update_edges_trailing_edge(self):
        profile = create_sample_profile()
        profile._update_edges()
        np.testing.assert_equal(profile.trailing_edge, np.array([1.0, 0.0]))

    def test_update_edges_trailing_edge_finite(self):
        profile = create_sample_profile()
        profile.yup_coordinates[-1] = 1.0
        profile._update_edges()
        np.testing.assert_equal(profile.trailing_edge, np.array([1.0, 0.5]))

    def test_interpolate_crds_exceptions_1(self):
        profile = pb.ProfileBase()
        with self.assertRaises(TypeError):
            profile.interpolate_coordinates(num=500.5, radius=1.0)

    def test_interpolate_crds_exceptions_2(self):
        profile = pb.ProfileBase()
        with self.assertRaises(ValueError):
            profile.interpolate_coordinates(num=-500, radius=1.0)

    def test_interpolate_crds_exceptions_3(self):
        profile = pb.ProfileBase()
        with self.assertRaises(ValueError):
            profile.interpolate_coordinates(num=500, radius=-1.0)

    def test_interpolate_coordinates_xup(self):
        profile = create_sample_profile_3()
        xup_actual = profile.interpolate_coordinates(num=500, radius=10)[0]
        xup_expected = np.load('tests/test_datasets/interp_xup_unit_circle.npy')
        np.testing.assert_almost_equal(xup_actual, xup_expected)

    def test_interpolate_coordinates_yup(self):
        profile = create_sample_profile_3()
        yup_actual = profile.interpolate_coordinates(num=500, radius=10)[2]
        yup_expected = np.load('tests/test_datasets/interp_yup_unit_circle.npy')
        np.testing.assert_almost_equal(yup_actual, yup_expected, decimal=5)

    def test_interpolate_coordinates_xdown(self):
        profile = create_sample_profile_3()
        xdown_actual = profile.interpolate_coordinates(num=500, radius=10)[1]
        xdown_expected = np.load(
            'tests/test_datasets/interp_xdown_unit_circle.npy')
        np.testing.assert_almost_equal(xdown_actual, xdown_expected)

    def test_interpolate_coordinates_ydown(self):
        profile = create_sample_profile_3()
        ydown_actual = profile.interpolate_coordinates(num=500, radius=10)[3]
        ydown_expected = np.load(
            'tests/test_datasets/interp_ydown_unit_circle.npy')
        np.testing.assert_almost_equal(ydown_actual, ydown_expected, decimal=5)

    def test_chord_line_shape(self):
        profile = create_sample_profile()
        profile.compute_chord_line(n_interpolated_points=None)
        assert profile.chord_line.shape == (2, 5)

    def test_chord_line(self):
        profile = create_sample_profile()
        profile.compute_chord_line(n_interpolated_points=None)
        expected_chord = np.array([np.linspace(-1, 1, 5), np.zeros(5)])
        np.testing.assert_equal(profile.chord_line, expected_chord)

    def test_chord_line_interpolated(self):
        # Interpolation by keyword
        profile = create_sample_profile()
        profile.compute_chord_line(n_interpolated_points=500)
        assert profile.chord_line.shape == (2, 500)

    def test_chord_line_interpolated_2(self):
        # Interpolation because x_up != x_down elementwise
        profile = create_sample_profile_2()
        profile.compute_chord_line(n_interpolated_points=None)
        assert profile.chord_line.shape == (2, 500)

    def test_camber_line_shape(self):
        profile = create_sample_profile()
        profile.compute_camber_line(n_interpolated_points=None)
        assert profile.camber_line.shape == (2, 5)

    def test_camber_line(self):
        profile = create_sample_profile()
        profile.compute_camber_line(n_interpolated_points=None)
        expected_camber = np.array(
            [np.linspace(-1, 1, 5),
             np.array([0, 0.375, 0.5, 0.375, 0])])
        np.testing.assert_equal(profile.camber_line, expected_camber)

    def test_camber_line_interpolated(self):
        profile = create_sample_profile()
        profile.compute_camber_line(n_interpolated_points=500)
        assert profile.camber_line.shape == (2, 500)

    def test_camber_line_interpolated_2(self):
        profile = create_sample_profile_2()
        profile.compute_camber_line(n_interpolated_points=None)
        assert profile.camber_line.shape == (2, 500)

    def test_deform_camber_exceptions(self):
        profile = create_sample_profile()
        with self.assertRaises(TypeError):
            profile.deform_camber_line()

    def test_deform_camber(self):
        profile = create_sample_profile()
        profile.deform_camber_line(
            percent_change=100, n_interpolated_points=None)
        expected_deformed_camber = np.array(
            [np.linspace(-1.0, 1.0, 5),
             np.array([0.0, 0.75, 1.0, 0.75, 0.0])])
        np.testing.assert_equal(profile.camber_line, expected_deformed_camber)

    def test_deform_camber_yup(self):
        profile = create_sample_profile()
        profile.deform_camber_line(
            percent_change=100, n_interpolated_points=None)
        expected_yup_coordinates = np.array([0.0, 1.125, 1.5, 1.125, 0.0])
        np.testing.assert_equal(profile.yup_coordinates,
                                expected_yup_coordinates)

    def test_deform_camber_ydown(self):
        profile = create_sample_profile()
        profile.deform_camber_line(
            percent_change=100, n_interpolated_points=None)
        expected_ydown_coordinates = np.array([0.0, 0.375, 0.5, 0.375, 0.0])
        np.testing.assert_equal(profile.ydown_coordinates,
                                expected_ydown_coordinates)

    def test_deform_camber_interpolated(self):
        profile = create_sample_profile()
        profile.deform_camber_line(percent_change=50, n_interpolated_points=500)
        assert profile.camber_line[1].size == 500

    def test_deform_camber_interpolated_2(self):
        profile = create_sample_profile_2()
        profile.deform_camber_line(
            percent_change=50, n_interpolated_points=None)
        assert profile.camber_line[1].size == 500

    def test_reference_point_shape(self):
        profile = create_sample_profile()
        assert profile.reference_point.size == 2

    def test_reference_point(self):
        profile = create_sample_profile()
        np.testing.assert_equal(profile.reference_point, np.zeros(2))

    def test_chord_length(self):
        profile = create_sample_profile()
        assert profile.chord_length == 2.0

    def test_max_thickness(self):
        profile = create_sample_profile()
        thickness = profile.max_thickness(n_interpolated_points=None)
        assert thickness == 1.0

    def test_max_thickness_interpolated(self):
        profile = create_sample_profile()
        thickness = profile.max_thickness(n_interpolated_points=500)
        np.testing.assert_almost_equal(thickness, 1.0, decimal=4)

    def test_max_thickness_interpolated_2(self):
        profile = create_sample_profile_2()
        thickness = profile.max_thickness(n_interpolated_points=None)
        np.testing.assert_almost_equal(thickness, 1.0, decimal=4)

    def test_max_camber(self):
        profile = create_sample_profile()
        camber_max = profile.max_camber(n_interpolated_points=None)
        assert camber_max == 0.5

    def test_max_camber_interpolated(self):
        profile = create_sample_profile()
        camber_max = profile.max_camber(n_interpolated_points=500)
        np.testing.assert_almost_equal(camber_max, 0.5, decimal=5)

    def test_max_camber_interpolated_2(self):
        profile = create_sample_profile_2()
        camber_max = profile.max_camber(n_interpolated_points=None)
        np.testing.assert_almost_equal(camber_max, 0.5, decimal=5)

    def test_max_camber_negative(self):
        profile = create_sample_profile()
        profile.ydown_coordinates = np.array([0.0, -0.75, -1.0, -0.75, 0.0])
        profile.yup_coordinates = np.zeros(5)
        camber_max = profile.max_camber(n_interpolated_points=None)
        assert camber_max == -0.5

    def test_rotate_exceptions(self):
        profile = pb.ProfileBase()
        with self.assertRaises(ValueError):
            profile.rotate(rad_angle=None, deg_angle=None)

    def test_rotate_exceptions_2(self):
        profile = pb.ProfileBase()
        with self.assertRaises(ValueError):
            profile.rotate(rad_angle=np.pi, deg_angle=180)

    def test_rotate_in_degrees_xup(self):
        profile = create_sample_profile()
        profile.rotate(deg_angle=135)
        rotated_xup_coordinates = np.array(
            [0.70710678, -0.1767767, -0.70710678, -0.88388348, -0.70710678])
        np.testing.assert_almost_equal(profile.xup_coordinates,
                                       rotated_xup_coordinates)

    def test_rotate_in_degrees_yup(self):
        profile = create_sample_profile()
        profile.rotate(deg_angle=135)
        rotated_yup_coordinates = np.array(
            [-0.70710678, -0.88388348, -0.70710678, -0.1767767, 0.70710678])
        np.testing.assert_almost_equal(profile.yup_coordinates,
                                       rotated_yup_coordinates)

    def test_rotate_in_degrees_xdown(self):
        profile = create_sample_profile()
        profile.rotate(deg_angle=135)
        rotated_xdown_coordinates = np.array(
            [0.70710678, 0.35355339, 0.0, -0.35355339, -0.70710678])
        np.testing.assert_almost_equal(profile.xdown_coordinates,
                                       rotated_xdown_coordinates)

    def test_rotate_in_degrees_ydown(self):
        profile = create_sample_profile()
        profile.rotate(deg_angle=135)
        rotated_ydown_coordinates = np.array(
            [-0.70710678, -0.35355339, 0.0, 0.35355339, 0.70710678])
        np.testing.assert_almost_equal(profile.ydown_coordinates,
                                       rotated_ydown_coordinates)

    def test_rotate_in_radians_xup(self):
        profile = create_sample_profile()
        profile.rotate(rad_angle=3 * np.pi / 4)
        rotated_xup_coordinates = np.array(
            [0.70710678, -0.1767767, -0.70710678, -0.88388348, -0.70710678])
        np.testing.assert_almost_equal(profile.xup_coordinates,
                                       rotated_xup_coordinates)

    def test_rotate_in_radians_yup(self):
        profile = create_sample_profile()
        profile.rotate(rad_angle=3 * np.pi / 4)
        rotated_yup_coordinates = np.array(
            [-0.70710678, -0.88388348, -0.70710678, -0.1767767, 0.70710678])
        np.testing.assert_almost_equal(profile.yup_coordinates,
                                       rotated_yup_coordinates)

    def test_rotate_in_radians_xdown(self):
        profile = create_sample_profile()
        profile.rotate(rad_angle=3 * np.pi / 4)
        rotated_xdown_coordinates = np.array(
            [0.70710678, 0.35355339, 0.0, -0.35355339, -0.70710678])
        np.testing.assert_almost_equal(profile.xdown_coordinates,
                                       rotated_xdown_coordinates)

    def test_rotate_in_radians_ydown(self):
        profile = create_sample_profile()
        profile.rotate(rad_angle=3 * np.pi / 4)
        rotated_ydown_coordinates = np.array(
            [-0.70710678, -0.35355339, 0.0, 0.35355339, 0.70710678])
        np.testing.assert_almost_equal(profile.ydown_coordinates,
                                       rotated_ydown_coordinates)

    def test_translate_exceptions(self):
        profile = pb.ProfileBase()
        with self.assertRaises(TypeError):
            profile.translate()

    def test_translate_xup(self):
        profile = create_sample_profile()
        profile.translate(translation=[1.0, 2.0])
        translated_xup_coordinates = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_equal(profile.xup_coordinates,
                                translated_xup_coordinates)

    def test_translate_yup(self):
        profile = create_sample_profile()
        profile.translate(translation=[1.0, 2.0])
        translated_yup_coordinates = np.array([2.0, 2.75, 3.0, 2.75, 2.0])
        np.testing.assert_equal(profile.yup_coordinates,
                                translated_yup_coordinates)

    def test_translate_xdown(self):
        profile = create_sample_profile()
        profile.translate(translation=[1.0, 2.0])
        translated_xdown_coordinates = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_equal(profile.xdown_coordinates,
                                translated_xdown_coordinates)

    def test_translate_ydown(self):
        profile = create_sample_profile()
        profile.translate(translation=[1.0, 2.0])
        translated_ydown_coordinates = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        np.testing.assert_equal(profile.ydown_coordinates,
                                translated_ydown_coordinates)

    def test_flip_xup(self):
        profile = create_sample_profile()
        profile.flip()
        flipped_xup_coordinates = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
        np.testing.assert_equal(profile.xup_coordinates,
                                flipped_xup_coordinates)

    def test_flip_yup(self):
        profile = create_sample_profile()
        profile.flip()
        flipped_yup_coordinates = np.array([0.0, -0.75, -1.0, -0.75, 0.0])
        np.testing.assert_equal(profile.yup_coordinates,
                                flipped_yup_coordinates)

    def test_flip_xdown(self):
        profile = create_sample_profile()
        profile.flip()
        flipped_xdown_coordinates = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
        np.testing.assert_equal(profile.xdown_coordinates,
                                flipped_xdown_coordinates)

    def test_flip_ydown(self):
        profile = create_sample_profile()
        profile.flip()
        flipped_ydown_coordinates = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_equal(profile.ydown_coordinates,
                                flipped_ydown_coordinates)

    def test_scale_exceptions(self):
        profile = pb.ProfileBase()
        with self.assertRaises(TypeError):
            profile.scale()

    def test_scale_xup(self):
        profile = create_sample_profile()
        profile.scale(factor=1.5)
        scaled_xup_coordinates = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
        np.testing.assert_equal(profile.xup_coordinates, scaled_xup_coordinates)

    def test_scale_yup(self):
        profile = create_sample_profile()
        profile.scale(factor=1.5)
        scaled_yup_coordinates = np.array([0.0, 1.125, 1.5, 1.125, 0.0])
        np.testing.assert_equal(profile.yup_coordinates, scaled_yup_coordinates)

    def test_scale_xdown(self):
        profile = create_sample_profile()
        profile.scale(factor=1.5)
        scaled_xdown_coordinates = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
        np.testing.assert_equal(profile.xdown_coordinates,
                                scaled_xdown_coordinates)

    def test_scale_ydown(self):
        profile = create_sample_profile()
        profile.scale(factor=1.5)
        scaled_ydown_coordinates = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_equal(profile.ydown_coordinates,
                                scaled_ydown_coordinates)

    def test_plot_exceptions(self):
        profile = create_sample_profile()
        with self.assertRaises(ValueError):
            profile.plot(outfile=1.2)

    def test_plot(self):
        profile = create_sample_profile()
        profile.plot()
        plt.close()

    def test_plot_save(self):
        profile = create_sample_profile()
        profile.plot(outfile='tests/test_datasets/test_plot.png')
        plt.close()
        self.assertTrue(os.path.isfile('tests/test_datasets/test_plot.png'))
        self.addCleanup(os.remove, 'tests/test_datasets/test_plot.png')
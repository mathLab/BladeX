from unittest import TestCase
import bladex.params as par
import bladex.deform as dfm
import matplotlib.pyplot as plt
import numpy as np
import os

import unittest

radii = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
chord = np.array([0.18, 0.2, 0.3, 0.37, 0.41, 0.42, 0.41, 0.1])
pitch = np.array([1.4, 1.5, 1.58, 1.6, 1.64, 1.61, 1.5, 1.3])
rake = np.array([0.0, 0.005, 0.013, 0.02, 0.03, 0.029, 0.022, 0.01])
skew = np.array([3.6, -4.4, -7.4, -7.4, -5.0, -1.3, 4.1, 11.4])
camber = np.array([0.015, 0.024, 0.029, 0.03, 0.03, 0.028, 0.025, 0.02])


class TestDeformation(TestCase):
    def test_member_param_radii(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(deform.param.radii, radii)

    def test_member_param_pitch(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(deform.param.parameters['pitch'], pitch)

    def test_member_param_rake(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(deform.param.parameters['rake'], rake)

    def test_member_param_skew(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(deform.param.parameters['skew'], skew)

    def test_member_param_camber(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(deform.param.parameters['camber'], camber)

    def test_member_param_degree_chord(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        assert deform.param.degree['chord'] == 3

    def test_member_param_npoints_pitch(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        assert deform.param.npoints['pitch'] == 400

    def test_member_param_nbasis_rake(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        assert deform.param.nbasis['rake'] == 4

    def test_member_param_deformations_skew(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(deform.param.deformations['skew'],
                                np.arange(7, 0, -1))

    def test_member_deformed_parameter_chord_init(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(deform.deformed_parameters['chord'],
                                np.zeros(8))

    def test_member_deformed_parameter_pitch_init(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(deform.deformed_parameters['pitch'],
                                np.zeros(8))

    def test_member_deformed_parameter_rake_init(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        expected = np.zeros(8)
        np.testing.assert_equal(deform.deformed_parameters['rake'], expected)

    def test_member_deformed_parameter_skew_init(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        expected = np.zeros(8)
        np.testing.assert_equal(deform.deformed_parameters['skew'], expected)

    def test_member_deformed_parameter_camber_init(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(deform.deformed_parameters['camber'],
                                np.zeros(8))

    def test_member_control_points_chord_init(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        assert deform.control_points['chord'] is None

    def test_member_control_points_pitch_init(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        assert deform.control_points['pitch'] is None

    def test_member_control_points_rake_init(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        assert deform.control_points['rake'] is None

    def test_member_control_points_skew_init(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        assert deform.control_points['skew'] is None

    def test_member_control_points_camber_init(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        assert deform.control_points['camber'] is None

    def test_member_spline_chord_init(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        assert deform.spline['chord'] is None

    def test_member_spline_pitch_init(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        assert deform.spline['pitch'] is None

    def test_member_spline_rake_init(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        assert deform.spline['rake'] is None

    def test_member_spline_skew_init(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        assert deform.spline['skew'] is None

    def test_member_spline_camber_init(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        assert deform.spline['camber'] is None

    def test_compute_control_points_incorrect_input(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        with self.assertRaises(ValueError):
            deform.compute_control_points(param='angle', rbf_points=1000)

    def test_compute_control_points_param_pitch(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.compute_control_points(param='pitch', rbf_points=1000)
        expected = np.load('tests/test_datasets/pitch_control_points.npy')
        actual = deform.control_points['pitch']
        np.testing.assert_almost_equal(actual, expected)

    def test_compute_control_points_no_rbf_param_pitch(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.compute_control_points(param='pitch', rbf_points=-1)
        expected = np.load(
            'tests/test_datasets/pitch_control_points_no_rbf.npy')
        actual = deform.control_points['pitch']
        np.testing.assert_almost_equal(actual, expected)

    def test_compute_control_points_rbf_points_not_int(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.compute_control_points(param='pitch', rbf_points=None)
        expected = np.load(
            'tests/test_datasets/pitch_control_points_no_rbf.npy')
        actual = deform.control_points['pitch']
        np.testing.assert_almost_equal(actual, expected)
        
    def test_update_control_points_while_not_computed(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        with self.assertRaises(ValueError):
            deform.update_control_points(param='pitch')

    def test_update_control_points_not_equal_deformation_size(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.param.nbasis['skew'] = 6
        deform.compute_control_points(param='skew')
        with self.assertRaises(ValueError):
            deform.update_control_points(param='skew')

    def test_update_control_points_deformations_array_different_length(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.compute_control_points(param='pitch')
        deform.param.deformations['pitch'] = np.arange(4)
        with self.assertRaises(ValueError):
            deform.update_control_points(param='pitch')

    def test_update_control_points_pitch(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.compute_control_points(param='pitch')
        deform.update_control_points(param='pitch')
        expected = np.load(
            'tests/test_datasets/pitch_updated_control_points.npy')
        actual = deform.control_points['pitch']
        np.testing.assert_almost_equal(actual, expected)

    def test_spline_camber(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.compute_control_points(param='camber', rbf_points=1000)
        deform.generate_spline(param='camber')
        expected = np.load('tests/test_datasets/camber_spline.npy')
        np.testing.assert_almost_equal(deform.spline['camber'], expected)

    def test_deformed_parameters_incorrect_input(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        with self.assertRaises(ValueError):
            deform.compute_deformed_parameters(param='angle')

    def test_deformed_parameters_spline_not_computed(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        with self.assertRaises(ValueError):
            deform.compute_deformed_parameters(param='camber')

    def test_deformed_parameters_small_tolerance(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.compute_control_points(param='camber', rbf_points=1000)
        deform.update_control_points(param='camber')
        deform.generate_spline(param='camber')
        with self.assertRaises(ValueError):
            deform.compute_deformed_parameters(param='camber', tol=1e-7)

    def test_deformed_chord(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.compute_control_points(param='chord', rbf_points=1000)
        deform.update_control_points(param='chord')
        deform.generate_spline(param='chord')
        deform.compute_deformed_parameters(param='chord', tol=1e-3)
        expected = np.load('tests/test_datasets/deformed_chord.npy')
        np.testing.assert_almost_equal(deform.deformed_parameters['chord'],
                                       expected)

    def test_deformed_pitch(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.compute_control_points(param='pitch', rbf_points=1000)
        deform.update_control_points(param='pitch')
        deform.generate_spline(param='pitch')
        deform.compute_deformed_parameters(param='pitch', tol=1e-3)
        expected = np.load('tests/test_datasets/deformed_pitch.npy')
        np.testing.assert_almost_equal(deform.deformed_parameters['pitch'],
                                       expected)

    def test_deformed_rake(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.param.degree['rake'] = 3
        deform.compute_control_points(param='rake', rbf_points=1000)
        deform.update_control_points(param='rake')
        deform.generate_spline(param='rake')
        deform.compute_deformed_parameters(param='rake', tol=1e-2)
        expected = np.load('tests/test_datasets/deformed_rake.npy')
        np.testing.assert_almost_equal(deform.deformed_parameters['rake'],
                                       expected)

    def test_deformed_skew(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.compute_control_points(param='skew', rbf_points=1000)
        deform.update_control_points(param='skew')
        deform.generate_spline(param='skew')
        deform.compute_deformed_parameters(param='skew', tol=1e-2)
        expected = np.load('tests/test_datasets/deformed_skew.npy')
        np.testing.assert_almost_equal(deform.deformed_parameters['skew'],
                                       expected)

    def test_deformed_camber(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.compute_control_points(param='camber', rbf_points=1000)
        deform.update_control_points(param='camber')
        deform.generate_spline(param='camber')
        deform.compute_deformed_parameters(param='camber', tol=1e-2)
        expected = np.load('tests/test_datasets/deformed_camber.npy')
        np.testing.assert_almost_equal(deform.deformed_parameters['camber'],
                                       expected)

    def test_compute_all(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.param.degree['rake'] = 3
        deform.compute_all(
            rbf_points=1000,
            tol_chord=1e-3,
            tol_pitch=1e-3,
            tol_rake=1e-2,
            tol_skew=1e-2,
            tol_camber=1e-2)
        expected = np.load('tests/test_datasets/deformed_camber.npy')
        np.testing.assert_almost_equal(deform.deformed_parameters['camber'],
                                       expected)

    def test_plot_chord_original_points(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform._plot(
            param='chord',
            original=True,
            ctrl_points=False,
            spline=False,
            rbf=False,
            rbf_points=500,
            deformed=False,
            outfile=None)
        plt.close()

    def test_plot_pitch_control_points_exception(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        with self.assertRaises(ValueError):
            deform._plot(
                param='pitch',
                original=False,
                ctrl_points=True,
                spline=False,
                rbf=False,
                rbf_points=500,
                deformed=False,
                outfile=None)

    def test_plot_pitch_control_points(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.compute_control_points(param='pitch')
        deform._plot(
            param='pitch',
            original=False,
            ctrl_points=True,
            spline=False,
            rbf=False,
            rbf_points=500,
            deformed=False,
            outfile=None)
        plt.close()

    def test_plot_rake_spline_exception(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        with self.assertRaises(ValueError):
            deform._plot(
                param='rake',
                original=False,
                ctrl_points=False,
                spline=True,
                rbf=False,
                rbf_points=500,
                deformed=False,
                outfile=None)

    def test_plot_rake_spline(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.param.degree['rake'] = 3
        deform.compute_control_points(param='rake')
        deform.generate_spline(param='rake')
        deform._plot(
            param='rake',
            original=False,
            ctrl_points=False,
            spline=True,
            rbf=False,
            rbf_points=500,
            deformed=False,
            outfile=None)
        plt.close()

    def test_plot_skew_rbf(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform._plot(
            param='skew',
            original=False,
            ctrl_points=False,
            spline=False,
            rbf=True,
            rbf_points=500,
            deformed=False,
            outfile=None)
        plt.close()

    def test_plot_deformed_camber_exception(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        with self.assertRaises(ValueError):
            deform._plot(
                param='camber',
                original=False,
                ctrl_points=False,
                spline=False,
                rbf=False,
                rbf_points=500,
                deformed=True,
                outfile=None)

    def test_plot_deformed_camber(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.compute_control_points(param='camber', rbf_points=1000)
        deform.update_control_points(param='camber')
        deform.generate_spline(param='camber')
        deform.compute_deformed_parameters(param='camber', tol=1e-2)
        deform._plot(
            param='camber',
            original=False,
            ctrl_points=False,
            spline=False,
            rbf=False,
            rbf_points=500,
            deformed=True,
            outfile=None)
        plt.close()

    def test_plot_save_outfile_not_string(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        outfile = 5
        with self.assertRaises(ValueError):
            deform._plot(
                param='chord',
                original=True,
                ctrl_points=False,
                spline=False,
                rbf=False,
                rbf_points=500,
                deformed=False,
                outfile=outfile)

    def test_plot_save(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        outfile = 'tests/test_datasets/test_plot_temp.png'
        deform._plot(
            param='chord',
            original=True,
            ctrl_points=False,
            spline=False,
            rbf=False,
            rbf_points=500,
            deformed=False,
            outfile=outfile)
        self.assertTrue(os.path.isfile(outfile))
        self.addCleanup(os.remove, outfile)

    def test_plot_several(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        params = ['chord', 'pitch', 'skew', 'camber']
        for param in params:
            deform.compute_control_points(param=param, rbf_points=1000)
            deform.update_control_points(param=param)
            deform.generate_spline(param=param)
            deform.compute_deformed_parameters(param=param, tol=1e-2)
        deform.plot(
            param=params,
            original=True,
            ctrl_points=True,
            spline=True,
            rbf=False,
            rbf_points=500,
            deformed=True,
            outfile=None)
        plt.close()

    def test_export_param_file(self):
        deform = dfm.Deformation('tests/test_datasets/parameters.prm')
        deform.compute_control_points(param='camber', rbf_points=1000)
        deform.update_control_points(param='camber')
        deform.generate_spline(param='camber')
        deform.compute_deformed_parameters(param='camber', tol=1e-2)
        outfile = 'tests/test_datasets/parameters_mod.prm'
        deform.export_param_file(outfile=outfile)
        self.assertTrue(os.path.isfile(outfile))
        self.addCleanup(os.remove, outfile)

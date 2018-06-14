from unittest import TestCase
import bladex.params as param
import numpy as np
import os

radii = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
chord = np.array([0.18, 0.2, 0.3, 0.37, 0.41, 0.42, 0.41, 0.1])
pitch = np.array([1.4, 1.5, 1.58, 1.6, 1.64, 1.61, 1.5, 1.3])
rake = np.array([0.0, 0.005, 0.013, 0.02, 0.03, 0.029, 0.022, 0.01])
skew = np.array([3.6, -4.4, -7.4, -7.4, -5.0, -1.3, 4.1, 11.4])
camber = np.array([0.015, 0.024, 0.029, 0.03, 0.03, 0.028, 0.025, 0.02])


class TestParam(TestCase):
    def test_member_radii_init(self):
        prm = param.ParamFile()
        assert prm.radii == None

    def test_member_chord_init(self):
        prm = param.ParamFile()
        assert prm.parameters['chord'] == None

    def test_member_pitch_init(self):
        prm = param.ParamFile()
        assert prm.parameters['pitch'] == None

    def test_member_rake_init(self):
        prm = param.ParamFile()
        assert prm.parameters['rake'] == None

    def test_member_skew_init(self):
        prm = param.ParamFile()
        assert prm.parameters['skew'] == None

    def test_member_camber_init(self):
        prm = param.ParamFile()
        assert prm.parameters['camber'] == None

    def test_nbasis_chord_init(self):
        prm = param.ParamFile()
        assert prm.nbasis['chord'] == 10

    def test_nbasis_pitch_init(self):
        prm = param.ParamFile()
        assert prm.nbasis['pitch'] == 10

    def test_nbasis_rake_init(self):
        prm = param.ParamFile()
        assert prm.nbasis['rake'] == 10

    def test_nbasis_skew_init(self):
        prm = param.ParamFile()
        assert prm.nbasis['skew'] == 10

    def test_nbasis_camber_init(self):
        prm = param.ParamFile()
        assert prm.nbasis['camber'] == 10

    def test_degree_chord_init(self):
        prm = param.ParamFile()
        assert prm.degree['chord'] == 3

    def test_degree_pitch_init(self):
        prm = param.ParamFile()
        assert prm.degree['pitch'] == 3

    def test_degree_rake_init(self):
        prm = param.ParamFile()
        assert prm.degree['rake'] == 3

    def test_degree_skew_init(self):
        prm = param.ParamFile()
        assert prm.degree['skew'] == 3

    def test_degree_camber_init(self):
        prm = param.ParamFile()
        assert prm.degree['camber'] == 3

    def test_npoints_chord_init(self):
        prm = param.ParamFile()
        assert prm.npoints['chord'] == 500

    def test_npoints_pitch_init(self):
        prm = param.ParamFile()
        assert prm.npoints['pitch'] == 500

    def test_npoints_rake_init(self):
        prm = param.ParamFile()
        assert prm.npoints['rake'] == 500

    def test_npoints_skew_init(self):
        prm = param.ParamFile()
        assert prm.npoints['skew'] == 500

    def test_npoints_camber_init(self):
        prm = param.ParamFile()
        assert prm.npoints['camber'] == 500

    def test_deformations_chord_init(self):
        prm = param.ParamFile()
        np.testing.assert_equal(prm.deformations['chord'], np.zeros(10))

    def test_deformations_pitch_init(self):
        prm = param.ParamFile()
        np.testing.assert_equal(prm.deformations['pitch'], np.zeros(10))

    def test_deformations_rake_init(self):
        prm = param.ParamFile()
        np.testing.assert_equal(prm.deformations['rake'], np.zeros(10))

    def test_deformations_skew_init(self):
        prm = param.ParamFile()
        np.testing.assert_equal(prm.deformations['skew'], np.zeros(10))

    def test_deformations_camber_init(self):
        prm = param.ParamFile()
        np.testing.assert_equal(prm.deformations['camber'], np.zeros(10))

    def test_member_radii(self):
        prm = param.ParamFile()
        prm.radii = [2, 3]
        np.testing.assert_equal(prm.radii, [2, 3])

    def test_read_file_not_string(self):
        prm = param.ParamFile()
        parameters = 3
        with self.assertRaises(TypeError):
            prm.read_parameters(filename=parameters)

    def test_read_file_not_exist(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/temp_parameters.prm')
        self.assertTrue(
            os.path.isfile('tests/test_datasets/temp_parameters.prm'))
        self.addCleanup(os.remove, 'tests/test_datasets/temp_parameters.prm')

    def test_read_file_corrupted(self):
        prm = param.ParamFile()
        with self.assertRaises(ValueError):
            prm.read_parameters('tests/test_datasets/parameters_corrupt.prm')

    def test_read_file_corrupted_2(self):
        prm = param.ParamFile()
        with self.assertRaises(ValueError):
            prm.read_parameters('tests/test_datasets/parameters_corrupt_2.prm')

    def test_read_file_corrupted_3(self):
        prm = param.ParamFile()
        with self.assertRaises(ValueError):
            prm.read_parameters('tests/test_datasets/parameters_corrupt_3.prm')

    def test_read_file_corrupted_4(self):
        prm = param.ParamFile()
        with self.assertRaises(ValueError):
            prm.read_parameters('tests/test_datasets/parameters_corrupt_4.prm')

    def test_read_file_corrupted_5(self):
        prm = param.ParamFile()
        with self.assertRaises(ValueError):
            prm.read_parameters('tests/test_datasets/parameters_corrupt_5.prm')

    def test_read_file_corrupted_6(self):
        prm = param.ParamFile()
        with self.assertRaises(ValueError):
            prm.read_parameters('tests/test_datasets/parameters_corrupt_6.prm')

    def test_read_parameter_file_radii(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(prm.radii, radii)

    def test_read_parameter_file_chord(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(prm.parameters['chord'], chord)

    def test_read_parameter_file_pitch(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(prm.parameters['pitch'], pitch)

    def test_read_parameter_file_rake(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(prm.parameters['rake'], rake)

    def test_read_parameter_file_skew(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(prm.parameters['skew'], skew)

    def test_read_parameter_file_camber(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(prm.parameters['camber'], camber)

    def test_read_parameter_degree_chord(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        assert prm.degree['chord'] == 3

    def test_read_parameter_degree_pitch(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        assert prm.degree['pitch'] == 5

    def test_read_parameter_degree_rake(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        assert prm.degree['rake'] == 4

    def test_read_parameter_degree_skew(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        assert prm.degree['skew'] == 3

    def test_read_parameter_degree_camber(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        assert prm.degree['camber'] == 2

    def test_read_parameter_npoints_chord(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        assert prm.npoints['chord'] == 500

    def test_read_parameter_npoints_pitch(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        assert prm.npoints['pitch'] == 400

    def test_read_parameter_npoints_rake(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        assert prm.npoints['rake'] == 300

    def test_read_parameter_npoints_skew(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        assert prm.npoints['skew'] == 200

    def test_read_parameter_npoints_camber(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        assert prm.npoints['camber'] == 100

    def test_read_parameter_nbasis_chord(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        assert prm.nbasis['chord'] == 5

    def test_read_parameter_nbasis_pitch(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        assert prm.nbasis['pitch'] == 6

    def test_read_parameter_nbasis_rake(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        assert prm.nbasis['rake'] == 4

    def test_read_parameter_nbasis_skew(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        assert prm.nbasis['skew'] == 7

    def test_read_parameter_nbasis_camber(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        assert prm.nbasis['camber'] == 5

    def test_read_parameter_chord_deformations(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(prm.deformations['chord'], np.arange(1, 6))

    def test_read_parameter_pitch_deformations(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(prm.deformations['pitch'], np.arange(1, 7))

    def test_read_parameter_rake_deformations(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(prm.deformations['rake'], np.arange(1, 5))

    def test_read_parameter_skew_deformations(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        np.testing.assert_equal(prm.deformations['skew'], np.arange(7, 0, -1))

    def test_read_parameter_camber_deformations(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/parameters.prm')
        expected = np.arange(5, 0, -1)
        np.testing.assert_equal(prm.deformations['camber'], expected)

    def test_write_parameter_file_not_string(self):
        prm = param.ParamFile()
        parameters = 5
        with self.assertRaises(TypeError):
            prm.write_parameters(filename=parameters)

    def test_read_parameter_pptc_radii(self):
        prm = param.ParamFile()
        prm.read_parameters('bladex/parameter_files/pptc.prm')
        assert prm.radii[1] == 0.35

    def test_read_parameter_pptc_pitch(self):
        prm = param.ParamFile()
        prm.read_parameters('bladex/parameter_files/pptc.prm')
        assert prm.parameters['pitch'][9] == 1.458

    def test_read_parameter_pptc_nbasis_chord(self):
        prm = param.ParamFile()
        prm.read_parameters('bladex/parameter_files/pptc.prm')
        assert prm.nbasis['chord'] == 10

    def test_read_parameter_pptc_degree_rake(self):
        prm = param.ParamFile()
        prm.read_parameters('bladex/parameter_files/pptc.prm')
        assert prm.degree['rake'] == 3

    def test_read_parameter_pptc_deformation_camber(self):
        prm = param.ParamFile()
        prm.read_parameters('bladex/parameter_files/pptc.prm')
        assert prm.deformations['camber'][5] == 0.0

    def test_read_parameter_default_radii(self):
        prm = param.ParamFile()
        prm.read_parameters('bladex/parameter_files/default.prm')
        assert prm.radii[1] == 0.4

    def test_read_parameter_default_chord(self):
        prm = param.ParamFile()
        prm.read_parameters('bladex/parameter_files/default.prm')
        assert prm.parameters['chord'][1] == 0.0

    def test_read_parameter_default_pitch(self):
        prm = param.ParamFile()
        prm.read_parameters('bladex/parameter_files/default.prm')
        assert prm.parameters['pitch'][3] == 0.0

    def test_read_parameter_default_rake(self):
        prm = param.ParamFile()
        prm.read_parameters('bladex/parameter_files/default.prm')
        assert prm.parameters['rake'][5] == 0.0

    def test_read_parameter_default_skew(self):
        prm = param.ParamFile()
        prm.read_parameters('bladex/parameter_files/default.prm')
        assert prm.parameters['skew'][7] == 0.0

    def test_read_parameter_default_camber(self):
        prm = param.ParamFile()
        prm.read_parameters('bladex/parameter_files/default.prm')
        assert prm.parameters['camber'][2] == 0.0

    def test_read_parameter_default_nbasis_chord(self):
        prm = param.ParamFile()
        prm.read_parameters('bladex/parameter_files/default.prm')
        assert prm.nbasis['chord'] == 10

    def test_read_parameter_default_degree_rake(self):
        prm = param.ParamFile()
        prm.read_parameters('bladex/parameter_files/default.prm')
        assert prm.degree['rake'] == 3

    def test_read_parameter_default_deformation_camber(self):
        prm = param.ParamFile()
        prm.read_parameters('bladex/parameter_files/default.prm')
        assert prm.deformations['camber'][5] == 0.0

    def test_check_parameter_no_radii(self):
        prm = param.ParamFile()
        with self.assertRaises(ValueError):
            prm.write_parameters('tests/test_datasets/temp_parameters.prm')

    def test_check_parameter_chord_zeros(self):
        prm = param.ParamFile()
        prm.radii = np.arange(1, 6)
        prm.write_parameters('tests/test_datasets/temp_parameters.prm')
        np.testing.assert_equal(prm.parameters['chord'], np.zeros(5))
        self.addCleanup(os.remove, 'tests/test_datasets/temp_parameters.prm')

    def test_check_parameter_pitch_zeros(self):
        prm = param.ParamFile()
        prm.radii = np.arange(1, 6)
        prm.write_parameters('tests/test_datasets/temp_parameters.prm')
        np.testing.assert_equal(prm.parameters['pitch'], np.zeros(5))
        self.addCleanup(os.remove, 'tests/test_datasets/temp_parameters.prm')

    def test_check_parameter_rake_zeros(self):
        prm = param.ParamFile()
        prm.radii = np.arange(1, 6)
        prm.write_parameters('tests/test_datasets/temp_parameters.prm')
        np.testing.assert_equal(prm.parameters['rake'], np.zeros(5))
        self.addCleanup(os.remove, 'tests/test_datasets/temp_parameters.prm')

    def test_check_parameter_skew_zeros(self):
        prm = param.ParamFile()
        prm.radii = np.arange(1, 6)
        prm.write_parameters('tests/test_datasets/temp_parameters.prm')
        np.testing.assert_equal(prm.parameters['skew'], np.zeros(5))
        self.addCleanup(os.remove, 'tests/test_datasets/temp_parameters.prm')

    def test_check_parameter_camber_zeros(self):
        prm = param.ParamFile()
        prm.radii = np.arange(1, 6)
        prm.write_parameters('tests/test_datasets/temp_parameters.prm')
        np.testing.assert_equal(prm.parameters['camber'], np.zeros(5))
        self.addCleanup(os.remove, 'tests/test_datasets/temp_parameters.prm')

    def test_check_parameter_radii_not_numpy(self):
        prm = param.ParamFile()
        prm.radii = list(range(1, 6))
        prm.write_parameters('tests/test_datasets/temp_parameters.prm')
        assert isinstance(prm.radii, np.ndarray)
        self.addCleanup(os.remove, 'tests/test_datasets/temp_parameters.prm')

    def test_check_parameter_chord_not_numpy(self):
        prm = param.ParamFile()
        prm.radii = np.arange(1, 6)
        prm.parameters['chord'] = list(range(1, 6))
        prm.write_parameters('tests/test_datasets/temp_parameters.prm')
        assert isinstance(prm.parameters['chord'], np.ndarray)
        self.addCleanup(os.remove, 'tests/test_datasets/temp_parameters.prm')

    def test_check_parameter_pitch_not_numpy(self):
        prm = param.ParamFile()
        prm.radii = np.arange(1, 6)
        prm.parameters['pitch'] = list(range(1, 6))
        prm.write_parameters('tests/test_datasets/temp_parameters.prm')
        assert isinstance(prm.parameters['pitch'], np.ndarray)
        self.addCleanup(os.remove, 'tests/test_datasets/temp_parameters.prm')

    def test_check_parameter_rake_not_numpy(self):
        prm = param.ParamFile()
        prm.radii = np.arange(1, 6)
        prm.parameters['rake'] = list(range(1, 6))
        prm.write_parameters('tests/test_datasets/temp_parameters.prm')
        assert isinstance(prm.parameters['rake'], np.ndarray)
        self.addCleanup(os.remove, 'tests/test_datasets/temp_parameters.prm')

    def test_check_parameter_skew_not_numpy(self):
        prm = param.ParamFile()
        prm.radii = np.arange(1, 6)
        prm.parameters['skew'] = list(range(1, 6))
        prm.write_parameters('tests/test_datasets/temp_parameters.prm')
        assert isinstance(prm.parameters['skew'], np.ndarray)
        self.addCleanup(os.remove, 'tests/test_datasets/temp_parameters.prm')

    def test_check_parameter_camber_not_numpy(self):
        prm = param.ParamFile()
        prm.radii = np.arange(1, 6)
        prm.parameters['camber'] = list(range(1, 6))
        prm.write_parameters('tests/test_datasets/temp_parameters.prm')
        assert isinstance(prm.parameters['camber'], np.ndarray)
        self.addCleanup(os.remove, 'tests/test_datasets/temp_parameters.prm')

    def test_check_parameter_chord_inhomogeneous(self):
        prm = param.ParamFile()
        prm.radii = np.arange(1, 6)
        prm.parameters['chord'] = np.arange(1, 5)
        with self.assertRaises(ValueError):
            prm.write_parameters('tests/test_datasets/temp_parameters.prm')

    def test_check_parameter_pitch_inhomogeneous(self):
        prm = param.ParamFile()
        prm.radii = np.arange(1, 6)
        prm.parameters['pitch'] = np.arange(1, 5)
        with self.assertRaises(ValueError):
            prm.write_parameters('tests/test_datasets/temp_parameters.prm')

    def test_check_parameter_rake_inhomogeneous(self):
        prm = param.ParamFile()
        prm.radii = np.arange(1, 6)
        prm.parameters['rake'] = np.arange(1, 5)
        with self.assertRaises(ValueError):
            prm.write_parameters('tests/test_datasets/temp_parameters.prm')

    def test_check_parameter_skew_inhomogeneous(self):
        prm = param.ParamFile()
        prm.radii = np.arange(1, 6)
        prm.parameters['skew'] = np.arange(1, 5)
        with self.assertRaises(ValueError):
            prm.write_parameters('tests/test_datasets/temp_parameters.prm')

    def test_check_parameter_camber_inhomogeneous(self):
        prm = param.ParamFile()
        prm.radii = np.arange(1, 6)
        prm.parameters['camber'] = np.arange(1, 5)
        with self.assertRaises(ValueError):
            prm.write_parameters('tests/test_datasets/temp_parameters.prm')

    def test_check_parameter_deformations_not_equal_nbasis(self):
        prm = param.ParamFile()
        prm.radii = np.arange(1, 6)
        prm.nbasis['pitch'] = 6
        prm.deformations['pitch'] = np.zeros(7)
        with self.assertRaises(ValueError):
            prm.write_parameters('tests/test_datasets/temp_parameters.prm')

    def test_str_method(self):
        prm = param.ParamFile()
        prm.read_parameters('tests/test_datasets/temp_parameters.prm')
        string = ''
        string += '\nradii = {}\n'.format(prm.radii)
        params = ['chord', 'pitch', 'rake', 'skew', 'camber']
        for par in params:
            string += '\n\n' + par + ' = {}\n'.format(prm.parameters[par])
            string += '\n' + par + ' degree = {}\n'.format(prm.degree[par])
            string += par + ' npoints = {}\n'.format(prm.npoints[par])
            string += par + ' nbasis = {}\n'.format(prm.nbasis[par])
            string += par + ' control points deformations =\n'
            string += '{}\n'.format(prm.deformations[par])
        assert prm.__str__() == string
        self.addCleanup(os.remove, 'tests/test_datasets/temp_parameters.prm')

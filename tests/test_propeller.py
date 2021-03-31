from unittest import TestCase
import os
import numpy as np
import bladex.profiles as pr
import bladex.blade as bl
from bladex import NacaProfile, Shaft, Propeller


def create_sample_blade_NACApptc():
    sections = np.asarray([NacaProfile('5407') for i in range(13)])
    radii=np.array([0.034375, 0.0375, 0.04375, 0.05, 0.0625, 0.075, 0.0875, 
                    0.1, 0.10625, 0.1125, 0.11875, 0.121875, 0.125])
    chord_lengths = np.array([0.039, 0.045, 0.05625, 0.06542, 0.08125, 
                              0.09417, 0.10417, 0.10708, 0.10654, 0.10417, 
                              0.09417, 0.07867, 0.025])
    pitch = np.array([0.35, 0.35, 0.36375, 0.37625, 0.3945, 0.405, 0.40875, 
                      0.4035, 0.3955, 0.38275, 0.3645, 0.35275, 0.33875])
    rake=np.array([0.0 ,0.0, 0.0005, 0.00125, 0.00335, 0.005875, 0.0075, 
                   0.007375, 0.006625, 0.00545, 0.004033, 0.0033, 0.0025])
    skew_angles=np.array([6.6262795, 3.6262795, -1.188323, -4.4654502, 
                          -7.440779, -7.3840979, -5.0367916, -1.3257914, 
                          1.0856404, 4.1448947, 7.697235, 9.5368917, 
                          11.397609])
    return bl.Blade(
        sections=sections,
        radii=radii,
        chord_lengths=chord_lengths,
        pitch=pitch,
        rake=rake,
        skew_angles=skew_angles)


class TestPropeller(TestCase):
    """
    Test case for the Propeller class.
    """

    def test_sections_inheritance_NACApptc(self):
        prop= create_sample_blade_NACApptc()
        self.assertIsInstance(prop.sections[0], pr.NacaProfile)
        
    def test_radii_NACApptc(self):
        prop = create_sample_blade_NACApptc()
        np.testing.assert_equal(prop.radii, np.array([0.034375, 0.0375, 0.04375, 
                                                      0.05, 0.0625, 0.075, 
                                                      0.0875, 0.1, 0.10625, 
                                                      0.1125, 0.11875, 0.121875, 
                                                      0.125]))

    def test_chord_NACApptc(self):
        prop = create_sample_blade_NACApptc()
        np.testing.assert_equal(prop.chord_lengths,np.array([0.039, 0.045, 
                                                             0.05625, 0.06542, 
                                                             0.08125, 0.09417, 
                                                             0.10417, 0.10708, 
                                                             0.10654, 0.10417, 
                                                             0.09417, 0.07867, 
                                                             0.025]))

    def test_pitch_NACApptc(self):
        prop = create_sample_blade_NACApptc()
        np.testing.assert_equal(prop.pitch, np.array([0.35, 0.35, 0.36375, 
                                                      0.37625, 0.3945, 0.405, 
                                                      0.40875, 0.4035, 0.3955, 
                                                      0.38275, 0.3645, 0.35275, 
                                                      0.33875]))

    def test_rake_NACApptc(self):
        prop = create_sample_blade_NACApptc()
        np.testing.assert_equal(prop.rake, np.array([0.0 ,0.0, 0.0005, 0.00125, 
                                                     0.00335, 0.005875, 0.0075, 
                                                     0.007375, 0.006625, 0.00545, 
                                                     0.004033, 0.0033, 0.0025]))

    def test_skew_NACApptc(self):
        prop = create_sample_blade_NACApptc()
        np.testing.assert_equal(prop.skew_angles, np.array([6.6262795, 
                                                            3.6262795, 
                                                            -1.188323, 
                                                            -4.4654502, 
                                                            -7.440779, 
                                                            -7.3840979, 
                                                            -5.0367916, 
                                                            -1.3257914, 
                                                            1.0856404, 
                                                            4.1448947, 
                                                            7.697235, 
                                                            9.5368917, 
                                                            11.397609]))

    def test_sections_array_different_length(self):
        prop = create_sample_blade_NACApptc()
        prop.sections = np.arange(9)
        with self.assertRaises(ValueError):
            prop._check_params()

    def test_radii_array_different_length(self):
        prop = create_sample_blade_NACApptc()
        prop.radii = np.arange(9)
        with self.assertRaises(ValueError):
            prop._check_params()

    def test_chord_array_different_length(self):
        prop = create_sample_blade_NACApptc()
        prop.chord_lengths = np.arange(9)
        with self.assertRaises(ValueError):
            prop._check_params()

    def test_pitch_array_different_length(self):
        prop = create_sample_blade_NACApptc()
        prop.pitch = np.arange(9)
        with self.assertRaises(ValueError):
            prop._check_params()

    def test_rake_array_different_length(self):
        prop = create_sample_blade_NACApptc()
        prop.rake = np.arange(9)
        with self.assertRaises(ValueError):
            prop._check_params()

    def test_skew_array_different_length(self):
        prop = create_sample_blade_NACApptc()
        prop.skew_angles = np.arange(9)
        with self.assertRaises(ValueError):
            prop._check_params()

    def test_generate_propeller_not_string(self):
        sh = Shaft("tests/test_datasets/shaft.iges")
        prop = create_sample_blade_NACApptc()
        prop = Propeller(sh, prop, 1)
        propeller_and_shaft = 1
        with self.assertRaises(Exception):
            prop.generate_propeller(propeller_and_shaft)

    def test_generate_propeller(self):
        sh = Shaft("tests/test_datasets/shaft.iges")
        prop = create_sample_blade_NACApptc()                                      errors=None)
        prop = Propeller(sh, prop, 4)
        prop.generate_propeller("tests/test_datasets/propeller_and_shaft")
        self.assertTrue(os.path.isfile('tests/test_datasets/propeller_and_shaft.iges'))
        self.addCleanup(os.remove, 'tests/test_datasets/propeller_and_shaft.iges')
        self.assertTrue(os.path.isfile('tests/test_datasets/propeller_and_shaft.stl'))
        self.addCleanup(os.remove, 'tests/test_datasets/propeller_and_shaft.stl')
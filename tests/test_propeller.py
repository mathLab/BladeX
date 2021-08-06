from unittest import TestCase
import os
import numpy as np
import bladex.profiles as pr
import bladex.blade as bl
from bladex import NacaProfile, Shaft, Propeller
from smithers.io.obj import ObjHandler
from smithers.io.stlhandler import STLHandler


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

    def test_generate_iges_not_string(self):
        sh = Shaft("tests/test_datasets/shaft.iges")
        prop = create_sample_blade_NACApptc()
        prop = Propeller(sh, prop, 1)
        propeller_and_shaft = 1
        with self.assertRaises(Exception):
            prop.generate_iges(propeller_and_shaft)

    def test_generate_stl_not_string(self):
        sh = Shaft("tests/test_datasets/shaft.iges")
        prop = create_sample_blade_NACApptc()
        prop = Propeller(sh, prop, 1)
        propeller_and_shaft = 1
        with self.assertRaises(Exception):
            prop.generate_stl(propeller_and_shaft)

    def test_generate_iges(self):
        sh = Shaft("tests/test_datasets/shaft.iges")
        prop = create_sample_blade_NACApptc()
        prop = Propeller(sh, prop, 4)
        prop.generate_iges("tests/test_datasets/propeller_and_shaft.iges")
        self.assertTrue(os.path.isfile('tests/test_datasets/propeller_and_shaft.iges'))
        self.addCleanup(os.remove, 'tests/test_datasets/propeller_and_shaft.iges')

    def test_generate_stl(self):
        sh = Shaft("tests/test_datasets/shaft.iges")
        prop = create_sample_blade_NACApptc()
        prop = Propeller(sh, prop, 4)
        prop.generate_stl("tests/test_datasets/propeller_and_shaft.stl")
        self.assertTrue(os.path.isfile('tests/test_datasets/propeller_and_shaft.stl'))
        self.addCleanup(os.remove, 'tests/test_datasets/propeller_and_shaft.stl')

    def test_generate_obj_by_coords(self):
        sh = Shaft("tests/test_datasets/shaft.iges")
        prop = create_sample_blade_NACApptc()
        prop = Propeller(sh, prop, 4)
        prop.generate_obj("tests/test_datasets/propeller_and_shaft.obj", region_selector='by_coords')

        data = ObjHandler.read('tests/test_datasets/propeller_and_shaft.obj')
        assert data.regions == ['propellerTip','propellerStem']

        # we want 0 to be the first index
        data.polygons = np.asarray(data.polygons) - 1

        tip_poly = data.polygons[:data.regions_change_indexes[1][0]]
        stem_poly = data.polygons[data.regions_change_indexes[1][0]:]

        blades_stl = STLHandler.read('/tmp/temp_blades.stl')
        shaft_stl = STLHandler.read('/tmp/temp_shaft.stl')

        # same vertices
        all_vertices = np.concatenate(
            [shaft_stl["points"], blades_stl["points"]], axis=0
        )
        unique_vertices = np.unique(all_vertices, axis=0)
        np.testing.assert_almost_equal(data.vertices, unique_vertices, decimal=3)

        blades_min_x = np.min(blades_stl['points'][:,0])

        assert np.all(data.vertices[np.asarray(tip_poly).flatten()][:,0] >= blades_min_x)
        assert not any(np.all(data.vertices[np.asarray(stem_poly).flatten()][:,0].reshape(-1,data.polygons.shape[1]) >= blades_min_x, axis=1))

    def test_generate_obj_blades_and_stem(self):
        sh = Shaft("tests/test_datasets/shaft.iges")
        prop = create_sample_blade_NACApptc()
        prop = Propeller(sh, prop, 4)
        prop.generate_obj("tests/test_datasets/propeller_and_shaft.obj", region_selector='blades_and_stem')

        data = ObjHandler.read('tests/test_datasets/propeller_and_shaft.obj')
        assert data.regions == ['propellerTip','propellerStem']

        tip_polygons = np.asarray(data.polygons[:data.regions_change_indexes[1][0]]) - 1
        stem_polygons = np.asarray(data.polygons[data.regions_change_indexes[1][0]:]) - 1

        blades_stl = STLHandler.read('/tmp/temp_blades.stl')
        shaft_stl = STLHandler.read('/tmp/temp_shaft.stl')

        # same vertices
        all_vertices = np.concatenate(
            [shaft_stl["points"], blades_stl["points"]], axis=0
        )

        unique_vertices, indexing = np.unique(
            all_vertices, return_index=True, axis=0
        )
        np.testing.assert_almost_equal(data.vertices, unique_vertices, decimal=3)

        assert np.all(indexing[stem_polygons.flatten()] < shaft_stl['points'].shape[0])
        assert np.all(indexing[tip_polygons.flatten()] >= shaft_stl['points'].shape[0])

    def test_isdisplay(self):
        assert hasattr(Propeller, "display") == True

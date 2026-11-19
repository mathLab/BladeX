from unittest import TestCase
import os
import struct

import numpy as np
import pytest
from bladex import Blade, CylinderShaft, NacaProfile, Propeller, Shaft
from bladex.export import make_compound, to_solid
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.TopAbs import (TopAbs_COMPOUND, TopAbs_EDGE, TopAbs_FACE,
                             TopAbs_SHELL, TopAbs_SOLID)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Iterator
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
    return Blade(
        sections=sections,
        radii=radii,
        chord_lengths=chord_lengths,
        pitch=pitch,
        rake=rake,
        skew_angles=skew_angles)

def test_constructor():
    shaft = Shaft("tests/test_datasets/shaft.iges")
    print("Creating propeller...")
    blade = create_sample_blade_NACApptc()
    print("Creating propeller instance...")
    propeller = Propeller(shaft, blade, 4)
    assert isinstance(propeller, Propeller)

test_constructor()
class TestPropeller(TestCase):
    """
    Test case for the Propeller class.
    """

    # def test_sections_inheritance_NACApptc(self):
    #     prop= create_sample_blade_NACApptc()
    #     self.assertIsInstance(prop.sections[0], NacaProfile)

    # def test_radii_NACApptc(self):
    #     prop = create_sample_blade_NACApptc()
    #     np.testing.assert_equal(prop.radii, np.array([0.034375, 0.0375, 0.04375,
    #                                                   0.05, 0.0625, 0.075,
    #                                                   0.0875, 0.1, 0.10625,
    #                                                   0.1125, 0.11875, 0.121875,
    #                                                   0.125]))

    # def test_chord_NACApptc(self):
    #     prop = create_sample_blade_NACApptc()
    #     np.testing.assert_equal(prop.chord_lengths,np.array([0.039, 0.045,
    #                                                          0.05625, 0.06542,
    #                                                          0.08125, 0.09417,
    #                                                          0.10417, 0.10708,
    #                                                          0.10654, 0.10417,
    #                                                          0.09417, 0.07867,
    #                                                          0.025]))

    # def test_pitch_NACApptc(self):
    #     prop = create_sample_blade_NACApptc()
    #     np.testing.assert_equal(prop.pitch, np.array([0.35, 0.35, 0.36375,
    #                                                   0.37625, 0.3945, 0.405,
    #                                                   0.40875, 0.4035, 0.3955,
    #                                                   0.38275, 0.3645, 0.35275,
    #                                                   0.33875]))

    # def test_rake_NACApptc(self):
    #     prop = create_sample_blade_NACApptc()
    #     np.testing.assert_equal(prop.rake, np.array([0.0 ,0.0, 0.0005, 0.00125,
    #                                                  0.00335, 0.005875, 0.0075,
    #                                                  0.007375, 0.006625, 0.00545,
    #                                                  0.004033, 0.0033, 0.0025]))

    # def test_skew_NACApptc(self):
    #     prop = create_sample_blade_NACApptc()
    #     np.testing.assert_equal(prop.skew_angles, np.array([6.6262795,
    #                                                         3.6262795,
    #                                                         -1.188323,
    #                                                         -4.4654502,
    #                                                         -7.440779,
    #                                                         -7.3840979,
    #                                                         -5.0367916,
    #                                                         -1.3257914,
    #                                                         1.0856404,
    #                                                         4.1448947,
    #                                                         7.697235,
    #                                                         9.5368917,
    #                                                         11.397609]))

    # def test_sections_array_different_length(self):
    #     prop = create_sample_blade_NACApptc()
    #     prop.sections = np.arange(9)
    #     with self.assertRaises(ValueError):
    #         prop._check_params()

    # def test_radii_array_different_length(self):
    #     prop = create_sample_blade_NACApptc()
    #     prop.radii = np.arange(9)
    #     with self.assertRaises(ValueError):
    #         prop._check_params()

    # def test_chord_array_different_length(self):
    #     prop = create_sample_blade_NACApptc()
    #     prop.chord_lengths = np.arange(9)
    #     with self.assertRaises(ValueError):
    #         prop._check_params()

    # def test_pitch_array_different_length(self):
    #     prop = create_sample_blade_NACApptc()
    #     prop.pitch = np.arange(9)
    #     with self.assertRaises(ValueError):
    #         prop._check_params()

    # def test_rake_array_different_length(self):
    #     prop = create_sample_blade_NACApptc()
    #     prop.rake = np.arange(9)
    #     with self.assertRaises(ValueError):
    #         prop._check_params()

    # def test_skew_array_different_length(self):
    #     prop = create_sample_blade_NACApptc()
    #     prop.skew_angles = np.arange(9)
    #     with self.assertRaises(ValueError):
    #         prop._check_params()

    # def test_generate_iges_not_string(self):
    #     sh = Shaft("tests/test_datasets/shaft.iges")
    #     prop = create_sample_blade_NACApptc()
    #     prop = Propeller(sh, prop, 1)
    #     propeller_and_shaft = 1
    #     with self.assertRaises(Exception):
    #         prop.generate_iges(propeller_and_shaft)

    # def test_generate_stl_not_string(self):
    #     sh = Shaft("tests/test_datasets/shaft.iges")
    #     prop = create_sample_blade_NACApptc()
    #     prop = Propeller(sh, prop, 1)
    #     propeller_and_shaft = 1
    #     with self.assertRaises(Exception):
    #         prop.generate_stl(propeller_and_shaft)


    # def test_generate_iges(self):
    #     sh = Shaft("tests/test_datasets/shaft.iges")
    #     prop = create_sample_blade_NACApptc()
    #     prop = Propeller(sh, prop, 4)
    #     prop.generate_iges("tests/test_datasets/propeller_and_shaft.iges")
    #     self.assertTrue(os.path.isfile('tests/test_datasets/propeller_and_shaft.iges'))
    #     self.addCleanup(os.remove, 'tests/test_datasets/propeller_and_shaft.iges')

    # def test_generate_stl(self):
    #     sh = Shaft("tests/test_datasets/shaft.iges")
    #     prop = create_sample_blade_NACApptc()
    #     prop = Propeller(sh, prop, 4)
    #     prop.generate_stl("tests/test_datasets/propeller_and_shaft.stl")
    #     self.assertTrue(os.path.isfile('tests/test_datasets/propeller_and_shaft.stl'))
    #     self.addCleanup(os.remove, 'tests/test_datasets/propeller_and_shaft.stl')

    '''
    # TODO revert asap
    def test_generate_obj_by_coords(self):
        sh = Shaft("tests/test_datasets/shaft.iges")
        prop = create_sample_blade_NACApptc()
        prop = Propeller(sh, prop, 4)
        prop.generate_obj("tests/test_datasets/propeller_and_shaft.obj", region_selector='by_coords')

        data = ObjHandler.read('tests/test_datasets/propeller_and_shaft.obj')
        assert set(data.regions) == set(['blades', 'shaftHead', 'shaftTail'])

        # we want 0 to be the first index
        data.polygons = np.asarray(data.polygons) - 1

        tip_poly = data.polygons[:data.regions_change_indexes[1][0]]
        stem_poly = data.polygons[data.regions_change_indexes[1][0]:data.regions_change_indexes[2][0]]
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
        assert data.regions == ['blades', 'shaft']

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
    '''

    def test_isdisplay(self):
        assert hasattr(Propeller, "display") == True


def _stl_triangles(filename):
    with open(filename, "rb") as stl_file:
        header = stl_file.read(84)  # 80 byte header + uint32 count
    return struct.unpack("<I", header[80:84])[0]


def _iges_faces(filename):
    reader = IGESControl_Reader()
    reader.ReadFile(filename)
    roots = reader.TransferRoots()
    explorer = TopExp_Explorer(reader.Shape(), TopAbs_FACE)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return roots, count


def _iges_edges(filename):
    reader = IGESControl_Reader()
    reader.ReadFile(filename)
    reader.TransferRoots()
    explorer = TopExp_Explorer(reader.Shape(), TopAbs_EDGE)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _first_sub_shape(shape):
    iterator = TopoDS_Iterator(shape)
    assert iterator.More()
    return iterator.Value()


def test_to_solid_converts_a_nested_single_shell():
    box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
    nested = make_compound([make_compound([_first_sub_shape(box)])])

    assert to_solid(nested).ShapeType() == TopAbs_SOLID


def test_to_solid_leaves_other_shapes_unchanged():
    # no shell at all, and a shell mixed with another sub-shape
    assert to_solid(make_compound([])).ShapeType() == TopAbs_COMPOUND

    box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
    explorer = TopExp_Explorer(box, TopAbs_FACE)
    assert explorer.More()
    mixed = make_compound([_first_sub_shape(box), explorer.Current()])

    assert to_solid(mixed).ShapeType() == TopAbs_COMPOUND


def test_export_stl_with_and_without_shaft(tmp_path):
    blade = create_sample_blade_NACApptc()
    propeller = Propeller(CylinderShaft(radius=0.02, height=0.05), blade, 4)

    full = str(tmp_path / "propeller.stl")
    blades = str(tmp_path / "blades.stl")
    propeller.export_stl(full)
    propeller.export_stl(blades, include_shaft=False)

    assert _stl_triangles(full) > _stl_triangles(blades) > 0
    with open(full, "rb") as stl_file:
        assert not stl_file.read(5).startswith(b"solid")  # binary


def test_blade_shell_is_a_single_body():
    blade = create_sample_blade_NACApptc()
    blade.build()

    assert blade.shell().ShapeType() in (TopAbs_SHELL, TopAbs_SOLID)


def test_export_iges_is_readable(tmp_path):
    blade = create_sample_blade_NACApptc()
    propeller = Propeller(CylinderShaft(radius=0.02, height=0.05), blade, 4)

    path = str(tmp_path / "propeller.iges")
    propeller.export_iges(path)

    roots, _ = _iges_faces(path)
    assert roots > 0


def test_blade_export_iges(tmp_path):
    blade = create_sample_blade_NACApptc()
    blade.build()

    plain = str(tmp_path / "blade.iges")
    with_curves = str(tmp_path / "blade_le.iges")
    blade.export_iges(plain)
    blade.export_iges(with_curves, include_le_curves=True)

    roots, faces = _iges_faces(plain)
    assert roots > 0
    assert faces >= 4
    assert _iges_edges(with_curves) > _iges_edges(plain)


def test_export_survives_unsewable_faces(tmp_path, monkeypatch):
    blade = create_sample_blade_NACApptc()
    blade.build()

    def broken_shell(self, sewing_tolerance=1e-2):
        return make_compound([self.upper_face, self.lower_face,
                              self.root_face, self.tip_face])

    monkeypatch.setattr(Blade, "shell", broken_shell)

    assert blade.generate_solid().ShapeType() == TopAbs_COMPOUND

    stl = str(tmp_path / "blade.stl")
    iges = str(tmp_path / "blade.iges")
    blade.export_stl(stl)
    blade.export_iges(iges)

    assert _stl_triangles(stl) > 0
    assert _iges_faces(iges)[1] == 4


def test_tmp_stl_is_an_alias_of_export_stl_without_shaft(tmp_path):
    blade = create_sample_blade_NACApptc()
    propeller = Propeller(CylinderShaft(radius=0.02, height=0.05), blade, 4)

    exported = str(tmp_path / "exported.stl")
    deprecated = str(tmp_path / "deprecated.stl")
    propeller.export_stl(exported, include_shaft=False)
    with pytest.warns(DeprecationWarning):
        propeller.tmp_stl(deprecated)

    assert _stl_triangles(exported) == _stl_triangles(deprecated)


def test_reflect_blade_requires_an_unbuilt_blade():
    blade = create_sample_blade_NACApptc()
    blade.build()

    with pytest.raises(ValueError):
        Propeller(CylinderShaft(), blade, 4, reflect_blade=True)


def test_generate_obj_regions(tmp_path):
    blade = create_sample_blade_NACApptc()
    propeller = Propeller(CylinderShaft(radius=0.02, height=0.05), blade, 3)

    path = str(tmp_path / "propeller.obj")
    propeller.generate_obj(path)

    data = ObjHandler.read(path)
    assert set(data.regions) == {"blades", "shaftHead", "shaftTail"}


def test_generate_obj_blades_regions(tmp_path):
    blade = create_sample_blade_NACApptc()
    propeller = Propeller(CylinderShaft(radius=0.02, height=0.05), blade, 3)

    path = str(tmp_path / "blades.obj")
    propeller.generate_obj_blades(path)

    data = ObjHandler.read(path)
    assert data.regions == ["blades"]
    assert len(data.polygons) > 0

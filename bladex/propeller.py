"""
Module for the propeller with shaft bottom-up parametrized construction.
"""
import copy
import os
import tempfile
import warnings

import numpy as np
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Display.SimpleGui import init_display

from .export import make_compound, sew_shapes, write_iges, write_stl


class Propeller(object):
    """
    Bottom-up parametrized propeller (including shaft) construction.
    The constructor requires PythonOCC to be installed.

    :param shaft.Shaft shaft: shaft to be added to the propeller
    :param blade.Blade blade: blade of the propeller
    :param int n_blades: number of blades composing the propeller
    :cvar list blades: the blades of the propeller
    :cvar shaft.Shaft shaft: the shaft of the propeller
    """

    def __init__(self, shaft, blade, n_blades, reflect_blade=False):
        self.shaft = shaft

        # The Propeller assembles an existing blade: rebuilding it here would
        # silently discard any mirror/rotate the caller applied after build().
        if blade.upper_face is None:
            blade.build(reflect=reflect_blade)
        elif reflect_blade:
            raise ValueError(
                "reflect_blade is applied by Blade.build: build the blade "
                "with the desired reflection instead of passing a built one")

        self.blades = [copy.deepcopy(blade) for _ in range(n_blades)]
        [
            blade.rotate(rad_angle=i * 2.0 * np.pi / float(n_blades))
            for i, blade in enumerate(self.blades)
        ]

    def _blades_shapes(self, sewing_tolerance=1e-2):
        """Sew the four boundary faces of every blade."""
        return [
            sew_shapes([
                blade.upper_face, blade.lower_face, blade.root_face,
                blade.tip_face
            ], sewing_tolerance) for blade in self.blades
        ]

    def blades_shape(self, sewing_tolerance=1e-2):
        """The blades only, collected in a single compound."""
        return make_compound(self._blades_shapes(sewing_tolerance))

    def compound(self, sewing_tolerance=1e-2):
        """The propeller and shaft assembled in a compound, without booleans."""
        shapes = self._blades_shapes(sewing_tolerance)
        shapes.append(self.shaft.generate_solid())
        return make_compound(shapes)

    def tmp_stl(self, filename, linear_deflection=0.05):
        """
        Export the .stl CAD for the blades only.

        Deprecated: use :meth:`export_stl` with ``include_shaft=False``.
        """
        warnings.warn(
            "tmp_stl is deprecated, use export_stl(include_shaft=False)",
            DeprecationWarning,
            stacklevel=2)
        self.export_stl(filename, linear_deflection=linear_deflection,
                        include_shaft=False)

    @property
    def shape(self):
        """The assembled propeller (alias of :meth:`compound`)."""
        return self.compound()

    @property
    def solid(self):
        """
        Best-effort boolean union of shaft and blades.

        It is not used by the export methods: a boolean failure, or a blade
        that does not close into a solid, only affects callers that explicitly
        ask for a single solid.

        :raises RuntimeError: if the boolean union fails
        :rtype: OCC.Core.TopoDS.TopoDS_Shape
        """
        blade_solids = [blade.generate_solid() for blade in self.blades]
        fuse = BRepAlgoAPI_Fuse(self.shaft.generate_solid(),
                                make_compound(blade_solids))
        fuse.SetFuzzyValue(1e-4)
        fuse.Build()
        if not fuse.IsDone():
            raise RuntimeError("Fuse shaft + blades failed")
        return fuse.Shape()

    def export_iges(self, filename, include_shaft=True,
                    sewing_tolerance=1e-2):
        """
        Export the .iges CAD for the propeller.

        Every blade is sewn into a shell and collected, together with the
        shaft, in a single compound. No boolean fuse is performed: IGES does
        not need it and the fuse is the operation that fails most often.

        :param string filename: path (with the file extension) where to store
            the .iges CAD
        :param bool include_shaft: if False, only the blades are exported
        :param float sewing_tolerance: tolerance used to sew each blade
        :raises RuntimeError: if the IGES file cannot be written
        """
        if include_shaft:
            shape = self.compound(sewing_tolerance)
        else:
            shape = self.blades_shape(sewing_tolerance)
        write_iges(shape, filename)

    def export_stl(self, filename, linear_deflection=0.05,
                   angular_deflection=0.3, include_shaft=True):
        """
        Export the .stl CAD for the propeller.

        :param string filename: path (with the file extension) where to store
            the .stl CAD
        :param float linear_deflection: relative linear deflection used for the
            tessellation; lower values give smoother surfaces
        :param float angular_deflection: angular deflection (in radians) used
            for the tessellation; lower values give smoother surfaces
        :param bool include_shaft: if False, only the blades are exported
        :raises RuntimeError: if the STL file cannot be written
        """
        if include_shaft:
            shape = self.compound()
        else:
            shape = self.blades_shape()
        write_stl(shape, filename, linear_deflection, angular_deflection)

    def generate_obj(self, filename, region_selector="by_coords", **kwargs):
        """
        Export the .obj CAD for the propeller with shaft. The resulting
        file contains the regions `blades`, `shaftHead` and `shaftTail` (or
        `blades` and `shaft` with `region_selector="blades_and_stem"`),
        selected according to the criteria passed in the parameter
        `region_selector`.

        :param string filename: path (with the file extension).
        :param string region_selector: Two selectors available:

            * `by_coords`: We compute :math:`x`, the smallest X coordinate of
                the solid which represents the blades of the propeller. Then all
                the polygons (belonging to both blades and shaft) composed of
                points whose X coordinate is greater than :math:`x` are
                considered to be part of the region `shaftHead`. The rest
                belongs to `shaftTail`;
            * `blades_and_stem`: The two regions are simply given by the blades
                and the shaft solids.
        :raises RuntimeError: if the STL file cannot be written
        """
        from smithers.io.obj import ObjHandler, WavefrontOBJ
        from smithers.io.stlhandler import STLHandler

        # we write the propeller to STL, then re-open it to obtain the points
        with tempfile.TemporaryDirectory() as tmpdir:
            shaft_stl = os.path.join(tmpdir, "shaft.stl")
            blades_stl = os.path.join(tmpdir, "blades.stl")
            write_stl(self.shaft.generate_solid(), shaft_stl)
            write_stl(make_compound(self._blades_shapes()), blades_stl)
            shaft = STLHandler.read(shaft_stl)
            blades = STLHandler.read(blades_stl)

        obj_instance = WavefrontOBJ()

        # add vertexes. first of all we check for duplicated vertexes
        all_vertices = np.concatenate(
            [shaft["points"], blades["points"]], axis=0
        )

        # unique_mapping maps items in all_vertices to items in unique_vertices
        unique_vertices, unique_mapping = np.unique(
            all_vertices, return_inverse=True, axis=0
        )
        obj_instance.vertices = unique_vertices

        def cells_to_np(cells):
            cells = np.asarray(cells)
            return unique_mapping[cells.flatten()].reshape(-1, cells.shape[1])

        # append a list of cells to obj_instance.polygons, possibly with a
        # region name
        def append_cells(cells, region_name=None):
            if region_name is not None:
                obj_instance.regions_change_indexes.append(
                    (
                        np.asarray(obj_instance.polygons).shape[0],
                        len(obj_instance.regions),
                    )
                )
                obj_instance.regions.append(region_name)

            if len(obj_instance.polygons) == 0:
                obj_instance.polygons = np.array(cells_to_np(cells))
            else:
                obj_instance.polygons = np.concatenate(
                    [obj_instance.polygons, cells_to_np(cells)], axis=0
                )

        shaft_cells = np.asarray(shaft["cells"])
        # the 0th point in blades if the last+1 point in shaft
        blades_cells = np.asarray(blades["cells"]) + len(shaft["points"])

        if region_selector == "blades_and_stem":
            defaultKwargs = {'blades_name' : 'blades', 'shaft_name' : 'shaft'}
            kwargs = {**defaultKwargs, **kwargs}
            append_cells(blades_cells, region_name=kwargs['blades_name'])
            append_cells(shaft_cells, region_name=kwargs['shaft_name'])

        elif region_selector == "by_coords":
            defaultKwargs = {'blades_name' : 'blades', 'shafthead_name' : 'shaftHead', 'shafttail_name' : 'shaftTail'}
            kwargs = {**defaultKwargs, **kwargs}
            minimum_blades_x = np.min(blades["points"][:, 0])
            maximum_blades_x = np.max(blades["points"][:, 0])
            shaft_x = shaft["points"][:, 0]

            if np.count_nonzero(shaft_x > maximum_blades_x) >= np.count_nonzero(shaft_x < maximum_blades_x):
                tip_boolean_array = shaft["points"][:, 0] <= maximum_blades_x

            elif np.count_nonzero(shaft_x < maximum_blades_x) > np.count_nonzero(shaft_x > maximum_blades_x):
                tip_boolean_array = shaft["points"][:, 0] >= minimum_blades_x

            shaft_cells_tip = np.all(
                tip_boolean_array[shaft_cells.flatten()].reshape(
                    -1, shaft_cells.shape[1]
                ),
                axis=1,
            )

            append_cells(
                shaft_cells[shaft_cells_tip],
                region_name=kwargs['shafthead_name'],
            )
            append_cells(
                shaft_cells[np.logical_not(shaft_cells_tip)],
                region_name=kwargs['shafttail_name'],
            )
            append_cells(
                blades_cells,
                region_name=kwargs['blades_name'],
            )
        else:
            raise ValueError("This selector is not supported at the moment")

        # this is needed because indexes start at 1
        obj_instance.polygons += 1

        ObjHandler.write(obj_instance, filename)

    def generate_obj_blades(self, filename):
        """
        Export the .obj CAD for the blades.

        :param string filename: path (with the file extension).
        """
        from smithers.io.obj import ObjHandler, WavefrontOBJ
        from smithers.io.stlhandler import STLHandler

        # we write the propeller to STL, then re-open it to obtain the points
        with tempfile.TemporaryDirectory() as tmpdir:
            blades_stl = os.path.join(tmpdir, "blades.stl")
            write_stl(make_compound(self._blades_shapes()), blades_stl)
            blades = STLHandler.read(blades_stl)

        obj_instance = WavefrontOBJ()

        # add vertexes. first of all we check for duplicated vertexes
        all_vertices = blades["points"]
        # unique_mapping maps items in all_vertices to items in unique_vertices
        unique_vertices, unique_mapping = np.unique(
            all_vertices, return_inverse=True, axis=0
        )
        obj_instance.vertices = unique_vertices

        def cells_to_np(cells):
            cells = np.asarray(cells)
            return unique_mapping[cells.flatten()].reshape(-1, cells.shape[1])
        # append a list of cells to obj_instance.polygons, possibly with a
        # region name
        def append_cells(cells):
            obj_instance.regions_change_indexes.append(
                (
                    np.asarray(obj_instance.polygons).shape[0],
                    len(obj_instance.regions),
                )
            )
            obj_instance.regions.append('blades')

            if len(obj_instance.polygons) == 0:
                obj_instance.polygons = np.array(cells_to_np(cells))
            else:
                obj_instance.polygons = np.concatenate(
                    [obj_instance.polygons, cells_to_np(cells)], axis=0
                )

        blades_cells = np.asarray(blades["cells"])

        append_cells(blades_cells)
        obj_instance.polygons += 1

        ObjHandler.write(obj_instance, filename)

    def display(self):
        """
        Display the propeller with shaft.
        """
        display, start_display = init_display()[:2]
        display.DisplayShape(self.compound(), update=True)
        start_display()
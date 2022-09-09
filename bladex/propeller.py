"""
Module for the propeller with shaft bottom-up parametrized construction.
"""
import numpy as np
from OCC.Core.IGESControl import IGESControl_Writer
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Extend.DataExchange import write_stl_file
from OCC.Display.SimpleGui import init_display
from smithers.io.obj import ObjHandler, WavefrontOBJ
from smithers.io.stlhandler import STLHandler


class Propeller(object):
    """
    Bottom-up parametrized propeller (including shaft) construction.
    The constructor requires PythonOCC to be installed.

    :param shaft.Shaft shaft: shaft to be added to the propeller
    :param blade.Blade blade: blade of the propeller
    :param int n_blades: number of blades composing the propeller
    :cvar OCC.Core.TopoDS.TopoDS_Solid shaft_solid: solid shaft
    :cvar OCC.Core.TopoDS.TopoDS_Shell sewed_full_body: propeller with shaft shell
    """

    def __init__(self, shaft, blade, n_blades):
        self.shaft_solid = shaft.generate_solid()

        blade.apply_transformations(reflect=True)
        blade_solid = blade.generate_solid(
            max_deg=2, display=False, errors=None
        )
        blades = []
        blades.append(blade_solid)
        for i in range(n_blades - 1):
            blade.rotate(rad_angle=1.0 * 2.0 * np.pi / float(n_blades))
            blade_solid = blade.generate_solid(
                max_deg=2, display=False, errors=None
            )
            blades.append(blade_solid)
        blades_combined = blades[0]
        for i in range(len(blades) - 1):
            boolean_union = BRepAlgoAPI_Fuse(blades_combined, blades[i + 1])
            boolean_union.Build()
            if not boolean_union.IsDone():
                raise RuntimeError("Unsuccessful assembling of blade")
            blades_combined = boolean_union.Shape()
        self.blades_solid = blades_combined

        boolean_union = BRepAlgoAPI_Fuse(self.shaft_solid, blades_combined)
        boolean_union.Build()
        result_compound = boolean_union.Shape()

        sewer = BRepBuilderAPI_Sewing(1e-2)
        sewer.Add(result_compound)
        sewer.Perform()
        self.sewed_full_body = sewer.SewedShape()

    def generate_iges(self, filename):
        """
        Export the .iges CAD for the propeller with shaft.

        :param string filename: path (with the file extension) where to store
            the .iges CAD for the propeller and shaft
        :raises RuntimeError: if the solid assembling of blades is not
            completed successfully
        """
        iges_writer = IGESControl_Writer()
        iges_writer.AddShape(self.sewed_full_body)
        iges_writer.Write(filename)

    def generate_stl(self, filename):
        """
        Export the .stl CAD for the propeller with shaft.

        :param string filename: path (with the file extension) where to store
            the .stl CAD for the propeller and shaft
        :raises RuntimeError: if the solid assembling of blades is not
            completed successfully
        """
        write_stl_file(self.sewed_full_body, filename)

    def generate_obj(self, filename, region_selector="by_coords", **kwargs):
        """
        Export the .obj CAD for the propeller with shaft. The resulting
        file contains two regions: `propellerTip` and `propellerStem`, selected
        according to the criteria passed in the parameter `region_selector`.

        :param string filename: path (with the file extension).
        :param string region_selector: Two selectors available:

            * `by_coords`: We compute :math:`x`, the smallest X coordinate of
                the solid which represents the blades of the propeller. Then all
                the polygons (belonging to both blades and shaft) composed of
                points whose X coordinate is greater than :math:`x` are
                considered to be part of the region `propellerTip`. The rest
                belongs to `propellerStem`;
            * `blades_and_stem`: The two regions are simply given by the two
                solids which are used in :func:`__init__`.
        :raises RuntimeError: if the solid assembling of blades is not
            completed successfully
        """

        # we write the propeller to STL, then re-open it to obtain the points
        write_stl_file(self.shaft_solid, "/tmp/temp_shaft.stl")
        shaft = STLHandler.read("/tmp/temp_shaft.stl")
        write_stl_file(self.blades_solid, "/tmp/temp_blades.stl")
        blades = STLHandler.read("/tmp/temp_blades.stl")

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

        # we write the propeller to STL, then re-open it to obtain the points
        write_stl_file(self.blades_solid, "/tmp/temp_blades.stl")
        blades = STLHandler.read("/tmp/temp_blades.stl")

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
        display.DisplayShape(self.sewed_full_body, update=True)
        start_display()

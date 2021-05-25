"""
Module for the propeller with shaft bottom-up parametrized construction.
"""
import numpy as np
from OCC.Core.IGESControl import IGESControl_Writer
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Extend.DataExchange import write_stl_file
from OCC.Display.SimpleGui import init_display

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
        blade_solid = blade.generate_solid(max_deg=2, 
                                           display=False, 
                                           errors=None)
        blades = []
        blades.append(blade_solid)
        for i in range(n_blades-1):
            blade.rotate(rad_angle=1.0*2.0*np.pi/float(n_blades))
            blade_solid = blade.generate_solid(max_deg=2, display=False, errors=None)
            blades.append(blade_solid)
        blades_combined = blades[0]
        for i in range(len(blades)-1):
            boolean_union = BRepAlgoAPI_Fuse(blades_combined, blades[i+1])
            boolean_union.Build()
            if not boolean_union.IsDone():
                raise RuntimeError('Unsuccessful assembling of blade')
            blades_combined = boolean_union.Shape()
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

    def display(self):
        """
        Display the propeller with shaft.
        """
        display, start_display = init_display()[:2]
        display.DisplayShape(self.sewed_full_body, update=True)
        start_display()

"""
Module for the propeller with shaft bottom-up parametrized construction.
"""
import os
import numpy as np
from bladex import Blade, Shaft
import OCC.Core.TopoDS
from OCC.Core.gp import gp_Dir, gp_Pnt, gp_Ax1, gp_Trsf
from OCC.Core.IGESControl import IGESControl_Reader, IGESControl_Writer
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_Sewing
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Extend.DataExchange import write_stl_file
from OCC.Display.SimpleGui import init_display

class Propeller(object):
    """
    Bottom-up parametrized propeller (including shaft) construction.

    :param shaft.Shaft shaft: shaft to be added to the propeller
    :param blade.Blade blade: blade of the propeller
    :param int n_blades: number of blades componing the propeller
    """

    def __init__(self, shaft, blade, n_blades):
        self.shaft_solid = shaft.generate_shaft_solid()
        blade.apply_transformations(reflect=True)
        blade_solid = blade.generate_blade_solid(max_deg=2,
                                                 display=False,
                                                 errors=None)
        blades = []
        blades.append(blade_solid)
        for i in range(n_blades-1):
            rot_dir = gp_Dir(1.0, 0.0, 0.0)
            origin = gp_Pnt(0.0, 0.0, 0.0)        
            rot_axis = gp_Ax1(origin, rot_dir)
            transf = gp_Trsf()
            transf.SetRotation(rot_axis, float(i+1)*2.0*np.pi/float(n_blades));
            transformer = BRepBuilderAPI_Transform(blade_solid, transf);
            blades.append(transformer.Shape())
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
        section_edges = boolean_union.SectionEdges()
        sewer = BRepBuilderAPI_Sewing(1e-2)
        sewer.Add(result_compound)
        sewer.Perform()
        self.sewed_full_body = sewer.SewedShape()

    def generate_propeller_iges(self, propeller_and_shaft, display=False):
        """
        Export and plot the .iges CAD for the propeller with shaft.
        This method requires PythonOCC to be installed.

        :raises RuntimeError: if the solid assembling of blades is not 
            completed successfully
        :param string propeller_and_shaft: path where to store the .iges CAD 
            for the propeller and shaft
        :param bool display: if True, then display the propeller with shaft. 
            Default value is False
        """
        iges_writer = IGESControl_Writer()
        iges_writer.AddShape(self.sewed_full_body)
        iges_writer.Write(propeller_and_shaft + '.iges')
        if display:
            display, start_display, add_menu, add_function_to_menu = init_display()
            display.DisplayShape(self.sewed_full_body, update=True)
            start_display()

    def generate_propeller_stl(self, propeller_and_shaft, display=False):
        """
        Export and plot the .stl CAD for the propeller with shaft.
        This method requires PythonOCC and numpy-stl to be installed.

        :raises RuntimeError: if the solid assembling of blades is not 
            completed successfully
        :param string propeller_and_shaft: path where to store the .stl CAD 
            for the propeller and shaft
        :param bool display: if True, then display the propeller with shaft. 
            Default value is False
        """
        write_stl_file(self.sewed_full_body, propeller_and_shaft + '.stl')
        if display:
            display, start_display, add_menu, add_function_to_menu = init_display()
            display.DisplayShape(self.sewed_full_body, update=True)
            start_display()
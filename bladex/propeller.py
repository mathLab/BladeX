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

class Propeller(object):
    """
    Bottom-up parametrized propeller (including shaft) construction.

    :param shaft.Shaft shaft: shaft to be added to the propeller
    :param blade.Blade blade: blade of the propeller
    :param int number_of blades: number of blades componing the propeller
    """

    def __init__(self, shaft, blade, number_of_blades):
        self.shaft = shaft
        self.blade = blade
        self.number_of_blades = number_of_blades

    def generate_propeller(self, propeller_and_shaft):
        """
        Generate and export the .iges and .stl CAD for the propeller with shaft.
        This method requires PythonOCC (7.4.0) to be installed.

        :raises RuntimeError: if the solid assembling of blades is not 
            completed successfully
        :param string propeller_and_shaft: path where to store the .iges and 
            .stl CAD for the propeller and shaft
        """
        shaft_solid = self.shaft.generate_shaft_solid()
        self.blade.apply_transformations(reflect=True)
        blade_solid = self.blade.generate_blade_solid(max_deg=2,
                                                 display=False,
                                                 errors=None)
        blades = []
        blades.append(blade_solid)
        for i in range(self.number_of_blades-1):
            rot_dir = gp_Dir(1.0,0.0,0.0)
            origin = gp_Pnt(0.0,0.0,0.0)        
            rot_axis = gp_Ax1(origin, rot_dir)
            transf = gp_Trsf()
            transf.SetRotation(rot_axis, float(i+1)*2.0*np.pi/float(self.number_of_blades));
            transformer = BRepBuilderAPI_Transform(blade_solid, transf);
            blades.append(transformer.Shape())
        blades_combined = blades[0]
        for i in range(len(blades)-1):
            boolean_union = BRepAlgoAPI_Fuse(blades_combined,blades[i+1])
            boolean_union.Build()
            if not boolean_union.IsDone():
                raise RuntimeError('Unsuccessful assembling of blade')
            blades_combined = boolean_union.Shape()
        boolean_union = BRepAlgoAPI_Fuse(shaft_solid, blades_combined)
        boolean_union.Build()
        result_solid = boolean_union.Shape()
        section_edges = boolean_union.SectionEdges()
        sewer = BRepBuilderAPI_Sewing(1e-2)
        sewer.Add(result_solid)
        sewer.Perform()
        sewed_full_body = sewer.SewedShape()
        iges_writer = IGESControl_Writer()
        iges_writer.AddShape(sewed_full_body)
        iges_writer.Write(propeller_and_shaft + '.iges')
        write_stl_file(sewed_full_body, propeller_and_shaft + '.stl')
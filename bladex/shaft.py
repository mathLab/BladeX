import os
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
import OCC.Core.TopoDS
from OCC.Display.SimpleGui import init_display

class Shaft(object):
    """
    Bottom-up parametrized shaft construction.

    :param string shaft_path: path of a .iges file with stored shaft information.
    :param bool display: if True, then display the shaft. Default value is False.
    """

    def __init__(self, shaft_path, display=False):
        self.shaft_path = shaft_path
        self.display = display

    def generate_shaft_solid(self):
        """
        Generate an assembled solid shaft using the 
        BRepBuilderAPI_MakeSolid  algorithm. 
        This method requires PythonOCC to be installed.

        :raises RuntimeError: if the assembling of the solid shaft is not 
            completed successfully
        :return: solid shaft
        :rtype: OCC.Core.TopoDS.TopoDS_Solid
        """
        iges_reader = IGESControl_Reader()
        iges_reader.ReadFile(self.shaft_path)
        iges_reader.TransferRoots()
        shaft_compound = iges_reader.Shape()
        sewer = BRepBuilderAPI_Sewing(1e-2)
        sewer.Add(shaft_compound)
        sewer.Perform()
        result_sewed_shaft = sewer.SewedShape()
        shaft_solid_maker = BRepBuilderAPI_MakeSolid()
        shaft_solid_maker.Add(OCC.Core.TopoDS.topods_Shell(result_sewed_shaft))
        if not shaft_solid_maker.IsDone():
            raise RuntimeError('Unsuccessful assembling of solid shaft')
        shaft_solid = shaft_solid_maker.Solid()
        if self.display:
            display, start_display, add_menu, add_function_to_menu = init_display()
            display.DisplayShape(shaft_solid, update=True)
            start_display()
        return shaft_solid
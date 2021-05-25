import os
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Extend.DataExchange import read_stl_file
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
import OCC.Core.TopoDS
from OCC.Display.SimpleGui import init_display

class Shaft(object):
    """
    Bottom-up parametrized shaft construction.

    :param string filename: path (with the file extension) of a .stl or .iges file with 
        stored shaft information.
    :cvar string filename: path (with the file extension) of a .stl or .iges file with 
        stored shaft information.
    :raises Exception: if the extension in the filename is not in .stl or .iges formats.
    """

    def __init__(self, filename):
        self.filename = filename

    def generate_solid(self):
        """
        Generate an assembled solid shaft using the BRepBuilderAPI_MakeSolid  
        algorithm. This method requires PythonOCC to be installed.

        :raises RuntimeError: if the assembling of the solid shaft is not 
            completed successfully
        :return: solid shaft
        :rtype: OCC.Core.TopoDS.TopoDS_Solid
        """
        ext = os.path.splitext(self.filename)[1][1:]
        if ext == 'stl':
            shaft_compound = read_stl_file(self.filename)
        elif ext == 'iges':
            iges_reader = IGESControl_Reader()
            iges_reader.ReadFile(self.filename)
            iges_reader.TransferRoots()
            shaft_compound = iges_reader.Shape()
        else:
            raise Exception('The shaft file is not in iges/stl formats')
        sewer = BRepBuilderAPI_Sewing(1e-2)
        sewer.Add(shaft_compound)
        sewer.Perform()
        result_sewed_shaft = sewer.SewedShape()
        shaft_solid_maker = BRepBuilderAPI_MakeSolid()
        shaft_solid_maker.Add(OCC.Core.TopoDS.topods_Shell(result_sewed_shaft))
        if not shaft_solid_maker.IsDone():
            raise RuntimeError('Unsuccessful assembling of solid shaft')
        shaft_solid = shaft_solid_maker.Solid()
        return shaft_solid

    def display(self):
        """
        Display the shaft.
        """
        shaft_solid = self.generate_solid()
        display, start_display, add_menu, add_function_to_menu = init_display()
        display.DisplayShape(shaft_solid, update=True)
        start_display()
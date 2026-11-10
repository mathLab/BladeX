import os
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Extend.DataExchange import read_stl_file
from OCC.Display.SimpleGui import init_display

from bladex.export import sew_shapes, to_solid

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
        Generate an assembled solid shaft from its IGES/STL file.
        This method requires PythonOCC to be installed.

        :return: the shaft as a solid, or as the sewed shape when it cannot be
            closed into a single shell
        :rtype: OCC.Core.TopoDS.TopoDS_Shape
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
        return to_solid(sew_shapes([shaft_compound], tolerance=1e-2))

    def display(self):
        """
        Display the shaft.
        """
        shaft_solid = self.generate_solid()
        display, start_display = init_display()[:2]
        display.DisplayShape(shaft_solid, update=True)
        start_display()

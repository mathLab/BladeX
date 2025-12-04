import OCC.Core.TopoDS
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2


class CylinderShaft(object):
    """
    Cylinder shaft construction.

    :param float radius: radius of the cylinder shaft. Defaults to 1.0.
    :param float height: height of the cylinder shaft. Defaults to 1.0.
    :param list orientation: orientation vector of the cylinder shaft. Defaults
        to [1.0, 0.0, 0.0], so along X axis.
    :param list origin: origin point of the cylinder shaft. Defaults to 
        [0.0, 0.0, 0.0].
    """

    def __init__(self, radius=1.0, height=1.0, orientation=None, origin=None):
        self.radius = radius
        self.height = height

        if orientation is None:
            self.orientation = [1.0, 0.0, 0.0]  # default orientation along X

        if origin is None:
            self.origin = [0.0, 0.0, 0.0]  # default origin at (0,0,0)

    def generate_solid(self):
        """
        Generate a cylindrical shaft using the BRepBuilderAPI_MakeCylinder  
        algorithm. This method requires PythonOCC to be installed.

        :return: solid shaft
        :rtype: OCC.Core.TopoDS.TopoDS_Solid
        """

        origin = gp_Pnt(*self.origin)
        orientation = gp_Dir(*self.orientation)
        ax2 = gp_Ax2(origin, orientation)

        shape = BRepPrimAPI_MakeCylinder(ax2, self.radius, self.height).Shape()

        return shape

    def display(self):
        """
        Display the shaft.
        """
        shaft_solid = self.generate_solid()
        display, start_display = init_display()[:2]
        display.DisplayShape(shaft_solid, update=True)
        start_display()
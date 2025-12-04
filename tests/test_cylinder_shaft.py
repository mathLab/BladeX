from unittest import TestCase
from bladex import CylinderShaft
from OCC.Core.TopoDS import TopoDS_Solid


def test_generate_solid_01():
    sh = CylinderShaft()
    shaft_solid = sh.generate_solid()
    assert isinstance(shaft_solid, TopoDS_Solid)
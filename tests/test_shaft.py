from unittest import TestCase
import os
from bladex import Shaft
from OCC.Core.TopoDS import TopoDS_Solid

class TestShaft(TestCase):
    """
    Test case for the Shaft class.
    """

    def test_generate_shaft_solid(self):
        sh = Shaft("tests/test_datasets/shaft.iges")
        shaft_solid = sh.generate_shaft_solid()
        self.assertIsInstance(shaft_solid, TopoDS_Solid)
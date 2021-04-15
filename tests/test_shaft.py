from unittest import TestCase
import os
from bladex import Shaft
from OCC.Core.TopoDS import TopoDS_Solid

class TestShaft(TestCase):
    """
    Test case for the Shaft class.
    """

    def test_generate_solid(self):
        sh = Shaft("tests/test_datasets/shaft.iges")
        shaft_solid = sh.generate_solid()
        self.assertIsInstance(shaft_solid, TopoDS_Solid)

    def test_display(self):
        sh = Shaft("tests/test_datasets/shaft.iges")
        sh.display()
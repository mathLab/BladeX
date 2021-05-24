from unittest import TestCase
import os
from bladex import Shaft
from OCC.Core.TopoDS import TopoDS_Solid

class TestShaft(TestCase):
    """
    Test case for the Shaft class.
    """

    def test_generate_solid_01(self):
        sh = Shaft("tests/test_datasets/shaft.iges")
        shaft_solid = sh.generate_solid()
        self.assertIsInstance(shaft_solid, TopoDS_Solid)

    def test_generate_solid_02(self):
        sh = Shaft("tests/test_datasets/shaft.stl")
        shaft_solid = sh.generate_solid()
        self.assertIsInstance(shaft_solid, TopoDS_Solid)

    def test_display_01(self):
        sh = Shaft("tests/test_datasets/shaft.iges")
        sh.display()

    def test_display_02(self):
        sh = Shaft("tests/test_datasets/shaft.stl")
        sh.display()

    def test_exception(self):
    	sh = Shaft("tests/test_datasets/parameters.prm")
    	with self.assertRaises(Exception):
            sh.generate_solid()

    def test_init(self):
        sh = Shaft("tests/test_datasets/shaft.iges")
        assert sh.filename == "tests/test_datasets/shaft.iges"

from unittest import TestCase
import unittest
import pkgutil
from os import walk
from os import path
import numpy as np

class TestPackage(TestCase):
    def test_import_blade_1(self):
        from bladex import ProfileBase
        profile = ProfileBase()

    def test_import_blade_2(self):
        from bladex import CustomProfile
        vec = np.array([1.2, 2.4])
        profile = CustomProfile(xup=vec, yup=vec, xdown=vec, ydown=vec)

    def test_import_blade_3(self):
        from bladex import NacaProfile
        profile = NacaProfile('0012')

    def test_import_blade_4(self):
        from bladex import NacaProfile, Blade
        profile = NacaProfile('0012')
        sections = [profile, profile]
        radii = np.array([1.0, 2.0])
        chord = np.array([0.5, 1.0])
        pitch = np.array([0.2, 0.3])
        rake = np.array([0.2, 0.3])
        skew = np.array([5.0, 10.0])
        blade = Blade(sections, radii, chord, pitch, rake, skew)

    def test_import_blade_5(self):
        from bladex import RBF
        inter = RBF('gaussian_spline', 1.2)

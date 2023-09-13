import os
from unittest import TestCase
from bladex import ProfileInterface
import numpy as np
import matplotlib.pyplot as plt

class TestProfileInterface(TestCase):
    def test_instantiation(self):
        with self.assertRaises(TypeError):
            ProfileInterface()
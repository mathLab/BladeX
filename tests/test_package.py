from unittest import TestCase
import unittest
import pkgutil
from os import walk
from os import path


class TestPackage(TestCase):
    def test_import_blade_1(self):
        import bladex as bx
        profile = bx.profilebase.ProfileBase()

    def test_modules_name(self):
        # it checks that __all__ includes all the .py files in bladex folder
        import bladex
        package = bladex

        f_aux = []
        for (__, __, filenames) in walk('bladex'):
            f_aux.extend(filenames)

        f = []
        for i in f_aux:
            file_name, file_ext = path.splitext(i)
            if file_name != '__init__' and file_ext == '.py':
                f.append(file_name)

        print(f)
        print(package.__all__)
        assert sorted(package.__all__) == sorted(f)

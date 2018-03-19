from unittest import TestCase
import unittest
import bladex.profilebase as pb


class TestProfileBase(TestCase):
    def test_xup_init(self):
        profile = pb.ProfileBase()
        assert profile.xup_coordinates is None

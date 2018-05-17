from unittest import TestCase
import bladex.profiles as pr
import bladex.profilebase as pb
import numpy as np


def create_custom_profile():
    """
    xup_coordinates, yup_coordinates, xdown_coordinates, ydown_coordinates
    """
    xup = np.linspace(-1.0, 1.0, 5)
    yup = np.array([0.0, 0.75, 1.0, 0.75, 0.0])
    xdown = np.linspace(-1.0, 1.0, 5)
    ydown = np.zeros(5)
    return pr.CustomProfile(xup=xup, yup=yup, xdown=xdown, ydown=ydown)


class TestCustomProfile(TestCase):
    def test_inheritance_custom(self):
        self.assertTrue(issubclass(pr.CustomProfile, pb.ProfileBase))

    def test_xup_member(self):
        profile = create_custom_profile()
        np.testing.assert_equal(profile.xup_coordinates,
                                np.linspace(-1.0, 1.0, 5))

    def test_yup_member(self):
        profile = create_custom_profile()
        np.testing.assert_equal(profile.yup_coordinates,
                                np.array([0.0, 0.75, 1.0, 0.75, 0.0]))

    def test_xdown_member(self):
        profile = create_custom_profile()
        np.testing.assert_equal(profile.xdown_coordinates,
                                np.linspace(-1.0, 1.0, 5))

    def test_ydown_member(self):
        profile = create_custom_profile()
        np.testing.assert_equal(profile.ydown_coordinates, np.zeros(5))

    def test_xup_None(self):
        profile = create_custom_profile()
        profile.xup_coordinates = None
        with self.assertRaises(ValueError):
            profile._check_coordinates()

    def test_yup_None(self):
        profile = create_custom_profile()
        profile.yup_coordinates = None
        with self.assertRaises(ValueError):
            profile._check_coordinates()

    def test_xdown_None(self):
        profile = create_custom_profile()
        profile.xdown_coordinates = None
        with self.assertRaises(ValueError):
            profile._check_coordinates()

    def test_ydown_None(self):
        profile = create_custom_profile()
        profile.ydown_coordinates = None
        with self.assertRaises(ValueError):
            profile._check_coordinates()

    def test_xup_not_ndarray(self):
        profile = create_custom_profile()
        profile.xup_coordinates = [-1.0, -0.5, 0.0, 0.5, 1.0]
        profile._check_coordinates()
        self.assertIsInstance(profile.xup_coordinates, np.ndarray)

    def test_yup_not_ndarray(self):
        profile = create_custom_profile()
        profile.yup_coordinates = [0.0, 0.75, 1.0, 0.75, 0.0]
        profile._check_coordinates()
        self.assertIsInstance(profile.yup_coordinates, np.ndarray)

    def test_xdown_not_ndarray(self):
        profile = create_custom_profile()
        profile.xdown_coordinates = [-1.0, -0.5, 0.0, 0.5, 1.0]
        profile._check_coordinates()
        self.assertIsInstance(profile.xdown_coordinates, np.ndarray)

    def test_ydown_not_ndarray(self):
        profile = create_custom_profile()
        profile.ydown_coordinates = [0.0, 0.0, 0.0, 0.0, 0.0]
        profile._check_coordinates()
        self.assertIsInstance(profile.ydown_coordinates, np.ndarray)

    def test_upper_profile_not_same_shape(self):
        profile = create_custom_profile()
        profile.xup_coordinates = np.array([1.0, 2.0])
        with self.assertRaises(ValueError):
            profile._check_coordinates()

    def test_lower_profile_not_same_shape(self):
        profile = create_custom_profile()
        profile.xdown_coordinates = np.array([1.0, 2.0])
        with self.assertRaises(ValueError):
            profile._check_coordinates()

    def test_yup_lower_ydown(self):
        profile = create_custom_profile()
        profile.yup_coordinates *= -1.0
        with self.assertRaises(ValueError):
            profile._check_coordinates()

    def test_leading_edge_x(self):
        profile = create_custom_profile()
        profile.xup_coordinates[0] = -1.5
        with self.assertRaises(ValueError):
            profile._check_coordinates()

    def test_leading_edge_y(self):
        profile = create_custom_profile()
        profile.yup_coordinates[0] = 1.5
        with self.assertRaises(ValueError):
            profile._check_coordinates()

    def test_trailing_edge_x(self):
        profile = create_custom_profile()
        profile.xup_coordinates[-1] = -1.5
        with self.assertRaises(ValueError):
            profile._check_coordinates()


class TestNacaProfile(TestCase):
    def test_inheritance_naca(self):
        self.assertTrue(issubclass(pr.NacaProfile, pb.ProfileBase))

    def test_digits_member(self):
        profile = pr.NacaProfile(digits='0012', n_points=240)
        assert profile.digits == '0012'

    def test_npoints_member(self):
        profile = pr.NacaProfile(digits='0012', n_points=240)
        assert profile.n_points == 240

    def test_digits_not_entered(self):
        with self.assertRaises(TypeError):
            pr.NacaProfile()

    def test_digits_not_string(self):
        with self.assertRaises(TypeError):
            pr.NacaProfile(digits=2412, n_points=240)

    def test_npoints_float_1(self):
        profile = pr.NacaProfile(digits='2412', n_points=240.0)
        self.assertIsInstance(profile.n_points, int)

    def test_npoints_float_2(self):
        profile = pr.NacaProfile(digits='2412', n_points=240.5)
        assert profile.n_points == 240

    def test_npoints_not_number(self):
        with self.assertRaises(TypeError):
            pr.NacaProfile(digits='0012', n_points='240')

    def test_npoints_negative(self):
        with self.assertRaises(ValueError):
            pr.NacaProfile(digits='0012', n_points=-240)

    def test_naca_series_not_implemented_1(self):
        with self.assertRaises(Exception):
            pr.NacaProfile(digits='1234-05', n_points=240)

    def test_naca_series_not_implemented_2(self):
        with self.assertRaises(Exception):
            pr.NacaProfile(digits='16-123', n_points=240)

    def test_naca_series_not_implemented_3(self):
        with self.assertRaises(Exception):
            pr.NacaProfile(digits='712A315', n_points=240)

    def test_naca_4_symmetric_xup(self):
        foil = pr.NacaProfile(digits='0012', n_points=240)
        xup_expected = np.load('tests/test_datasets/naca4_0012_xup.npy')
        np.testing.assert_almost_equal(foil.xup_coordinates, xup_expected)

    def test_naca_4_symmetric_yup(self):
        foil = pr.NacaProfile(digits='0012', n_points=240)
        yup_expected = np.load('tests/test_datasets/naca4_0012_yup.npy')
        np.testing.assert_almost_equal(foil.yup_coordinates, yup_expected)

    def test_naca_4_symmetric_xdown(self):
        foil = pr.NacaProfile(digits='0012', n_points=240)
        xdown_expected = np.load('tests/test_datasets/naca4_0012_xdown.npy')
        np.testing.assert_almost_equal(foil.xdown_coordinates, xdown_expected)

    def test_naca_4_symmetric_ydown(self):
        foil = pr.NacaProfile(digits='0012', n_points=240)
        ydown_expected = np.load('tests/test_datasets/naca4_0012_ydown.npy')
        np.testing.assert_almost_equal(foil.ydown_coordinates, ydown_expected)

    def test_naca_4_cambered_xup_cosine(self):
        foil = pr.NacaProfile(digits='2412', n_points=240, cosine_spacing=True)
        xup_expected = np.load('tests/test_datasets/naca4_2412_xup_cosine.npy')
        np.testing.assert_almost_equal(foil.xup_coordinates, xup_expected)

    def test_naca_4_cambered_yup_cosine(self):
        foil = pr.NacaProfile(digits='2412', n_points=240, cosine_spacing=True)
        yup_expected = np.load('tests/test_datasets/naca4_2412_yup_cosine.npy')
        np.testing.assert_almost_equal(foil.yup_coordinates, yup_expected)

    def test_naca_4_cambered_xdown_cosine(self):
        foil = pr.NacaProfile(digits='2412', n_points=240, cosine_spacing=True)
        xdown_expected = np.load(
            'tests/test_datasets/naca4_2412_xdown_cosine.npy')
        np.testing.assert_almost_equal(foil.xdown_coordinates, xdown_expected)

    def test_naca_4_cambered_ydown_cosine(self):
        foil = pr.NacaProfile(digits='2412', n_points=240, cosine_spacing=True)
        ydown_expected = np.load(
            'tests/test_datasets/naca4_2412_ydown_cosine.npy')
        np.testing.assert_almost_equal(foil.ydown_coordinates, ydown_expected)

    def test_naca_4_cambered_xup_linear(self):
        foil = pr.NacaProfile(digits='2412', n_points=240, cosine_spacing=False)
        xup_expected = np.load('tests/test_datasets/naca4_2412_xup_linear.npy')
        np.testing.assert_almost_equal(foil.xup_coordinates, xup_expected)

    def test_naca_4_cambered_yup_linear(self):
        foil = pr.NacaProfile(digits='2412', n_points=240, cosine_spacing=False)
        yup_expected = np.load('tests/test_datasets/naca4_2412_yup_linear.npy')
        np.testing.assert_almost_equal(foil.yup_coordinates, yup_expected)

    def test_naca_4_cambered_xdown_linear(self):
        foil = pr.NacaProfile(digits='2412', n_points=240, cosine_spacing=False)
        xdown_expected = np.load(
            'tests/test_datasets/naca4_2412_xdown_linear.npy')
        np.testing.assert_almost_equal(foil.xdown_coordinates, xdown_expected)

    def test_naca_4_cambered_ydown_linear(self):
        foil = pr.NacaProfile(digits='2412', n_points=240, cosine_spacing=False)
        ydown_expected = np.load(
            'tests/test_datasets/naca4_2412_ydown_linear.npy')
        np.testing.assert_almost_equal(foil.ydown_coordinates, ydown_expected)

    def test_naca_5_wrong_reflex(self):
        with self.assertRaises(ValueError):
            pr.NacaProfile(digits='24512', n_points=240)

    def test_naca_5_symmetric_xup(self):
        foil = pr.NacaProfile(digits='00012', n_points=240)
        xup_expected = np.load('tests/test_datasets/naca5_00012_xup.npy')
        np.testing.assert_almost_equal(foil.xup_coordinates, xup_expected)

    def test_naca_5_symmetric_yup(self):
        foil = pr.NacaProfile(digits='00012', n_points=240)
        yup_expected = np.load('tests/test_datasets/naca5_00012_yup.npy')
        np.testing.assert_almost_equal(foil.yup_coordinates, yup_expected)

    def test_naca_5_symmetric_xdown(self):
        foil = pr.NacaProfile(digits='00012', n_points=240)
        xdown_expected = np.load('tests/test_datasets/naca5_00012_xdown.npy')
        np.testing.assert_almost_equal(foil.xdown_coordinates, xdown_expected)

    def test_naca_5_symmetric_ydown(self):
        foil = pr.NacaProfile(digits='00012', n_points=240)
        ydown_expected = np.load('tests/test_datasets/naca5_00012_ydown.npy')
        np.testing.assert_almost_equal(foil.ydown_coordinates, ydown_expected)

    def test_naca_5_cambered_standard_xup_cosine(self):
        foil = pr.NacaProfile(digits='23012', n_points=240, cosine_spacing=True)
        xup_expected = np.load('tests/test_datasets/naca5_23012_xup_cosine.npy')
        np.testing.assert_almost_equal(foil.xup_coordinates, xup_expected)

    def test_naca_5_cambered_standard_yup_cosine(self):
        foil = pr.NacaProfile(digits='23012', n_points=240, cosine_spacing=True)
        yup_expected = np.load('tests/test_datasets/naca5_23012_yup_cosine.npy')
        np.testing.assert_almost_equal(foil.yup_coordinates, yup_expected)

    def test_naca_5_cambered_standard_xdown_cosine(self):
        foil = pr.NacaProfile(digits='23012', n_points=240, cosine_spacing=True)
        xdown_expected = np.load(
            'tests/test_datasets/naca5_23012_xdown_cosine.npy')
        np.testing.assert_almost_equal(foil.xdown_coordinates, xdown_expected)

    def test_naca_5_cambered_standard_ydown_cosine(self):
        foil = pr.NacaProfile(digits='23012', n_points=240, cosine_spacing=True)
        ydown_expected = np.load(
            'tests/test_datasets/naca5_23012_ydown_cosine.npy')
        np.testing.assert_almost_equal(foil.ydown_coordinates, ydown_expected)

    def test_naca_5_cambered_standard_xup_linear(self):
        foil = pr.NacaProfile(
            digits='23012', n_points=240, cosine_spacing=False)
        xup_expected = np.load('tests/test_datasets/naca5_23012_xup_linear.npy')
        np.testing.assert_almost_equal(foil.xup_coordinates, xup_expected)

    def test_naca_5_cambered_standard_yup_linear(self):
        foil = pr.NacaProfile(
            digits='23012', n_points=240, cosine_spacing=False)
        yup_expected = np.load('tests/test_datasets/naca5_23012_yup_linear.npy')
        np.testing.assert_almost_equal(foil.yup_coordinates, yup_expected)

    def test_naca_5_cambered_standard_xdown_linear(self):
        foil = pr.NacaProfile(
            digits='23012', n_points=240, cosine_spacing=False)
        xdown_expected = np.load(
            'tests/test_datasets/naca5_23012_xdown_linear.npy')
        np.testing.assert_almost_equal(foil.xdown_coordinates, xdown_expected)

    def test_naca_5_cambered_standard_ydown_linear(self):
        foil = pr.NacaProfile(
            digits='23012', n_points=240, cosine_spacing=False)
        ydown_expected = np.load(
            'tests/test_datasets/naca5_23012_ydown_linear.npy')
        np.testing.assert_almost_equal(foil.ydown_coordinates, ydown_expected)

    def test_naca_5_cambered_reflexed_xup(self):
        foil = pr.NacaProfile(digits='23112', n_points=240)
        xup_expected = np.load('tests/test_datasets/naca5_23112_xup.npy')
        np.testing.assert_almost_equal(foil.xup_coordinates, xup_expected)

    def test_naca_5_cambered_reflexed_yup(self):
        foil = pr.NacaProfile(digits='23112', n_points=240)
        yup_expected = np.load('tests/test_datasets/naca5_23112_yup.npy')
        np.testing.assert_almost_equal(foil.yup_coordinates, yup_expected)

    def test_naca_5_cambered_reflexed_xdown(self):
        foil = pr.NacaProfile(digits='23112', n_points=240)
        xdown_expected = np.load('tests/test_datasets/naca5_23112_xdown.npy')
        np.testing.assert_almost_equal(foil.xdown_coordinates, xdown_expected)

    def test_naca_5_cambered_reflexed_ydown(self):
        foil = pr.NacaProfile(digits='23112', n_points=240)
        ydown_expected = np.load('tests/test_datasets/naca5_23112_ydown.npy')
        np.testing.assert_almost_equal(foil.ydown_coordinates, ydown_expected)

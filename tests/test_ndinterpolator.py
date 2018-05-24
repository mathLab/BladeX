from unittest import TestCase
import bladex.ndinterpolator as nd
import numpy as np

def sample_data():
    x = np.arange(10)
    y = x*x
    xx = np.linspace(0,9,1000)
    yy = np.zeros(1000)

    return x, y, xx, yy

class TestRBF(TestCase):
    def test_gaussian_basis_member(self):
        rbf = nd.RBF(basis='gaussian_spline', radius=1.)
        assert rbf.basis == rbf.gaussian_spline

    def test_biharmonic_basis_member(self):
        rbf = nd.RBF(basis='multi_quadratic_biharmonic_spline', radius=1.)
        assert rbf.basis == rbf.multi_quadratic_biharmonic_spline

    def test_inv_biharmonic_basis_member(self):
        rbf = nd.RBF(basis='inv_multi_quadratic_biharmonic_spline', radius=1.)
        assert rbf.basis == rbf.inv_multi_quadratic_biharmonic_spline

    def test_thin_plate_basis_member(self):
        rbf = nd.RBF(basis='thin_plate_spline', radius=1.)
        assert rbf.basis == rbf.thin_plate_spline

    def test_wendland_basis_member(self):
        rbf = nd.RBF(basis='beckert_wendland_c2_basis', radius=1.)
        assert rbf.basis == rbf.beckert_wendland_c2_basis

    def test_radius_member(self):
        rbf = nd.RBF(basis='beckert_wendland_c2_basis', radius=1.)
        assert rbf.radius == 1.0

    def test_wrong_basis(self):
        with self.assertRaises(NameError):
            rbf = nd.RBF(basis='wendland', radius=1.)

    def test_gaussian_evaluation(self):
        rbf = nd.RBF(basis='gaussian_spline', radius=1.)
        result = rbf.basis(X=1., r=1.)
        assert result == 0.36787944117144233

    def test_biharmonic_evaluation(self):
        rbf = nd.RBF(basis='multi_quadratic_biharmonic_spline', radius=1.)
        result = rbf.basis(X=1., r=1.)
        assert result == 1.4142135623730951

    def test_inv_biharmonic_evaluation(self):
        rbf = nd.RBF(basis='inv_multi_quadratic_biharmonic_spline', radius=1.)
        result = rbf.basis(X=1., r=1.)
        assert result == 0.7071067811865475

    def test_thin_plate_evaluation(self):
        rbf = nd.RBF(basis='thin_plate_spline', radius=1.)
        result = rbf.basis(X=1., r=0.5)
        assert result == 2.772588722239781

    def test_wendland_evaluation(self):
        rbf = nd.RBF(basis='beckert_wendland_c2_basis', radius=1.)
        result = rbf.basis(X=1., r=2.)
        assert result == 0.1875

    def test_wendland_outside_cutoff(self):
        rbf = nd.RBF(basis='beckert_wendland_c2_basis', radius=1.)
        result = rbf.basis(X=2., r=1.)
        assert result == 0.0

    def test_weight_matrix(self):
        x = np.arange(10)
        rbf = nd.RBF(basis='beckert_wendland_c2_basis', radius=1.)
        weights_matrix = rbf.weights_matrix(X1=x, X2=x)
        expected = np.diag(np.ones(10))
        np.testing.assert_array_equal(weights_matrix, expected)

    def test_reconstruct_f_gaussian(self):
        x, y, xx, yy = sample_data()
        nd.reconstruct_f(original_input=x, original_output=y, rbf_input=xx, rbf_output=yy, basis='gaussian_spline',
                          radius=10.)
        # assert that the argmin(xx) nearest to point x=4.5 is the same index
        # for argmin(yy) nearest to point y=20.25, where y=x*x
        idx = (np.abs(xx - 4.5)).argmin()
        idx2 = (np.abs(yy - 20.25)).argmin()
        assert idx == idx2

    def test_reconstruct_f_biharmonic(self):
        x, y, xx, yy = sample_data()
        nd.reconstruct_f(original_input=x, original_output=y, rbf_input=xx, rbf_output=yy, basis='multi_quadratic_biharmonic_spline',
                          radius=10.)
        # assert that the argmin(xx) nearest to point x=4.5 is the same index
        # for argmin(yy) nearest to point y=20.25, where y=x*x
        idx = (np.abs(xx - 4.5)).argmin()
        idx2 = (np.abs(yy - 20.25)).argmin()
        assert idx == idx2

    def test_reconstruct_f_inv_biharmonic(self):
        x, y, xx, yy = sample_data()
        nd.reconstruct_f(original_input=x, original_output=y, rbf_input=xx, rbf_output=yy, basis='inv_multi_quadratic_biharmonic_spline',
                          radius=10.)
        # assert that the argmin(xx) nearest to point x=4.5 is the same index
        # for argmin(yy) nearest to point y=20.25, where y=x*x
        idx = (np.abs(xx - 4.5)).argmin()
        idx2 = (np.abs(yy - 20.25)).argmin()
        assert idx == idx2

    def test_reconstruct_f_plate(self):
        x, y, xx, yy = sample_data()
        nd.reconstruct_f(original_input=x, original_output=y, rbf_input=xx, rbf_output=yy, basis='thin_plate_spline',
                          radius=20.)
        # assert that the argmin(xx) nearest to point x=4.5 is the same index
        # for argmin(yy) nearest to point y=20.25, where y=x*x
        idx = (np.abs(xx - 4.5)).argmin()
        idx2 = (np.abs(yy - 20.25)).argmin()
        assert idx == idx2

    def test_reconstruct_f_wendland(self):
        x, y, xx, yy = sample_data()
        nd.reconstruct_f(original_input=x, original_output=y, rbf_input=xx, rbf_output=yy, basis='beckert_wendland_c2_basis',
                          radius=20.)
        # assert that the argmin(xx) nearest to point x=4.5 is the same index
        # for argmin(yy) nearest to point y=20.25, where y=x*x
        idx = (np.abs(xx - 4.5)).argmin()
        idx2 = (np.abs(yy - 20.25)).argmin()
        assert idx == idx2

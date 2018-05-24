import numpy as np
from scipy.spatial.distance import cdist

class RBF(object):
    """
    Module focused on the implementation of the Radial Basis Functions
    interpolation technique.  This technique is still based on the use of
    a set of parameters, the so-called control points, as for FFD, but RBF
    is interpolatory. Another important key point of RBF strategy relies in
    the way we can locate the control points: in fact, instead of FFD where
    control points need to be placed inside a regular lattice, with RBF we
    havo no more limitations. So we have the possibility to perform localized
    control points refiniments.

    :Theoretical Insight:
        As reference please consult M.D. Buhmann, Radial Basis Functions,
        volume 12 of Cambridge monographs on applied and computational
        mathematics. Cambridge University Press, UK, 2003.  This implementation
        follows D. Forti and G. Rozza, Efficient geometrical parametrization
        techniques of interfaces for reduced order modelling: application to
        fluid-structure interaction coupling problems, International Journal
        of Computational Fluid Dynamics.
        
        RBF shape parametrization technique is based on the definition of a map
        :math:`\\mathcal{M}(\\boldsymbol{x}) : \\mathbb{R}^n \\rightarrow
        \\mathbb{R}^n`, that allows the possibility of transferring data across
        non-matching grids and facing the dynamic mesh handling. The map
        introduced is defines as follows
        .. math::
            \\mathcal{M}(\\boldsymbol{x}) = p(\\boldsymbol{x}) + 
            \\sum_{i=1}^{\\mathcal{N}_C} \\gamma_i
            \\varphi(\\| \\boldsymbol{x} - \\boldsymbol{x_{C_i}} \\|)
        where :math:`p(\\boldsymbol{x})` is a low_degree polynomial term,
        :math:`\\gamma_i` is the weight, corresponding to the a-priori selected
        :math:`\\mathcal{N}_C` control points, associated to the :math:`i`-th
        basis function, and :math:`\\varphi(\\| \\boldsymbol{x} -
        \\boldsymbol{x_{C_i}} \\|)` a radial function based on the Euclidean
        distance between the control points position :math:`\\boldsymbol{x_{C_i}}`
        and :math:`\\boldsymbol{x}`. A radial basis function, generally, is a
        real-valued function whose value depends only on the distance from the
        origin, so that :math:`\\varphi(\\boldsymbol{x}) = \\tilde{\\varphi}(
        \\|\\boldsymbol{x} \\|)`.

        The matrix version of the formula above is:
        .. math::
            \\mathcal{M}(\\boldsymbol{x}) = \\boldsymbol{c} +
            \\boldsymbol{Q}\\boldsymbol{x} +
            \\boldsymbol{W^T}\\boldsymbol{d}(\\boldsymbol{x})
        The idea is that after the computation of the weights and the
        polynomial terms from the coordinates of the control points before and
        after the deformation, we can deform all the points of the mesh
        accordingly. Among the most common used radial basis functions for
        modelling 2D and 3D shapes, we consider Gaussian splines, Multi-
        uadratic biharmonic splines, Inverted multi-quadratic biharmonic
        splines, Thin-plate splines, Beckert and Wendland :math:`C^2` basis all
        defined and implemented below.

    :param string basis: RBF basis function
    :param float radius: cut-off radius

   :Example:
    >>> import numpy as np
    >>> from bladex.ndinterpolator import reconstruct_f
    >>> x = np.arange(10)
    >>> y = np.square(x)
    >>> radius = 10
    >>> n_interp = 50
    >>> x_rbf = np.linspace(x[0], x[-1], num=n_interp)
    >>> y_rbf = np.zeros(n_interp)
    >>> reconstruct_f(original_input=x, original_output=y, rbf_input=x_rbf,
        rbf_output=y_rbf, radius=radius, basis='beckert_wendland_c2_basis')
    """

    def __init__(self, basis, radius):
        self.bases = {
            'gaussian_spline':
            self.gaussian_spline,
            'multi_quadratic_biharmonic_spline':
            self.multi_quadratic_biharmonic_spline,
            'inv_multi_quadratic_biharmonic_spline':
            self.inv_multi_quadratic_biharmonic_spline,
            'thin_plate_spline':
            self.thin_plate_spline,
            'beckert_wendland_c2_basis':
            self.beckert_wendland_c2_basis
        }

        if basis in self.bases:
            self.basis = self.bases[basis]
        else:
            raise NameError(
                """The name of the basis function is not correct. Check 
                the documentation for all the available functions.""")

        self.radius = radius

    # The following static methods are the implementations
    # of the most common radial basis functions
    @staticmethod
    def gaussian_spline(X, r):
        """
        It implements the following formula:

        .. math::
            \\varphi(\\| \\boldsymbol{x} \\|) =
            e^{-\\frac{\\| \\boldsymbol{x} \\|^2}{r^2}}

        :param numpy.ndarray X: l2-norm between given inputs of a function
            and the locations to perform rbf approximation to that function.
        :param float r: smoothing length, also called the cut-ff radius.

        :return: result: the result of the formula above.
        :rtype: float
        """
        return np.exp(-(X * X) / (r * r))

    @staticmethod
    def multi_quadratic_biharmonic_spline(X, r):
        """
        It implements the following formula:

        .. math::
            \\varphi(\\| \\boldsymbol{x} \\|) =
            \\sqrt{\\| \\boldsymbol{x} \\|^2 + r^2}

        :param numpy.ndarray X: l2-norm between given inputs of a function
            and the locations to perform rbf approximation to that function.
        :param float r: smoothing length, also called the cut-ff radius.

        :return: result: the result of the formula above.
        :rtype: float
        """
        return np.sqrt((X * X) + (r * r))

    @staticmethod
    def inv_multi_quadratic_biharmonic_spline(X, r):
        """
        It implements the following formula:

        .. math::
            \\varphi(\\| \\boldsymbol{x} \\|) =
            (\\| \\boldsymbol{x} \\|^2 + r^2 )^{-\\frac{1}{2}}

        :param numpy.ndarray X: l2-norm between given inputs of a function
            and the locations to perform rbf approximation to that function.
        :param float r: smoothing length, also called the cut-ff radius.

        :return: result: the result of the formula above.
        :rtype: float
        """
        return 1.0 / (np.sqrt((X * X) + (r * r)))

    @staticmethod
    def thin_plate_spline(X, r):
        """
        It implements the following formula:

        .. math::
            \\varphi(\\| \\boldsymbol{x} \\|) =
            \\left\\| \\frac{\\boldsymbol{x} }{r} \\right\\|^2
            \\ln \\left\\| \\frac{\\boldsymbol{x} }{r} \\right\\|

        :param numpy.ndarray X: l2-norm between given inputs of a function
            and the locations to perform rbf approximation to that function.
        :param float r: smoothing length, also called the cut-ff radius.

        :return: result: the result of the formula above.
        :rtype: float
        """
        arg = X / r
        result = arg * arg
        result = np.where(arg > 0, result * np.log(arg), result)
        return result

    @staticmethod
    def beckert_wendland_c2_basis(X, r):
        """
        It implements the following formula:

        .. math::
            \\varphi(\\| \\boldsymbol{x} \\|) =
            \\left( 1 - \\frac{\\| \\boldsymbol{x} \\|}{r} \\right)^4 +
            \\left( 4 \\frac{\\| \\boldsymbol{x} \\|}{r} + 1 \\right)

        :param numpy.ndarray X: l2-norm between given inputs of a function
            and the locations to perform rbf approximation to that function.
        :param float r: smoothing length, also called the cut-ff radius.

        :return: result: the result of the formula above.
        :rtype: float
        """
        arg = X / r
        first = np.where((1 - arg) > 0, np.power((1 - arg), 4), 0)
        second = (4 * arg) + 1
        return first * second

    def weights_matrix(self, X1, X2):
        """
        This method returns the following matrix:
        :math:`\\boldsymbol{D_{ij}} = \\varphi(\\| \\boldsymbol{x_i} -
                                      \\boldsymbol{y_j} \\|)`

        :param numpy.ndarray X1: the vector x in the formula above.
        :param numpy.ndarray X2: the vector y in the formula above.

        :return: matrix: the matrix D.
        :rtype: numpy.ndarray
        """
        XA1 = X1.reshape(-1,1)
        XA2 = X2.reshape(-1,1)
        return self.basis(cdist(XA1, XA2), self.radius)


def reconstruct_f(original_input, original_output, rbf_input, rbf_output, basis,
                  radius):
    """
    Reconstruct a function by using the radial basis function approximations.

    :param array_like original_input: the original values of function inputs.
    :param array_like original_output: the original values of function output.
    :param array_like rbf_input: the input data for RBF approximation.
    :param array_like rbf_output: the array elements to be updated with the RBF
        interpolated outputs after the approximation.
    :param string basis: radial basis function.
    :param float radius: smoothing length, also called the cut-ff radius.
    """
    radial = RBF(basis=basis, radius=radius)

    weights_coeff = np.linalg.solve(radial.weights_matrix(original_input, original_input), original_output)

    weights_rbf = radial.weights_matrix(rbf_input, original_input)

    for i in range(rbf_input.shape[0]):
        for j in range(original_input.shape[0]):
            rbf_output[i] += weights_coeff[j] * weights_rbf[i][j]

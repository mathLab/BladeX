import numpy as np


class RBF(object):
    """
    Module focused on the implementation of the Radial Basis Functions interpolation
    technique.  This technique is still based on the use of a set of parameters, the
    so-called control points, as for FFD, but RBF is interpolatory. Another
    important key point of RBF strategy relies in the way we can locate the control
    points: in fact, instead of FFD where control points need to be placed inside a
    regular lattice, with RBF we hano no more limitations. So we have the
    possibility to perform localized control points refiniments.
    The module is analogous to the freeform one.

    :Theoretical Insight:
        As reference please consult M.D. Buhmann, Radial Basis Functions, volume 12
        of Cambridge monographs on applied and computational mathematics. Cambridge
        University Press, UK, 2003.  This implementation follows D. Forti and G.
        Rozza, Efficient geometrical parametrization techniques of interfaces for
        reduced order modelling: application to fluid-structure interaction coupling
        problems, International Journal of Computational Fluid Dynamics.
        
        RBF shape parametrization technique is based on the definition of a map,
        :math:`\\mathcal{M}(\\boldsymbol{x}) : \\mathbb{R}^n \\rightarrow
        \\mathbb{R}^n`, that allows the possibility of transferring data across
        non-matching grids and facing the dynamic mesh handling. The map introduced
        is defines as follows
        .. math::
            \\mathcal{M}(\\boldsymbol{x}) = p(\\boldsymbol{x}) + 
            \\sum_{i=1}^{\\mathcal{N}_C} \\gamma_i
            \\varphi(\\| \\boldsymbol{x} - \\boldsymbol{x_{C_i}} \\|)
        where :math:`p(\\boldsymbol{x})` is a low_degree polynomial term,
        :math:`\\gamma_i` is the weight, corresponding to the a-priori selected
        :math:`\\mathcal{N}_C` control points, associated to the :math:`i`-th basis
        function, and :math:`\\varphi(\\| \\boldsymbol{x} - \\boldsymbol{x_{C_i}}
                \\|)` a radial function based on the Euclidean distance between the
        control points position :math:`\\boldsymbol{x_{C_i}}` and
        :math:`\\boldsymbol{x}`. A radial basis function, generally, is a
        real-valued function whose value depends only on the distance from the
        origin, so that :math:`\\varphi(\\boldsymbol{x}) = \\tilde{\\varphi}(\\|
                \\boldsymbol{x} \\|)`.
        The matrix version of the formula above is:
        .. math::
            \\mathcal{M}(\\boldsymbol{x}) = \\boldsymbol{c} +
            \\boldsymbol{Q}\\boldsymbol{x} +
            \\boldsymbol{W^T}\\boldsymbol{d}(\\boldsymbol{x})
        The idea is that after the computation of the weights and the polynomial
        terms from the coordinates of the control points before and after the
        deformation, we can deform all the points of the mesh accordingly.  Among
        the most common used radial basis functions for modelling 2D and 3D shapes,
        we consider Gaussian splines, Multi-quadratic biharmonic splines, Inverted
        multi-quadratic biharmonic splines, Thin-plate splines, Beckert and
        Wendland :math:`C^2` basis and Polyharmonic splines all defined and
        implemented below.

   :Example:
    >>> import numpy as np
    >>> from bladex.utils.rbf import reconstruct_f
    >>> x = np.arange(10)
    >>> y = np.square(x)
    >>> radius = 10
    >>> n_interp = 50
    >>> x_rbf = np.linspace(x[0], x[-1], num=n_interp)
    >>> y_rbf = np.zeros(n_interp)
    >>> reconstruct_f(original_input=x, original_output=y, rbf_input=x_rbf, rbf_output=y_rbf, radius=radius, basis='beckert_wendland_c2_basis')
    """

    def __init__(self, radius, basis):
        self.radius = radius

        self.bases = {
            'gaussian_spline': self.gaussian_spline,
            'multi_quadratic_biharmonic_spline':
            self.multi_quadratic_biharmonic_spline,
            'inv_multi_quadratic_biharmonic_spline':
            self.inv_multi_quadratic_biharmonic_spline,
            'thin_plate_spline': self.thin_plate_spline,
            'beckert_wendland_c2_basis': self.beckert_wendland_c2_basis
        }

        if basis in self.bases:
            self.basis = self.bases[basis]
        else:
            raise NameError(
                """The name of the basis function is not correct. Check 
                the documentation for all the available functions.""")

    @staticmethod
    def gaussian_spline(X, r):
        """
        It implements the following formula:

        .. math::
            \\varphi(\\| \\boldsymbol{x} \\|) = e^{-\\frac{\\| \\boldsymbol{x} \\|^2}{r^2}}

        :param numpy.ndarray X: the vector x in the formula above.
        :param float r: the parameter r in the formula above.

        :return: result: the result of the formula above.
        :rtype: float
        """
        norm = np.linalg.norm(X)
        result = np.exp(-(norm * norm) / (r * r))
        return result

    @staticmethod
    def multi_quadratic_biharmonic_spline(X, r):
        """
        It implements the following formula:

        .. math::
            \\varphi(\\| \\boldsymbol{x} \\|) = \\sqrt{\\| \\boldsymbol{x} \\|^2 + r^2}

        :param numpy.ndarray X: the vector x in the formula above.
        :param float r: the parameter r in the formula above.

        :return: result: the result of the formula above.
        :rtype: float
        """
        norm = np.linalg.norm(X)
        result = np.sqrt((norm * norm) + (r * r))
        return result

    @staticmethod
    def inv_multi_quadratic_biharmonic_spline(X, r):
        """
        It implements the following formula:

        .. math::
            \\varphi(\\| \\boldsymbol{x} \\|) = (\\| \\boldsymbol{x} \\|^2 + r^2 )^{-\\frac{1}{2}}

        :param numpy.ndarray X: the vector x in the formula above.
        :param float r: the parameter r in the formula above.

        :return: result: the result of the formula above.
        :rtype: float
        """
        norm = np.linalg.norm(X)
        result = 1.0 / (np.sqrt((norm * norm) + (r * r)))
        return result

    @staticmethod
    def thin_plate_spline(X, r):
        """
        It implements the following formula:

        .. math::
            \\varphi(\\| \\boldsymbol{x} \\|) = \\left\\| \\frac{\\boldsymbol{x} }{r} \\right\\|^2
            \\ln \\left\\| \\frac{\\boldsymbol{x} }{r} \\right\\|

        :param numpy.ndarray X: the vector x in the formula above.
        :param float r: the parameter r in the formula above.

        :return: result: the result of the formula above.
        :rtype: float
        """
        arg = X / r
        norm = np.linalg.norm(arg)
        result = norm * norm
        if norm > 0:
            result *= np.log(norm)
        return result

    @staticmethod
    def beckert_wendland_c2_basis(X, r):
        """
        It implements the following formula:

        .. math::
            \\varphi(\\| \\boldsymbol{x} \\|) = \\left( 1 - \\frac{\\| \\boldsymbol{x} \\|}{r} \\right)^4_+
            \\left( 4 \\frac{\\| \\boldsymbol{x} \\|}{r} + 1 \\right)

        :param numpy.ndarray X: the vector x in the formula above.
        :param float r: the parameter r in the formula above.

        :return: result: the result of the formula above.
        :rtype: float
        """
        norm = np.linalg.norm(X)
        arg = norm / r
        first = 0
        if (1 - arg) > 0:
            first = np.power((1 - arg), 4)
        second = (4 * arg) + 1
        result = first * second
        return result

    def distance_matrix(self, X1, X2):
        """
        This method returns the following matrix:
        :math:`\\boldsymbol{D_{ij}} = \\varphi(\\| \\boldsymbol{x_i} - \\boldsymbol{y_j} \\|)`

        :param numpy.ndarray X1: the vector x in the formula above.
        :param numpy.ndarray X2: the vector y in the formula above.

        :return: matrix: the matrix D.
        :rtype: numpy.ndarray
        """
        m, n = X1.shape[0], X2.shape[0]
        matrix = np.zeros(shape=(m, n))
        for i in range(0, m):
            for j in range(0, n):
                matrix[i][j] = self.basis(X1[i] - X2[j], self.radius)
        return matrix


def reconstruct_f(original_input, original_output, rbf_input, rbf_output, basis,
                  radius):
    """
    Reconstruct a function by using the radial basis function approximations.

    :param array_like original_input: contains the original values of the function inputs
    :param array_like original_output: contains the original values of the function output
    :param array_like rbf_input: contains the interpolated input data for RBF approximation
    :param array_like rbf_output: contains the array elements to be updated with the RBF
        interpolated outputs after the approximation
    :param string basis: RBF basis function
    :param float radius: cut-off radius
    """
    radial = RBF(radius=radius, basis=basis)

    weights = np.dot(
        np.linalg.inv(radial.distance_matrix(original_input, original_input)),
        original_output)
    for i in range(rbf_input.shape[0]):
        for j in range(0, original_input.shape[0]):
            rbf_output[i] += weights[j] * radial.basis(
                rbf_input[i] - original_input[j], radial.radius)

"""
Base module that provides essential tools and transformations on airfoils.
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from ..ndinterpolator import reconstruct_f
from scipy.interpolate import RBFInterpolator

class ProfileInterface(ABC):
    """
    Base sectional profile of the propeller blade.

    Each sectional profile is a 2D airfoil that is split into two parts: the
    upper and lower parts. The coordinates of each part is represented by two
    arrays corresponding to the X and Y components in the 2D coordinate system.
    Such coordinates can be either generated using NACA functions, or be
    inserted directly by the user as custom profiles.

    :param numpy.ndarray xup_coordinates: 1D array that contains the
        X-components of the airfoil upper-half surface. Default value is None
    :param numpy.ndarray xdown_coordinates: 1D array that contains the
        X-components of the airfoil lower-half surface. Default value is None
    :param numpy.ndarray yup_coordinates: 1D array that contains the
        Y-components of the airfoil upper-half surface. Default value is None
    :param numpy.ndarray ydown_coordinates: 1D array that contains the
        Y-components of the airfoil lower-half surface. Default value is None
    :param numpy.ndarray chord_line: contains the X and Y coordinates of the
        straight line joining between the leading and trailing edges. Default
        value is None
    :param numpy.ndarray camber_line: contains the X and Y coordinates of the
        curve passing through all the mid-points between the upper and lower
        surfaces of the airfoil. Default value is None
    :param numpy.ndarray leading_edge: 2D coordinates of the airfoil's
        leading edge. Default values are zeros
    :param numpy.ndarray trailing_edge: 2D coordinates of the airfoil's
        trailing edge. Default values are zeros
    """
    @abstractmethod
    def generate_parameters(self, convention='british'):
        """
        Abstract method that generates the airfoil parameters based on the
        given coordinates.

        The method generates the airfoil's chord length, chord percentages,
        maximum camber, camber percentages, maximum thickness, thickness
        percentages.

        :param str convention: convention of the airfoil coordinates. Default
            value is 'british'
        """
        self._update_edges()
        # compute chord parameters
        self._chord_length = np.linalg.norm(self.leading_edge -
                self.trailing_edge)
        self._chord_percentage = (self.xup_coordinates - np.min(
            self.xup_coordinates))/self._chord_length
        # compute camber parameters
        _camber = (self.yup_coordinates + self.ydown_coordinates)/2
        self._camber_max = abs(np.max(_camber))
        if self._camber_max == 0:
            self._camber_percentage = np.zeros(self.xup_coordinates.shape[0])
        elif self.camber_max != 0:
            self._camber_percentage = _camber/self._camber_max
        # compute thickness parameters
        if convention == 'british' or self._camber_max==0:
            _thickness = abs(self.yup_coordinates - self.ydown_coordinates)
        elif convention == 'american':
            _thickness = self._compute_thickness_american()
        self._thickness_max = np.max(_thickness)
        if self._thickness_max == 0:
            self._thickness_percentage = np.zeros(self.xup_coordinates.shape[0])
        elif self._thickness_max != 0:
            self._thickness_percentage = _thickness/self._thickness_max

    @abstractmethod
    def generate_coordinates(self):
        """
        Abstract method that generates the airfoil coordinates based on the
        given parameters.

        The method generates the airfoil's upper and lower surfaces
        coordinates. The method is called automatically when the airfoil
        parameters are inserted by the user.

        :param str convention: convention of the airfoil coordinates. Default
            value is 'british'
        """
        pass

    @property
    def xup_coordinates(self):
        """
        X-coordinates of the upper surface of the airfoil.
        """
        return self._xup_coordinates

    @xup_coordinates.setter
    def xup_coordinates(self, xup_coordinates):
        self._xup_coordinates = xup_coordinates

    @property
    def xdown_coordinates(self):
        """
        X-coordinates of the lower surface of the airfoil.
        """
        return self._xdown_coordinates

    @xdown_coordinates.setter
    def xdown_coordinates(self, xdown_coordinates):
        self._xdown_coordinates = xdown_coordinates

    @property
    def yup_coordinates(self):
        """
        Y-coordinates of the upper surface of the airfoil.
        """
        return self._yup_coordinates

    @yup_coordinates.setter
    def yup_coordinates(self, yup_coordinates):
        self._yup_coordinates = yup_coordinates

    @property
    def ydown_coordinates(self):
        """
        Y-coordinates of the lower surface of the airfoil.
        """
        return self._ydown_coordinates

    @ydown_coordinates.setter
    def ydown_coordinates(self, ydown_coordinates):
        self._ydown_coordinates = ydown_coordinates

    @property
    def chord_length(self):
        """
        Chord length of the airfoil.
        """
        return self._chord_length

    @chord_length.setter
    def chord_length(self, chord_length):
        self._chord_length = chord_length

    @property
    def chord_percentage(self):
        """
        Chord percentages of the airfoil.
        """
        return self._chord_percentage

    @chord_percentage.setter
    def chord_percentage(self, chord_percentage):
        self._chord_percentage = chord_percentage

    @property
    def camber_max(self):
        """
        Maximum camber of the airfoil.
        """
        return self._camber_max

    @camber_max.setter
    def camber_max(self, camber_max):
        self._camber_max = camber_max

    @property
    def camber_percentage(self):
        """
        Camber percentages of the airfoil.
        """
        return self._camber_percentage

    @camber_percentage.setter
    def camber_percentage(self, camber_percentage):
        self._camber_percentage = camber_percentage

    @property
    def thickness_max(self):
        """
        Maximum thickness of the airfoil.
        """
        return self._thickness_max

    @thickness_max.setter
    def thickness_max(self, thickness_max):
        self._thickness_max = thickness_max

    @property
    def thickness_percentage(self):
        """
        Thickness percentages of the airfoil.
        """
        return self._thickness_percentage

    @thickness_percentage.setter
    def thickness_percentage(self, thickness_percentage):
        self._thickness_percentage = thickness_percentage

    def _update_edges(self):
        """
        Private method that identifies and updates the airfoil's leading and
        trailing edges.

        Given the airfoil coordinates from the leading to the trailing edge,
        if the trailing edge has a non-zero thickness, then the average value
        between the upper and lower trailing edges is taken as the true
        trailing edge, hence both the leading and the trailing edges are always
        unique.
        """
        self.leading_edge = np.zeros(2)
        self.trailing_edge = np.zeros(2)
        if np.fabs(self.xup_coordinates[0] - self.xdown_coordinates[0]) > 1e-4:
            raise ValueError('Airfoils must have xup_coordinates[0] '\
                            'almost equal to xdown_coordinates[0]')
        if np.fabs(
                self.xup_coordinates[-1] - self.xdown_coordinates[-1]) > 1e-4:
            raise ValueError('Airfoils must have xup_coordinates[-1] '\
                             'almost equal to xdown_coordinates[-1]')

        self.leading_edge[0] = self.xup_coordinates[0]
        self.leading_edge[1] = self.yup_coordinates[0]
        self.trailing_edge[0] = self.xup_coordinates[-1]

        if self.yup_coordinates[-1] == self.ydown_coordinates[-1]:
            self.trailing_edge[1] = self.yup_coordinates[-1]
        else:
            self.trailing_edge[1] = 0.5 * (
                self.yup_coordinates[-1] + self.ydown_coordinates[-1])

    def interpolate_coordinates(self, num=500, radius=1.0):
        """
        Interpolate the airfoil coordinates from the given data set of
        discrete points.

        The interpolation applies the Radial Basis Function (RBF) method,
        to construct approximations of the two functions that correspond to the
        airfoil upper half and lower half coordinates. The RBF implementation
        is present in :ref:`RBF ndinterpolator <ndinterpolator-label>`.

        References:

        Buhmann, Martin D. (2003), Radial Basis Functions: Theory
        and Implementations.
        http://www.cs.bham.ac.uk/~jxb/NN/l12.pdf
        https://www.cc.gatech.edu/~isbell/tutorials/rbf-intro.pdf

        :param int num: number of interpolated points. Default value is 500
        :param float radius: range of the cut-off radius necessary for the RBF
            interpolation. Default value is 1.0. It is quite necessary to
            adjust the value properly so as to ensure a smooth interpolation
        :return: interpolation points for the airfoil upper half X-component,
            interpolation points for the airfoil lower half X-component,
            interpolation points for the airfoil upper half Y-component,
            interpolation points for the airfoil lower half Y-component
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        :raises TypeError: if num is not of type int
        :raises ValueError: if num is not positive, or if radius is not
            positive
        """
        if not isinstance(num, int):
            raise TypeError('Inserted value must be of type integer.')
        if num <= 0 or radius <= 0:
            raise ValueError('Inserted value must be positive.')

        xx_up = np.linspace(
            self.xup_coordinates[0], self.xup_coordinates[-1], num=num)
        yy_up = np.zeros(num)
        reconstruct_f(
            basis='beckert_wendland_c2_basis',
            radius=radius,
            original_input=self.xup_coordinates,
            original_output=self.yup_coordinates,
            rbf_input=xx_up,
            rbf_output=yy_up)
        xx_down = np.linspace(
            self.xdown_coordinates[0], self.xdown_coordinates[-1], num=num)
        yy_down = np.zeros(num)
        reconstruct_f(
            basis='beckert_wendland_c2_basis',
            radius=radius,
            original_input=self.xdown_coordinates,
            original_output=self.ydown_coordinates,
            rbf_input=xx_down,
            rbf_output=yy_down)

        return xx_up, xx_down, yy_up, yy_down

    def compute_chord_line(self, n_interpolated_points=None):
        """
        Compute the 2D coordinates of the chord line. Also updates
        the chord_line class member.

        The chord line is the straight line that joins between the leading edge
        and the trailing edge. It is simply computed from the equation of
        a line passing through two points, the LE and TE.

        :param int n_interpolated_points: number of points to be used for the
            equally-spaced sample computations. If None then there is no
            interpolation, unless the arrays x_up != x_down elementwise which
            implies that the corresponding y_up and y_down can not be
            comparable, hence a uniform interpolation is required. Default
            value is None
        """
        self._update_edges()
        aratio = ((self.trailing_edge[1] - self.leading_edge[1]) /
                  (self.trailing_edge[0] - self.leading_edge[0]))
        if not (self.xup_coordinates == self.xdown_coordinates
                ).all() and n_interpolated_points is None:
            # If x_up != x_down element-wise, then the corresponding y_up and
            # y_down can not be comparable, hence a uniform interpolation is
            # required. Also in case the interpolated_points is None,
            # then we assume a default number of interpolated points
            n_interpolated_points = 500

        if n_interpolated_points:
            cl_x_coordinates = np.linspace(
                self.leading_edge[0],
                self.trailing_edge[0],
                num=n_interpolated_points)
            cl_y_coordinates = (aratio *
                                (cl_x_coordinates - self.leading_edge[0]) +
                                self.leading_edge[1])
            self.chord_line = np.array([cl_x_coordinates, cl_y_coordinates])
        else:
            cl_y_coordinates = (aratio *
                                (self.xup_coordinates - self.leading_edge[0]) +
                                self.leading_edge[1])
            self.chord_line = np.array([self.xup_coordinates, cl_y_coordinates])

    def compute_camber_line(self, n_interpolated_points=None):
        """
        Compute the 2D coordinates of the camber line. Also updates the
        camber_line class member.

        The camber line is defined by the curve passing through all the mid
        points between the upper surface and the lower surface of the airfoil.

        :param int n_interpolated_points: number of points to be used for the
            equally-spaced sample computations. If None then there is no
            interpolation, unless the arrays x_up != x_down elementwise which
            implies that the corresponding y_up and y_down can not be
            comparable, hence a uniform interpolation is required. Default
            value is None

        We note that a uniform interpolation becomes necessary for the cases
        when the X-coordinates of the upper and lower surfaces do not
        correspond to the same vertical sections, since this would imply
        inaccurate measurements for obtaining the camber line.
        """
        if not (self.xup_coordinates == self.xdown_coordinates
                ).all() and n_interpolated_points is None:
            # If x_up != x_down element-wise, then the corresponding y_up and
            # y_down can not be comparable, hence a uniform interpolation is
            # required. Also in case the interpolated_points is None,
            # then we assume a default number of interpolated points
            n_interpolated_points = 500

        if n_interpolated_points:
            cl_x_coordinates, yy_up, yy_down = (
                self.interpolate_coordinates(num=n_interpolated_points)[1:])
            cl_y_coordinates = 0.5 * (yy_up + yy_down)
            self.camber_line = np.array([cl_x_coordinates, cl_y_coordinates])
        else:
            cl_y_coordinates = (0.5 *
                                (self.ydown_coordinates + self.yup_coordinates))
            self.camber_line = np.array(
                [self.xup_coordinates, cl_y_coordinates])

    def deform_camber_line(self, percent_change, n_interpolated_points=None):
        """
        Deform camber line according to a given percentage of change of the
        maximum camber. Also reconstructs the deformed airfoil's coordinates.

        The percentage of change is defined as follows:

        .. math::
            \\frac{\\text{new magnitude of max camber - old magnitude of
            maximum \
            camber}}{\\text{old magnitude of maximum camber}} * 100

        A positive percentage means the new camber is larger than the max
        camber value, while a negative percentage indicates the new value
        is smaller.

        We note that the method works only for airfoils in the reference
        position, i.e. chord line lies on the X-axis and the foil is not
        rotated, since the measurements are based on the Y-values of the
        airfoil coordinates, hence any measurements or scalings will be
        inaccurate for the foils not in their reference position.

        :param float percent_change: percentage of change of the
            maximum camber. Default value is None
        :param bool interpolate:  if True, the interpolated coordinates are
            used to compute the camber line and foil's thickness, otherwise
            the original discrete coordinates are used. Default value is False.
        :param int n_interpolated_points: number of points to be used for the
            equally-spaced sample computations. If None then there is no
            interpolation, unless the arrays x_up != x_down elementwise which
            implies that the corresponding y_up and y_down can not be
            comparable, hence a uniform interpolation is required. Default
            value is None
        """
        # Updating camber line
        self.compute_camber_line(n_interpolated_points=n_interpolated_points)
        scaling_factor = percent_change / 100. + 1.
        self.camber_line[1] *= scaling_factor

        if not (self.xup_coordinates == self.xdown_coordinates
                ).all() and n_interpolated_points is None:
            # If x_up != x_down element-wise, then the corresponding y_up and
            # y_down can not be comparable, hence a uniform interpolation is
            # required. Also in case the interpolated_points is None,
            # then we assume a default number of interpolated points
            n_interpolated_points = 500

        # Evaluating half-thickness of the undeformed airfoil,
        # which should hold same values for the deformed foil.
        if n_interpolated_points:
            (self.xup_coordinates, self.xdown_coordinates, self.yup_coordinates,
             self.ydown_coordinates
             ) = self.interpolate_coordinates(num=n_interpolated_points)

        half_thickness = 0.5 * np.fabs(
            self.yup_coordinates - self.ydown_coordinates)

        self.yup_coordinates = self.camber_line[1] + half_thickness
        self.ydown_coordinates = self.camber_line[1] - half_thickness

    @property
    def yup_curve(self):
        """
        Return the spline function corresponding to the upper profile
        of the airfoil

        :return: a spline function
        :rtype: scipy interpolation object

        .. todo::
            generalize the interpolation function
        """
        spline = RBFInterpolator(self.xup_coordinates.reshape(-1,1),
               self.yup_coordinates.reshape(-1,1))
        return spline

    @property
    def ydown_curve(self):
        """
        Return the spline function corresponding to the lower profile
        of the airfoil

        :return: a spline function
        :rtype: scipy interpolation object

        .. todo::
            generalize the interpolation function
        """
        spline = RBFInterpolator(self.xdown_coordinates.reshape(-1,1),
               self.ydown_coordinates.reshape(-1,1))
        return spline

    @property
    def reference_point(self):
        """
        Return the coordinates of the chord's mid point.

        :return: reference point in 2D
        :rtype: numpy.ndarray
        """
        self._update_edges()
        reference_point = [
            0.5 * (self.leading_edge[0] + self.trailing_edge[0]),
            0.5 * (self.leading_edge[1] + self.trailing_edge[1])
        ]
        return np.asarray(reference_point)

    def _compute_thickness_american(self):
        """
        Compute the thickness of the airfoil using the American standard
        definition.
        """
        n_pos = self.xup_coordinates.shape[0]
        m = np.zeros(n_pos)
        for i in range(1, n_pos, 1):
            m[i] = (self._camber_percentage[i]-
                self._camber_percentage[i-1])/(self._chord_percentage[i]-
                self._chord_percentage[i-1])*self._camber_max/self._chord_length
        m_angle = np.arctan(m)

        # generating temporary profile coordinates orthogonal to the camber
        # line
        camber = self._camber_max*self._camber_percentage
        ind_horizontal_camber = np.sin(m_angle)==0
        def eq_to_solve(x):
            spline_curve = self.ydown_curve(x.reshape(-1,1)).reshape(
                        x.shape[0],)
            line_orth_camber = (camber[~ind_horizontal_camber] +
                np.cos(m_angle[~ind_horizontal_camber])/
                np.sin(m_angle[~ind_horizontal_camber])*(
                    self._chord_percentage[~ind_horizontal_camber]
                    *self._chord_length-x))
            return spline_curve - line_orth_camber

        xdown_tmp = self.xdown_coordinates.copy()
        xdown_tmp[~ind_horizontal_camber] = newton(eq_to_solve,
            xdown_tmp[~ind_horizontal_camber])
        xup_tmp = 2*self._chord_percentage*self._chord_length - xdown_tmp
        ydown_tmp = self.ydown_curve(xdown_tmp.reshape(-1,1)).reshape(
            xdown_tmp.shape[0],)
        yup_tmp = 2*self._camber_max*self._camber_percentage - ydown_tmp
        if xup_tmp[1]<self.xup_coordinates[0]:
            xup_tmp[1], xdown_tmp[1] = xup_tmp[2], xdown_tmp[2]
            yup_tmp[1], ydown_tmp[1] = yup_tmp[2], ydown_tmp[2]
        return np.sqrt((xup_tmp-xdown_tmp)**2 + (yup_tmp-ydown_tmp)**2)

    def max_thickness(self, n_interpolated_points=None):
        """
        Return the airfoil's maximum thickness.

        Thickness is defined as the distnace between the upper and lower
        surfaces of the airfoil, and can be measured in two different ways:

            - American convention: measures along the line perpendicular to \
                the mean camber line.

            - British convention: measures along the line perpendicular to \
                the chord line.

        In this implementation, the british convention is used to evaluate
        the maximum thickness.

        References:

            Phillips, Warren F. (2010). Mechanics of Flight (2nd ed.). \
                Wiley & Sons. p. 27. ISBN 978-0-470-53975-0.

            Bertin, John J.; Cummings, Russel M. (2009). Pearson Prentice Hall,\
                ed. Aerodynamics for Engineers (5th ed.). \
                p. 199. ISBN 978-0-13-227268-1.

        :param bool interpolate: if True, the interpolated coordinates are used
            to measure the thickness; otherwise, the original discrete
            coordinates are used. Default value is False
        :param int n_interpolated_points: number of points to be used for the
            equally-spaced sample computations. If None then there is no
            interpolation, unless the arrays x_up != x_down elementwise which
            implies that the corresponding y_up and y_down can not be
            comparable, hence a uniform interpolation is required. Default
            value is None
        :return: maximum thickness
        :rtype: float
        """
        if not (self.xup_coordinates == self.xdown_coordinates
                ).all() and n_interpolated_points is None:
            # If x_up != x_down element-wise, then the corresponding y_up and
            # y_down can not be comparable, hence a uniform interpolation is
            # required. Also in case the interpolated_points is None,
            # then we assume a default number of interpolated points
            n_interpolated_points = 500

        if n_interpolated_points:
            # Evaluation of the thickness requires comparing both y_up and
            # y_down for the same x-section, (i.e. same x_coordinate),
            # according to british convention. If x_up != x_down element-wise,
            # then the corresponding y_up and y_down can not be comparable,
            # hence a uniform interpolation is required.
            yy_up, yy_down = self.interpolate_coordinates(
                num=n_interpolated_points)[2:]
            return np.fabs(yy_up - yy_down).max()
        return np.fabs(self.yup_coordinates - self.ydown_coordinates).max()

    def max_camber(self, n_interpolated_points=500):
        """
        Return the magnitude of the airfoil's maximum camber.

        Camber is defined as the distance between the chord line and the mean
        camber line, and is measured along the line perpendicular to the chord
        line.

        :param bool interpolate: if True, the interpolated coordinates are used
            to measure the camber; otherwise, the original discrete coordinates
            are used. Default value is False
        :param int n_interpolated_points: number of points to be used for the
            equally-spaced sample computations. If None then there is no
            interpolation, unless the arrays x_up != x_down elementwise which
            implies that the corresponding y_up and y_down can not be
            comparable, hence a uniform interpolation is required. Default
            value is None
        :return: maximum camber
        :rtype: float
        """
        self.compute_chord_line(n_interpolated_points=n_interpolated_points)

        self.compute_camber_line(n_interpolated_points=n_interpolated_points)

        n_points = self.camber_line[0].size
        camber = np.zeros(n_points)

        for i in range(n_points):
            camber[i] = np.linalg.norm(
                self.chord_line[:, i] - self.camber_line[:, i])

        max_camber = camber.max()
        if (self.camber_line[1][camber.argmax()] <
                self.chord_line[1][camber.argmax()]):
            # Camber line is below the chord line, at the point of max camber
            max_camber *= -1

        return max_camber

    def rotate(self, rad_angle=None, deg_angle=None):
        """
        2D counter clockwise rotation about the origin of the Cartesian
        coordinate system.

        The rotation matrix, :math:`R(\\theta)`, is used to perform rotation
        in the 2D Euclidean space about the origin, which is -- by default --
        the leading edge.

        :math:`R(\\theta)` is defined by:

        .. math::
             \\left(\\begin{matrix} cos (\\theta) & - sin (\\theta) \\\\
            sin (\\theta) & cos (\\theta) \\end{matrix}\\right)

        Given the coordinates of point :math:`P` such that

        .. math::
            P = \\left(\\begin{matrix} x \\\\
            y \\end{matrix}\\right),

        Then, the rotated coordinates will be:

        .. math::
            P^{'} = \\left(\\begin{matrix} x^{'} \\\\
                     y^{'} \\end{matrix}\\right)
                  = R (\\theta) \\cdot P

        If a standard right-handed Cartesian coordinate system is used, with
        the X-axis to the right and the Y-axis up, the rotation
        :math:`R (\\theta)` is counterclockwise. If a left-handed Cartesian
        coordinate system is used, with X-axis directed to the right and Y-axis
        directed down, :math:`R (\\theta)` is clockwise.

        :param float rad_angle: angle in radians. Default value is None
        :param float deg_angle: angle in degrees. Default value is None
        :raises ValueError: if both rad_angle and deg_angle are inserted,
            or if neither is inserted
        """
        if rad_angle is not None and deg_angle is not None:
            raise ValueError(
                'You have to pass either the angle in radians or in degrees,' \
                ' not both.')
        if rad_angle is not None:
            cosine = np.cos(rad_angle)
            sine = np.sin(rad_angle)
        elif deg_angle is not None:
            cosine = np.cos(np.radians(deg_angle))
            sine = np.sin(np.radians(deg_angle))
        else:
            raise ValueError(
                'You have to pass either the angle in radians or in degrees.')

        rot_matrix = np.array([cosine, -sine, sine, cosine]).reshape((2, 2))

        coord_matrix_up = np.vstack((self.xup_coordinates,
                                     self.yup_coordinates))
        coord_matrix_down = np.vstack((self.xdown_coordinates,
                                       self.ydown_coordinates))

        new_coord_matrix_up = np.zeros(coord_matrix_up.shape)
        new_coord_matrix_down = np.zeros(coord_matrix_down.shape)

        for i in range(self.xup_coordinates.shape[0]):
            new_coord_matrix_up[:, i] = np.dot(rot_matrix,
                                               coord_matrix_up[:, i])
            new_coord_matrix_down[:, i] = np.dot(rot_matrix,
                                                 coord_matrix_down[:, i])

        self.xup_coordinates = new_coord_matrix_up[0]
        self.xdown_coordinates = new_coord_matrix_down[0]
        self.yup_coordinates = new_coord_matrix_up[1]
        self.ydown_coordinates = new_coord_matrix_down[1]

    def translate(self, translation):
        """
        Translate the airfoil coordinates according to a 2D translation vector.

        :param array_like translation: the translation vector in 2D
        """
        self.xup_coordinates += translation[0]
        self.xdown_coordinates += translation[0]
        self.yup_coordinates += translation[1]
        self.ydown_coordinates += translation[1]

    def reflect(self):
        """
        Reflect the airfoil coordinates about the origin, i.e. a mirror
        transformation is performed about both the X-axis and the Y-axis.
        """
        self.xup_coordinates *= -1
        self.xdown_coordinates *= -1
        self.yup_coordinates *= -1
        self.ydown_coordinates *= -1

    def scale(self, factor, translate=True):
        """
        Scale the airfoil coordinates according to a scaling factor.

        In order to apply the scaling without affecting the position of the
        reference point, the method translates the airfoil by its refernce
        point to be centered in the origin, then the scaling is applied, and
        finally the airfoil is translated back by its reference point to the
        initial position.

        :param float factor: the scaling factor
        """
        if translate:
            ref_point = self.reference_point
            self.translate(-ref_point)
            self.xup_coordinates *= factor
            self.xdown_coordinates *= factor
            self.yup_coordinates *= factor
            self.ydown_coordinates *= factor
            self.translate(ref_point)
        else:
            self.xup_coordinates *= factor
            self.xdown_coordinates *= factor
            self.yup_coordinates *= factor
            self.ydown_coordinates *= factor

    def plot(self,
             profile=True,
             chord_line=False,
             camber_line=False,
             ref_point=False,
             outfile=None):
        """
        Plot the airfoil coordinates.

        :param bool profile: if True, then plot the profile coordinates.
            Default value is True
        :param bool chord_line: if True, then plot the chord line. Default
            value is False
        :param bool camber_line: if True, then plot the camber line. Default
            value is False
        :param bool ref_point: if True, then scatter plot the reference point.
            Default value is False
        :param str outfile: outfile name. If a string is provided then the
            plot is saved with that name, otherwise the plot is not saved.
            Default value is None
        """
        plt.figure()

        if (self.xup_coordinates is None or self.yup_coordinates is None
                or self.xdown_coordinates is None
                or self.ydown_coordinates is None):
            raise ValueError('One or all the coordinates have None value.')

        if profile:
            plt.plot(
                self.xup_coordinates,
                self.yup_coordinates,
                label='Upper profile')
            plt.plot(
                self.xdown_coordinates,
                self.ydown_coordinates,
                label='Lower profile')

        if chord_line:
            if self.chord_line is None:
                raise ValueError(
                    'Chord line is None. You must compute it first')
            plt.plot(self.chord_line[0], self.chord_line[1], label='Chord line')

        if camber_line:
            if self.camber_line is None:
                raise ValueError(
                    'Camber line is None. You must compute it first')
            plt.plot(
                self.camber_line[0], self.camber_line[1], label='Camber line')

        if ref_point:
            plt.scatter(
                self.reference_point[0],
                self.reference_point[1],
                s=15,
                label='Reference point')

        plt.grid(linestyle='dotted')
        plt.axis('equal')
        plt.legend()

        if outfile:
            if not isinstance(outfile, str):
                raise ValueError('Output file name must be string.')
            plt.savefig(outfile)
        else:
            plt.show()


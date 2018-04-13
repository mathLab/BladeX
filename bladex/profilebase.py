"""
Base module for blade construction. Provide essential tools for airfoils.
"""

import numpy as np
import matplotlib.pyplot as plt
from .ndinterpolator import reconstruct_f


class ProfileBase(object):
    """
    Base sectional profile of the propeller blade.

    Each sectional profile is a 2D airfoil that is split into two parts: the
    upper and lower parts. The coordiates of each part is represented by two
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

    def __init__(self):
        self.xup_coordinates = None
        self.xdown_coordinates = None
        self.yup_coordinates = None
        self.ydown_coordinates = None
        self.chord_line = None
        self.camber_line = None
        self.leading_edge = np.zeros(2)
        self.trailing_edge = np.zeros(2)

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
        if self.xup_coordinates[0] != self.xdown_coordinates[0]:
            raise ValueError('Airfoils must have xup_coordinates[0] \
                            == xdown_coordinates[0]')
        if self.xup_coordinates[-1] != self.xdown_coordinates[-1]:
            raise ValueError('Airfoils must have xup_coordinates[-1] \
                                == xdown_coordinates[-1]')

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
        is present in :doc:`/utils/rbf`.

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
            \\frac{new magnitude of max camber - old magnitude of maximum \
             camber}{old magnitude of maximum camber} * 100

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
    def reference_point(self):
        """
        Return the coordinates of the airfoil's geometric center.

        :return: reference point in 2D
        :rtype: numpy.ndarray
        """
        self._update_edges()
        reference_point = [
            0.5 * (self.leading_edge[0] + self.trailing_edge[0]),
            0.5 * (self.leading_edge[1] + self.trailing_edge[1])
        ]
        return np.asarray(reference_point)

    @property
    def chord_length(self):
        """
        Measure the l2-norm (Euclidean distance) between the leading edge
        and the trailing edge.

        :return: chord length
        :rtype: float
        """
        self._update_edges()
        return np.linalg.norm(self.leading_edge - self.trailing_edge)

    def max_thickness(self, n_interpolated_points=None):
        """
        Return the airfoil's maximum thickness.

        Thickness is defined as the distnace between the upper and lower
        surfaces of the airfoil, and can be measured in two different ways:
            - American convention: measures along the line perpendicular to
              the mean camber line.
            - British convention: measures along the line perpendicular to
              the chord line.
        In this implementation, the british convention is used to evaluate
            the maximum thickness.

        References:
            Phillips, Warren F. (2010). Mechanics of Flight (2nd ed.).
                Wiley & Sons. p. 27. ISBN 978-0-470-53975-0.
            Bertin, John J.; Cummings, Russel M. (2009). Pearson Prentice Hall,
                ed. Aerodynamics for Engineers (5th ed.).
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
             \\left(\\begin{matrix} cos (\\theta) & - sin (\\theta) \\
            sin (\\theta) & cos (\\theta) \\end{matrix}\\right)

        Given the coordinates of point :math:`(P) =
            \\left(\\begin{matrix} x \\ y \\end{matrix}\\right)`,
        the rotated coordinates will be: .. math::
            P^{'} = \\left(\\begin{matrix} x^{'} \\ y^{'} \\end{matrix}\\right)
                  = R (\\theta) \\cdot P`

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
                'You have to pass either the angle in radians or in degrees, \
                not both.')
        if rad_angle:
            cosine = np.cos(rad_angle)
            sine = np.sin(rad_angle)
        elif deg_angle:
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

    def flip(self):
        """
        Flip the airfoil coordinates about both the X-axis and the Y-axis.
        """
        self.xup_coordinates *= -1
        self.xdown_coordinates *= -1
        self.yup_coordinates *= -1
        self.ydown_coordinates *= -1

    def scale(self, factor):
        """
        Scale the airfoil coordinates according to a scaling factor.

        In order to apply the scaling without affecting the position of the
        reference point, the method translates the airfoil by its refernce
        point to be centered in the origin, then the scaling is applied, and
        finally the airfoil is translated back by its reference point to the
        initial position.

        :param float factor: the scaling factor
        """
        ref_point = self.reference_point
        self.translate(-ref_point)
        self.xup_coordinates *= factor
        self.xdown_coordinates *= factor
        self.yup_coordinates *= factor
        self.ydown_coordinates *= factor
        self.translate(ref_point)

    def plot(self, outfile=None):
        """
        Plot the airfoil coordinates.

        :param string outfile: outfile name. If a string is provided then the
            plot is saved with that name, otherwise the plot is not saved.
            Default value is None
        """
        plt.figure()
        plt.plot(self.xup_coordinates, self.yup_coordinates)
        plt.plot(self.xdown_coordinates, self.ydown_coordinates)
        plt.grid(linestyle='dotted')
        plt.axis('equal')

        if outfile:
            if not isinstance(outfile, str):
                raise ValueError('Output file name must be string.')
            plt.savefig(outfile)

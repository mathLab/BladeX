import numpy as np
import matplotlib.pyplot as plt
from .utils.rbf import reconstruct_f


class ProfileBase(object):
    """
    Base sectional profile of the propeller blade.

    Each sectional profile is a 2D airfoil that is split into two parts: the upper and lower parts. The coordiates of each part is represented by two arrays corresponding to 
    the X and Y components in the 2D coordinate system. Such coordinates can be either generated using NACA functions, or be inserted directly by the user as custom profiles.

    :param numpy.ndarray xup_coordinates: 1D array that contains the X-components of the airfoil upper-half surface. Default value is None
    :param numpy.ndarray xdown_coordinates: 1D array that contains the X-components of the airfoil lower-half surface. Default value is None
    :param numpy.ndarray yup_coordinates: 1D array that contains the Y-components of the airfoil upper-half surface. Default value is None
    :param numpy.ndarray ydown_coordinates: 1D array that contains the Y-components of the airfoil lower-half surface. Default value is None
    :param numpy.ndarray leading_edge: 2D coordinates of the airfoil's leading edge. Default values are zeros
    :param numpy.ndarray trailing_edge: 2D coordinates of the airfoil's trailing edge. Default values are zeros
    """

    def __init__(self):
        self.xup_coordinates = None
        self.xdown_coordinates = None
        self.yup_coordinates = None
        self.ydown_coordinates = None
        self.leading_edge = np.zeros(2)
        self.trailing_edge = np.zeros(2)

    def _update_edges(self):
        """
        Private method that identifies and updates the airfoil's leading and trailing edges.

        Given the airfoil coordinates from the leading to the trailing edge, if the trailing edge has a non-zero thickness, 
        then the average value between the upper and lower trailing edges is taken as the true trailing edge, hence both 
        the leading and the trailing edges are always unique.
        """
        self.leading_edge[0] = self.xup_coordinates[0]
        self.leading_edge[1] = self.yup_coordinates[0]
        self.trailing_edge[0] = self.xup_coordinates[-1]

        if self.yup_coordinates[-1] == self.ydown_coordinates[-1]:
            self.trailing_edge[1] = self.yup_coordinates[-1]
        else:
            self.trailing_edge[1] = 0.5 * (
                self.yup_coordinates[-1] + self.ydown_coordinates[-1])

    @property
    def reference_point(self):
        """
        Returns the coordinates of the airfoil's geometric center.

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
        Measures the l2-norm (Euclidean distance) between the leading edge and the trailing edge.
        
        :return: chord length
        :rtype: float
        """
        self._update_edges()
        return np.linalg.norm(self.leading_edge - self.trailing_edge)

    def interpolate_coordinates(self, num=500, radius=1.0):
        """
        Interpolates the airfoil coordinates from the given data set of discrete points.

        The interpolation applies the Radial Basis Function (RBF) method, to construct approximations of the two functions
        that correspond to the airfoil upper half and lower half coordinates. The RBF implementation is present in :doc:`/utils/rbf`.

        References:
        Buhmann, Martin D. (2003), Radial Basis Functions: Theory and Implementations.
        http://www.cs.bham.ac.uk/~jxb/NN/l12.pdf
        https://www.cc.gatech.edu/~isbell/tutorials/rbf-intro.pdf

        :param int num: number of interpolated points. Default value is 500
        :param float radius: range of the cut-off radius necessary for the RBF interpolation. Default value is 1.0 
                             It is quite necessary to adjust the value properly so as to ensure a smooth interpolation
        :return: interpolation points for the airfoil upper half X-component, interpolation points for the airfoil lower half X-component, 
                 interpolation points for the airfoil upper half Y-component, interpolation points for the airfoil lower half Y-component
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        :raises TypeError: if num is not of type int
        :raises ValueError: if num is not positive, or if radius is not positive
        """
        if not isinstance(num, int):
            raise TypeError('inserted value must be of type integer.')
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

    def max_thickness(self, interpolate=False):
        """
        Returns the airfoil's maximum thickness.

        Thickness is defined as the distnace between the upper and lower surfaces of the airfoil, and can be measured 
        in two different ways:
            - American convention: measures along the line perpendicular to the mean camber line.
            - British convention: measures along the line perpendicular to the chord line.
        In this implementation, the british convention is used to evaluate the maximum thickness.

        References:
            Phillips, Warren F. (2010). Mechanics of Flight (2nd ed.). Wiley & Sons. p. 27. ISBN 978-0-470-53975-0.
            Bertin, John J.; Cummings, Russel M. (2009). Pearson Prentice Hall, ed. Aerodynamics for Engineers (5th ed.). p. 199. ISBN 978-0-13-227268-1.

        :param bool interpolate: if True, the interpolated coordinates are used to measure the thickness; 
                                 otherwise, the original discrete coordinates are used. Default value is False
        :return: maximum thickness
        :rtype: float
        """
        if (interpolate is True) or (
            (self.xup_coordinates == self.xdown_coordinates).all() is False):
            # Evaluation of the thickness requires comparing both y_up and y_down for the same x-section,
            # (i.e. same x_coordinate), according to the british convention. If x_up != x_down element-wise,
            # then the corresponding y_up and y_down can not be comparable, hence a uniform interpolation is required.
            xx_up, xx_down, yy_up, yy_down = self.interpolate_coordinates()
            return np.fabs(yy_up - yy_down).max()
        return np.fabs(self.yup_coordinates - self.ydown_coordinates).max()

    def max_camber(self, interpolate=False):
        """
        Returns the airfoil's maximum camber.

        Camber is defined as the distance between the chord line and the mean camber line, and is 
        measured along the line perpendicular to the chord line.

        :param bool interpolate: if True, the interpolated coordinates are used to measure the camber; 
                                 otherwise, the original discrete coordinates are used. Default value is False
        :return: maximum camber
        :rtype: float
        """
        if (interpolate is True) or (
            (self.xup_coordinates == self.xdown_coordinates).all() is False):
            # Evaluation of camber requires comparing both y_up and y_down for the same x-section (i.e. same x_coordinate).
            # If x_up != x_down element-wise, then the corresponding y_up and y_down can not be comparable, hence a uniform interpolation is required.
            xx_up, xx_down, yy_up, yy_down = self.interpolate_coordinates()
            camber = yy_down + 0.5 * np.fabs(yy_up - yy_down)
        else:
            camber = self.ydown_coordinates + 0.5 * np.fabs(
                self.yup_coordinates - self.ydown_coordinates)
        return camber.max()

    def rotate(self, rad_angle=None, deg_angle=None):
        """
        2D counter clockwise rotation about the origin of the Cartesian coordinate system.
        
        The rotation matrix, :math:`R(\\theta)`, is used to perform rotation in the 2D Euclidean space about the origin, which is -- by default -- the leading edge.

        :math:`R(\\theta)` is defined by:
        .. math::
             \\left(\\begin{matrix} cos (\\theta) & - sin (\\theta) \\
            sin (\\theta) & cos (\\theta) \\end{matrix}\\right)

        Given the coordinates of point :math:`(P) = \\left(\\begin{matrix} x \\ y \\end{matrix}\\right)`, the rotated coordinates will be: 
        .. math::
            P^{'} = \\left(\\begin{matrix} x^{'} \\ y^{'} \\end{matrix}\\right) = R (\\theta) \\cdot P` 

        If a standard right-handed Cartesian coordinate system is used, with the X-axis to the right and the Y-axis up, the rotation :math:`R (\\theta)` 
        is counterclockwise. If a left-handed Cartesian coordinate system is used, with X-axis directed to the right and Y-axis directed down, :math:`R (\\theta)` is clockwise.

        :param float rad_angle: angle in radians. Default value is None
        :param float deg_angle: angle in degrees. Default value is None
        :raises ValueError: if both rad_angle and deg_angle are inserted, or if neither is inserted
        """
        if rad_angle is not None and deg_angle is not None:
            raise ValueError(
                'You have to pass either the angle in radians or in degrees, not both.')
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
        Translates the airfoil coordinates according to a 2D translation vector.
        
        :param array_like translation: the translation vector in 2D
        """
        self.xup_coordinates += translation[0]
        self.xdown_coordinates += translation[0]
        self.yup_coordinates += translation[1]
        self.ydown_coordinates += translation[1]

    def flip(self):
        """
        Flips the airfoil coordinates about both the X-axis and the Y-axis.
        """
        self.xup_coordinates *= -1
        self.xdown_coordinates *= -1
        self.yup_coordinates *= -1
        self.ydown_coordinates *= -1

    def scale(self, factor):
        """
        Scales the airfoil coordinates according to a scaling factor.

        In order to apply the scaling without affecting the position of the reference point, the method translates the airfoil by its refernce point 
        to be centered in the origin, then the scaling is applied, and finally the airfoil is translated back by its reference point to the initial position.

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
        Plots the airfoil coordinates.

        :param string outfile: outfile name. If a string is provided then the plot is saved with that name, otherwise the plot is not saved. Default value is None
        """
        plt.figure()
        plt.plot(self.xup_coordinates, self.yup_coordinates)
        plt.plot(self.xdown_coordinates, self.ydown_coordinates)
        plt.grid(linestyle='dotted')
        plt.axis('equal')

        if outfile:
            if not isinstance(outfile, string):
                raise ValueError('Output file name must be string.')
            plt.savefig(outfile)

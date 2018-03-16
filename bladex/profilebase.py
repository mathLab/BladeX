from math import cos, sin, tan, atan, pi, radians, degrees, sqrt
import numpy as np
import matplotlib.pyplot as plt

class BaseProfile(object):
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
            self.trailing_edge[1] = 0.5 * (self.yup_coordinates[-1] + self.ydown_coordinates[-1])

    @property
    def reference_point(self):
        """
        Returns the coordinates of the airfoil's geometric center.

        :return: reference point in 2D
        :rtype: numpy.ndarray
        """
        self._update_edges()
        reference_point = [0.5 * (self.leading_edge[0] + self.trailing_edge[0]), 
                           0.5 * (self.leading_edge[1] + self.trailing_edge[1])]
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
            raise ValueError('You have to pass either the angle in radians or in degrees, not both.')
        if rad_angle:
            cosine = cos(rad_angle)
            sine = sin(rad_angle)
        elif deg_angle:
            cosine = cos(radians(deg_angle))
            sine = sin(radians(deg_angle))
        else:
            raise ValueError('You have to pass either the angle in radians or in degrees.')

        rot_matrix = np.array([cosine, -sine, sine, cosine]).reshape((2, 2))

        coord_matrix_up = np.vstack((self.xup_coordinates, self.yup_coordinates))
        coord_matrix_down = np.vstack((self.xdown_coordinates, self.ydown_coordinates))

        new_coord_matrix_up = np.zeros(coord_matrix_up.shape)
        new_coord_matrix_down = np.zeros(coord_matrix_down.shape)

        for i in range(self.xup_coordinates.shape[0]):
            new_coord_matrix_up[:, i] = np.dot(rot_matrix, coord_matrix_up[:, i])
            new_coord_matrix_down[:, i] = np.dot(rot_matrix, coord_matrix_down[:, i])

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
        self.xup_coordinates  *= -1 
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
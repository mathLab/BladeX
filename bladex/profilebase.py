from math import cos, sin, tan, atan, pi, radians, degrees, sqrt
import numpy as np
import matplotlib.pyplot as plt

class BaseProfile(object):
    """
    TO DOC
    Naca cambered airfoils have xup_coordinates != xdown_coordinates
    Coordinates can be either generated using NACA functions, or be inserted directly by user (eg. Custom profiles)
    Coordinates always must be inserted as floats
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
        TO DOC
        Always: self.xup_coordinates[0] = self.xdown_coordinates[0] | self.xup_coordinates[-1] = self.xdown_coordinates[-1] | self.yup_coordinates[0] = self.ydown_coordinates[0]
        We must ensure for the coordinates that x,y[0] correspond to LE, while x,y[-1] correspond to TE, even for custom profiles; otherwise the assignment names will be incorrect.
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
        TO DOC
        If TE is open, then take average value as true TE. Hence, both LE and TE points are always unique.
        """
        self._update_edges()
        reference_point = [0.5 * (self.leading_edge[0] + self.trailing_edge[0]), 
                           0.5 * (self.leading_edge[1] + self.trailing_edge[1])]
        return np.asarray(reference_point)

    @property
    def chord_length(self):
        """
        TO DOC
        Measures the l2-norm (Euclidean distance) between the leading edge and trailing edge.
        """
        self._update_edges()
        return np.linalg.norm(self.leading_edge - self.trailing_edge)

    def rotate(self, rad_angle=None, deg_angle=None):
        """
        TO DOC
        ##Input angle is given in degrees## Did not work. Switch to radians
        Rotation is 2D and counter clockwise (for standard orientation system) about Z-axis
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
        TO DOC
        """
        self.xup_coordinates += translation[0]
        self.xdown_coordinates += translation[0]
        self.yup_coordinates += translation[1]
        self.ydown_coordinates += translation[1]

    def flip(self):
        """
        TO DOC
        """
        self.xup_coordinates  *= -1 
        self.xdown_coordinates *= -1 
        self.yup_coordinates *= -1
        self.ydown_coordinates *= -1

    def scale(self, factor):
        """
        TO DOC
        it translates the profile to the origin with respect to the reference point. Then scaling and 
        translating back
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
        TO DOC
        the plot function, if an outfile is provided then save it
        """
        plt.figure()
        plt.plot(self.xup_coordinates, self.yup_coordinates)
        plt.plot(self.xdown_coordinates, self.ydown_coordinates)
        plt.grid(linestyle='dotted')
        plt.axis('equal')

        if outfile:
            plt.savefig(outfile)
"""
Derived module from profilebase.py to provide the airfoil coordinates for a general
profile. Input data can be:
    - the coordinates arrays;
    - the chord percentages, the associated nondimensional camber and thickness,
      the real values of chord lengths, camber and thickness associated to the 
      single blade sections.
"""

import numpy as np
from .profilebase import ProfileBase


class CustomProfile(ProfileBase):
    """
    Provide custom profile, given the airfoil coordinates or the airfoil parameters,
    i.e. , chord percentages and length, nondimensional and maximum camber, 
    nondimensional and maximum thickness.
    
    If coordinates are direclty given as input:

    :param numpy.ndarray xup: 1D array that contains the X-components of the
        airfoil's upper surface
    :param numpy.ndarray xdown: 1D array that contains the X-components of the
        airfoil's lower surface
    :param numpy.ndarray yup: 1D array that contains the Y-components of the
        airfoil's upper surface
    :param numpy.ndarray ydown: 1D array that contains the Y-components of the
        airfoil's lower surface
        
    If section parameters are given as input:
        
    :param numpy.ndarray chord_perc: 1D array that contains the chord percentages
        of an airfoil section for which camber and thickness are measured
    :param numpy.ndarray camber_perc: 1D array that contains the camber percentage
       of an airfoil section at all considered chord percentages. The percentage is
       taken with respect to the section maximum camber
    :param numpy.ndarray thickness_perc: 1D array that contains the thickness 
       percentage of an airfoil section at all considered chord percentages. 
       The percentage is with respect to the section maximum thickness
    :param numpy.ndarray chord_len: 1-element array expressing the length of the chord 
       line of a certain airfoil section
    :param numpy.ndarray camber_max: 1-element array expressing the maximum camber at a 
       certain airfoil section
    :param numpy.ndarray thickness_max: 1-element array expressing the maximum thickness 
       at a certain airfoil section
    """
    def __init__(self, **kwargs):
        super(CustomProfile, self).__init__()

        if len(kwargs) == 4:
            xup, yup, xdown, ydown = kwargs.values()
            self.xup_coordinates = xup
            self.yup_coordinates = yup
            self.xdown_coordinates = xdown
            self.ydown_coordinates = ydown
            self.chord_len = 1

            self._generate_parameters()

        if len(kwargs) == 5:
            xup, yup, xdown, ydown, chord_len = kwargs.values()
            self.xup_coordinates = xup
            self.yup_coordinates = yup
            self.xdown_coordinates = xdown
            self.ydown_coordinates = ydown
            self.chord_len = chord_len

            self._generate_parameters()

        if len(kwargs) == 6:
            chord_perc, camber_perc, thickness_perc, chord_len, camber_max, thickness_max = kwargs.values(
            )
            self.chord_percentage = chord_perc
            self.camber_percentage = camber_perc
            self.thickness_percentage = thickness_perc
            self.chord_len = chord_len
            self.camber_max = camber_max
            self.thickness_max = thickness_max

            self._check_args()
            self._generate_coordinates()

        self._check_coordinates()

    def _check_args(self):
        """
        Private method that checks whether the airfoil parameters defined
        are provided correctly.
        In particular, the chord, camber and thickness percentages are consistent and
        have the same length.
        """

        if self.chord_percentage is None:
            raise ValueError('object "chord_perc" refers to an empty array.')
        if self.camber_percentage is None:
            raise ValueError('object "camber_perc" refers to an empty array.')
        if self.thickness_percentage is None:
            raise ValueError(
                'object "thickness_perc" refers to an empty array.')
        if self.chord_len is None:
            raise ValueError('object "chorf_len" refers to an empty array.')
        if self.camber_max is None:
            raise ValueError('object "camber_max" refers to an empty array.')
        if self.thickness_max is None:
            raise ValueError('object "thickness_max" refers to an empty array.')

        if not isinstance(self.chord_percentage, np.ndarray):
            self.chord_percentage = np.asarray(self.chord_percentage,
                                               dtype=float)
        if not isinstance(self.camber_percentage, np.ndarray):
            self.camber_percentage = np.asarray(self.camber_percentage,
                                                dtype=float)
        if not isinstance(self.thickness_percentage, np.ndarray):
            self.thickness_percentage = np.asarray(self.thickness_percentage,
                                                   dtype=float)
        if not isinstance(self.chord_len, np.ndarray):
            self.chord_len = np.asarray(self.chord_len, dtype=float)
        if not isinstance(self.camber_max, np.ndarray):
            self.camber_max = np.asarray(self.camber_max, dtype=float)
        if not isinstance(self.thickness_max, np.ndarray):
            self.thickness_max = np.asarray(self.thickness_max, dtype=float)
        if self.chord_len < 0:
            raise ValueError('chord_len must be positive.')
        if self.camber_max < 0:
            raise ValueError('camber_max must be positive.')
        if self.thickness_max < 0:
            raise ValueError('thickness_max must be positive.')

        # Therefore the arrays camber_percentage and thickness_percentage
        # should have the same length of chord_percentage, equal to n_pos,
        # which is the number of cuts along the chord line
        if self.camber_percentage.shape != self.chord_percentage.shape:
            raise ValueError('camber_perc and chord_perc must have same shape.')
        if self.thickness_percentage.shape != self.chord_percentage.shape:
            raise ValueError(
                'thickness_perc and chord_perc must have same shape.')

    def _generate_coordinates(self):
        """
        Private method that generates the coordinates of a general airfoil profile, 
        starting from the chord percantages and the related nondimensional 
        camber and thickness. input data should be integrated with the information
        of chord length, camber and thickness of specific sections.
        """

        # compute the angular coefficient of the camber line at each chord
        # percentage and convert it from degrees to radiant
        n_pos = len(self.chord_percentage)
        m = np.zeros(n_pos)
        m[0] = 0
        for i in range(1, len(self.chord_percentage), 1):
            m[i] = (self.camber_percentage[i] -
                    self.camber_percentage[i - 1]) / (
                        self.chord_percentage[i] - self.chord_percentage[i - 1])

        m_angle = m * np.pi / 180
        self.xup_coordinates = np.zeros(n_pos)
        self.xdown_coordinates = np.zeros(n_pos)
        self.yup_coordinates = np.zeros(n_pos)
        self.ydown_coordinates = np.zeros(n_pos)

        #compute the coordinates starting from a single section data and parameters
        for j in range(0, n_pos, 1):
            self.xup_coordinates[j] = self.chord_percentage[j]
            self.xdown_coordinates[j] = self.chord_percentage[j]
            self.yup_coordinates[j] = (
                self.camber_percentage[j] * self.camber_max +
                self.thickness_percentage[j] * self.thickness_max *
                np.cos(m_angle[j])) * self.chord_len
            self.ydown_coordinates[j] = (
                self.camber_percentage[j] * self.camber_max -
                self.thickness_percentage[j] * self.thickness_max *
                np.cos(m_angle[j])) * self.chord_len

    def _generate_parameters(self):
        '''
        Private method to find parameters related to each section
        (chord percentages, camber max, camber percentages,
        thickness max and thickness percentage) starting from the
        xup, yup, xdown, ydown coordinates of the section.
        Useful for parametrization and deformation.
        '''
        self.chord_percentage = self.xup_coordinates

        camber = (self.yup_coordinates +
                  self.ydown_coordinates) / (2 * self.chord_len)
        self.camber_max = np.max(camber)
        self.camber_percentage = camber / self.camber_max

        delta_camber = np.zeros(len(camber))
        delta_x = np.zeros(len(camber))
        m_rad = np.zeros(len(camber))
        for i in range(1, len(camber), 1):
            delta_camber[
                i] = self.camber_percentage[i] - self.camber_percentage[i - 1]
            delta_x[i] = self.chord_percentage[i] - self.chord_percentage[i - 1]

        m_rad[1:] = delta_camber[1:] * np.pi / (delta_x[1:] * 180)

        thickness = (self.yup_coordinates - self.ydown_coordinates) / (
            2 * self.chord_len * np.cos(m_rad))
        self.thickness_max = np.max(thickness)
        self.thickness_percentage = thickness / self.thickness_max

    def _check_coordinates(self):
        """
        Private method that checks whether the airfoil coordinates defined (or
        computed starting from the section parameters) are provided correctly.

        We note that each array of coordinates must be consistent with the
        other arrays. The upper and lower surfaces should start from exactly
        the same point, the leading edge, and proceed on the way till the
        trailing edge. The trailing edge might have a non-zero thickness as
        in the case of some NACA-airfoils. In case of an open trailing edge,
        the average coordinate between upper and lower part is taken as the
        unique value.

        :raises ValueError: if either xup, xdown, yup, ydown is None
        :raises ValueError: if the 1D arrays xup, yup or xdown, ydown do not
            have the same length
        :raises ValueError: if array yup not greater than or equal array ydown
            element-wise
        :raises ValueError: if xdown[0] != xup[0] or ydown[0] != yup[0]
            or xdown[-1] != xup[-1]
        """
        if self.xup_coordinates is None:
            raise ValueError('object "xup" refers to an empty array.')
        if self.xdown_coordinates is None:
            raise ValueError('object "xdown" refers to an empty array.')
        if self.yup_coordinates is None:
            raise ValueError('object "yup" refers to an empty array.')
        if self.ydown_coordinates is None:
            raise ValueError('object "ydown" refers to an empty array.')

        if not isinstance(self.xup_coordinates, np.ndarray):
            self.xup_coordinates = np.asarray(self.xup_coordinates, dtype=float)
        if not isinstance(self.xdown_coordinates, np.ndarray):
            self.xdown_coordinates = np.asarray(self.xdown_coordinates,
                                                dtype=float)
        if not isinstance(self.yup_coordinates, np.ndarray):
            self.yup_coordinates = np.asarray(self.yup_coordinates, dtype=float)
        if not isinstance(self.ydown_coordinates, np.ndarray):
            self.ydown_coordinates = np.asarray(self.ydown_coordinates,
                                                dtype=float)

        # Therefore the arrays xup_coordinates and yup_coordinates must have
        # the same length = N, same holds for the arrays xdown_coordinates
        # and ydown_coordinates.
        if self.xup_coordinates.shape != self.yup_coordinates.shape:
            raise ValueError('xup and yup must have same shape.')
        if self.xdown_coordinates.shape != self.ydown_coordinates.shape:
            raise ValueError('xdown and ydown must have same shape.')

        # The condition yup_coordinates >= ydown_coordinates must be satisfied
        # element-wise to the whole elements in the mentioned arrays.
        if not all(
                np.greater_equal(self.yup_coordinates, self.ydown_coordinates)):
            raise ValueError('yup is not >= ydown elementwise.')

        if not self.xdown_coordinates[0] == self.xup_coordinates[0]:
            raise ValueError('(xdown[0]=xup[0]) not satisfied.')
        if not self.ydown_coordinates[0] == self.yup_coordinates[0]:
            raise ValueError('(ydown[0]=yup[0]) not satisfied.')
        if not self.xdown_coordinates[-1] == self.xup_coordinates[-1]:
            raise ValueError('(xdown[-1]=xup[-1]) not satisfied.')

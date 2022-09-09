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

    If coordinates are directly given as input:

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
    :param float chord_len: the length of the chord line of a certain airfoil section
    :param float camber_max: the maximum camber at a certain airfoil section
    :param float thickness_max: the maximum thickness at a certain airfoil section
    """
    def __init__(self, **kwargs):
        super(CustomProfile, self).__init__()

        if all([key in ['xup', 'yup', 'xdown', 'ydown'] for key in kwargs]):
            self.xup_coordinates = kwargs['xup']
            self.yup_coordinates = kwargs['yup']
            self.xdown_coordinates = kwargs['xdown']
            self.ydown_coordinates = kwargs['ydown']
            self._check_coordinates()

        elif set(kwargs.keys()) == set([
                'chord_perc', 'camber_perc', 'thickness_perc', 'chord_len',
                'camber_max', 'thickness_max'
        ]):

            self.chord_percentage = kwargs['chord_perc']
            self.camber_percentage = kwargs['camber_perc']
            self.thickness_percentage = kwargs['thickness_perc']
            self.chord_len = kwargs['chord_len']
            self.camber_max = kwargs['camber_max']
            self.thickness_max = kwargs['thickness_max']
            self._check_parameters()

        else:
            raise RuntimeError(
                """Input arguments should be the section coordinates
                (xup, yup, xdown, ydown) and chord_len (optional)
                or the section parameters (camber_perc, thickness_perc,
                camber_max, thickness_max, chord_perc, chord_len).""")


    def _check_parameters(self):
        """
        Private method that checks whether the airfoil parameters defined
        are provided correctly.
        In particular, the chord, camber and thickness percentages are
        consistent and have the same length.
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

    def _compute_orth_camber_coordinates(self):
        '''
        Compute the coordinates of points on upper and lower profile on the
        line orthogonal to the camber line.

        :return: x and y coordinates of section points on line orthogonal to
        camber line
        '''
        # Compute the angular coefficient of the camber line
        n_pos = self.chord_percentage.shape[0]
        m = np.zeros(n_pos)
        for i in range(1, n_pos, 1):
            m[i] = (self.camber_percentage[i]-
                self.camber_percentage[i-1])/(self.chord_percentage[i]-
                    self.chord_percentage[i-1])*self.camber_max/self.chord_len

        m_angle = np.arctan(m)

        xup_tmp = (self.chord_percentage*self.chord_len -
                self.thickness_percentage*np.sin(m_angle)*self.thickness_max/2)
        xdown_tmp = (self.chord_percentage*self.chord_len +
                self.thickness_percentage*np.sin(m_angle)*self.thickness_max/2)
        yup_tmp = (self.camber_percentage*self.camber_max +
                self.thickness_max/2*self.thickness_percentage*np.cos(m_angle))
        ydown_tmp =  (self.camber_percentage*self.camber_max -
                self.thickness_max/2*self.thickness_percentage*np.cos(m_angle))

        if xup_tmp[1]<0:
            xup_tmp[1], xdown_tmp[1] = xup_tmp[2]-1e-16, xdown_tmp[2]-1e-16
            yup_tmp[1], ydown_tmp[1] = yup_tmp[2]-1e-16, ydown_tmp[2]-1e-16

        return xup_tmp, xdown_tmp, yup_tmp, ydown_tmp



    def generate_coordinates(self):
        """
        Method that generates the coordinates of a general airfoil
        profile, starting from the chord percentages and the related
        nondimensional camber and thickness, the maximum values of thickness
        and camber.
        """

        self.xup_coordinates, self.xdown_coordinates, self.yup_coordinates, self.ydown_coordinates = self._compute_orth_camber_coordinates()

        self.ydown_coordinates = self.ydown_curve(
                self.chord_len*(self.chord_percentage).reshape(-1,1)).reshape(
                        self.chord_percentage.shape)
        self.xup_coordinates = self.chord_percentage * self.chord_len
        self.xdown_coordinates = self.xup_coordinates.copy()
        self.yup_coordinates = (2*self.camber_max*self.camber_percentage -
                self.ydown_coordinates)

        self.yup_coordinates[0] = 0
        self.yup_coordinates[-1] = 0
        self.ydown_coordinates[0] = 0
        self.ydown_coordinates[-1] = 0


    def adimensionalize(self):
        '''
        Rescale coordinates of upper and lower profiles of section such that
        coordinates on x axis are between 0 and 1.
        '''
        factor = abs(self.xup_coordinates[-1]-self.xup_coordinates[0])
        self.yup_coordinates *= 1/factor
        self.xdown_coordinates *= 1/factor
        self.ydown_coordinates *= 1/factor
        self.xup_coordinates *= 1/factor

    def generate_parameters(self):
        '''
        Method that generates the parameters of a general airfoil profile
        (chord length, chord percentages, camber max, thickness max, camber and
        thickness percentages), starting from the upper and lower
        coordinates of the section profile.
        '''
        n_pos = self.xup_coordinates.shape[0]

        self.chord_len = abs(np.max(self.xup_coordinates)-
                np.min(self.xup_coordinates))
        self.chord_percentage = self.xup_coordinates/self.chord_len
        camber = (self.yup_coordinates + self.ydown_coordinates)/2
        self.camber_max = abs(np.max(camber)-np.min(camber))
        self.camber_percentage = camber/self.camber_max

        n_pos = self.chord_percentage.shape[0]
        m = np.zeros(n_pos)
        for i in range(1, n_pos, 1):
            m[i] = (self.camber_percentage[i]-
                self.camber_percentage[i-1])/(self.chord_percentage[i]-
                    self.chord_percentage[i-1])*self.camber_max/self.chord_len
        m_angle = np.arctan(m)

        from scipy.optimize import newton

        ind_horizontal_camber = (np.sin(m_angle)==0)
        def eq_to_solve(x):
            spline_curve = self.ydown_curve(x.reshape(-1,1)).reshape(x.shape[0],)
            line_orth_camber = (camber[~ind_horizontal_camber] +
                    np.cos(m_angle[~ind_horizontal_camber])/
                    np.sin(m_angle[~ind_horizontal_camber])*(self.chord_len*
                        self.chord_percentage[~ind_horizontal_camber]-x))
            return spline_curve - line_orth_camber

        xdown_tmp = self.xdown_coordinates.copy()
        xdown_tmp[~ind_horizontal_camber] = newton(eq_to_solve,
                xdown_tmp[~ind_horizontal_camber])
        xup_tmp = 2*self.chord_len*self.chord_percentage - xdown_tmp
        ydown_tmp = self.ydown_curve(xdown_tmp.reshape(-1,1)).reshape(xdown_tmp.shape[0],)
        yup_tmp = 2*self.camber_max*self.camber_percentage - ydown_tmp
        if xup_tmp[1]<self.xup_coordinates[0]:
            xup_tmp[1], xdown_tmp[1] = xup_tmp[2], xdown_tmp[2]
            yup_tmp[1], ydown_tmp[1] = yup_tmp[2], ydown_tmp[2]

        thickness = np.sqrt((xup_tmp-xdown_tmp)**2 + (yup_tmp-ydown_tmp)**2)
        self.thickness_max = np.max(thickness)
        self.thickness_percentage = thickness/self.thickness_max

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
        if not np.allclose(self.ydown_coordinates[0], self.yup_coordinates[0]):
            raise ValueError('(ydown[0]=yup[0]) not satisfied.')
        if not self.xdown_coordinates[-1] == self.xup_coordinates[-1]:
            raise ValueError('(xdown[-1]=xup[-1]) not satisfied.')

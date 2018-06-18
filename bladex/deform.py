"""
Module to deform blade's parameters, based on a given parameter file.
"""

import numpy as np
import matplotlib.pyplot as plt
from .ndinterpolator import reconstruct_f, scipy_bspline
from .params import ParamFile


class Deformation(object):
    """
    Deform parameter curves {chord, pitch, rake, skew, camber} according to
    specific information passed through a parameter file.

    The module contains several methods that are able to:

        1. compute coordinates of the optimal control points, provided their
        number from the parameter file.

        2. update control points Y coordinates, provided the magnitude of the
        Y deformation from the parameter file.

        3. generate B-spline curve using the computed control points (before or
        after their deformation), provided npoints from the parameter file
        for spline estimation.

        4. plot parametric curves, with several options. Finally export a new
        parameter file containing the new deformed parameters.

    :param str paramfile: parameter file name
    :cvar str paramfile: parameter file name
    :cvar class param: class object instantiated by the module params, and
        contains all the attributes assigned by reading the parameter file.
        Possible attributes are:

        - `param.radii`
        - `param.parameters`
        - `param.nbasis`
        - `param.degree`
        - `param.deformations`

        First attribute is array defines the radial sections, while the
        remaining attributes are dictionaries with possible keys: `chord`,
        `pitch`, `rake`, `skew`, `camber`.

    :cvar dict deformed_parameters: dictionary that contains the deformed
        parameters at the same radial sections, provided some tolerance due to
        the spline interpolation. Possible dictionary keys are the parameters
        `chord`, `pitch`, `rake`, `skew`, `camber`. Default value is array of
        zeros with length equal to the radial sections.
    :cvar dict control_points: dictionary that contains the 2D coordinates of
        the control points associated with the B-spline parametric curve. The
        dictionary possible keys are the parameters `chord`, `pitch`, `rake`,
        `skew`, `camber`. Default value is None
    :cvar dict spline: dictionary that contains the B-spline interpolation for
        the parametric curves. The dictionary possible keys are the parameters
        `chord`, `pitch`, `rake`, `skew`, `camber`. Each value is a 2D numpy
        array containing the radii interpolations against the interpolations
        for one of the parameters mentioned in the dictionary keys. Default
        value is None.
    """

    def __init__(self, paramfile):
        self.paramfile = paramfile
        self.param = ParamFile()
        self.param.read_parameters(filename=paramfile)
        self.deformed_parameters = {
            'chord': np.zeros(self.param.radii.size),
            'pitch': np.zeros(self.param.radii.size),
            'rake': np.zeros(self.param.radii.size),
            'skew': np.zeros(self.param.radii.size),
            'camber': np.zeros(self.param.radii.size)
        }
        self.control_points = {
            'chord': None,
            'pitch': None,
            'rake': None,
            'skew': None,
            'camber': None
        }
        self.spline = {
            'chord': None,
            'pitch': None,
            'rake': None,
            'skew': None,
            'camber': None
        }

    @staticmethod
    def _optimum_control_points(X, Y, degree, nbasis, rbf_points):
        """
        Private static method that computes the optimum coordinates of the
        B-spline control points.

        :param array_like X: Array of original points of the parametric curve
            X-axis, usually array of the radii sections
        :param array_like Y: radial distribution of parameter `chord` or
            `pitch` or `rake` or `skew` or `camber`, corresponding to the
            radial sections in X
        :param int degree: degree of the B-spline construction for the
            parametric curve
        :param int nbasis: number of control points associated with the
            parametric curve
        :param int rbf_points: if specified greater than zero, then the X and Y
            arrays are interpolated using the Wendland C2 radial basis function
            to produce X and Y arrays with length = rbf_points. The larger
            number of rbf_points implies better estimation of the optimum
            control coordinates. To turn it off (i.e. compute control points
            based on original X, Y arrays) then insert 0. (Negative values or
            None results in same effect as zero)
        :return: control points 2D coordinates
        :rtype: numpy.ndarray
        """
        if not isinstance(rbf_points, int):
            # in case inserted as None, then converts to zero,
            # otherwise returns the inserted value. Useful when dealing with
            # the parameter as a flag
            rbf_points = int(rbf_points or 0)

        if rbf_points > 0:
            xx = np.linspace(X[0], X[-1], num=rbf_points)
            yy = np.zeros(rbf_points)
            reconstruct_f(
                original_input=X,
                original_output=Y,
                rbf_input=xx,
                rbf_output=yy,
                basis='beckert_wendland_c2_basis',
                radius=2.0)
            X = xx
            Y = yy

        A = np.zeros((len(X), nbasis))
        At = np.zeros((len(X), nbasis - 2))

        for i in range(nbasis):
            cv_new = np.zeros((nbasis, 3))
            cv_new[i, 0] = 1.
            # i-th basis function in the reference space
            A[:, i] = scipy_bspline(cv_new, A.shape[0], degree)[:, 0]

        # A tilde for the constraints on the first and last point
        At = A[:, 1:-1]
        # x and y of the ctrl points with constrained least square.
        # we subtract the contribution of the first and last basis function
        cvt_x = np.linalg.lstsq(
            At, X - A[:, 0] * X[0] - A[:, -1] * X[-1], rcond=-1)[0]
        cvt_y = np.linalg.lstsq(
            At, Y - A[:, 0] * Y[0] - A[:, -1] * Y[-1], rcond=-1)[0]

        # fill with the constraints the first and last point
        opt_ctrl = np.zeros((nbasis, 2))
        opt_ctrl[0, 0] = X[0]
        opt_ctrl[-1, 0] = X[-1]
        opt_ctrl[0, 1] = Y[0]
        opt_ctrl[-1, 1] = Y[-1]
        opt_ctrl[1:-1, 0] = cvt_x
        opt_ctrl[1:-1, 1] = cvt_y

        return opt_ctrl

    @staticmethod
    def _check_param(param):
        """
        Private static method that checks the passed parameter.

        :param str param: passed parameter to check. Valid values
            are: `chord`, `pitch`, `rake`, `skew`, `camber`
        :raises ValueError: if the param value is not one of the previous
        """
        params = ['chord', 'pitch', 'rake', 'skew', 'camber']
        if not param in params:
            raise ValueError(
                'Valid param values are: "chord", "pitch", "rake", "skew",'\
                 ' "camber".')

    def _check_control_points(self, param):
        """
        Private method to check if control points are computed.

        :param str param: passed parameter to check. Valid values
            are: `chord`, `pitch`, `rake`, `skew`, `camber`
        :raises ValueError: if the control points have None value, i.e. not
            computed
        """
        if self.control_points[param] is None:
            raise ValueError(
                'control_points has None value. You must compute them first.')

    def _check_spline(self, param):
        """
        Private method to check if spline interpolation is computed.

        :param str param: passed parameter to check. Valid values
            are: `chord`, `pitch`, `rake`, `skew`, `camber`
        :raises ValueError: if the spline of that parameter curve has None
            value, i.e. not computed
        """
        if self.spline[param] is None:
            raise ValueError(
                param + ' spline is None. You must first generate spline.')

    def _check_deformed(self, param):
        """
        Private method to check if the deformed parameters are computed.

        :param str param: passed parameter to check. Valid values
            are: `chord`, `pitch`, `rake`, `skew`, `camber`
        :raises ValueError: if the deformed parameters have array of zeros,
            i.e. not computed
        """
        if self.deformed_parameters[param].all() == 0:
            raise ValueError(param + ' deformed points are not computed.')

    def compute_control_points(self, param, rbf_points=1000):
        """
        Compute the control points 2D coordinates for one of the parametric
        curves.

        :param str param: parameter corresponding to the parametric curve.
            possible values are `chord`, `pitch`, `rake`, `skew`, `camber`
        :param int rbf_points: if greater than zero then the Wendland C2 radial
            basis function is used to interpolate the original arrays for the
            parametric curve, so that the control points are computed according
            to the interpolated arrays. Needless to mention that longer arrays
            would produce better estimation of the control points optimum
            coordinates. In order to turn off the rbf interpolation: specify
            either 0 or -1 (Also a None value can be used too). Default value
            is 1000
        """
        self._check_param(param=param)
        self.control_points[param] = self._optimum_control_points(
            X=self.param.radii,
            Y=self.param.parameters[param],
            degree=self.param.degree[param],
            nbasis=self.param.nbasis[param],
            rbf_points=rbf_points)

    def update_control_points(self, param):
        """
        Update the control point Y coordinate with the deformation values
        specified in the parameter file.

        :param str param: parameter corresponding to the parametric curve.
            possible values are `chord`, `pitch`, `rake`, `skew`, `camber`
        """
        self._check_param(param=param)
        self._check_control_points(param=param)

        if not self.control_points[param].shape[0] == len(
                self.param.deformations[param]):
            raise ValueError(
                'array of deformations must equal to number of control points'
            )

        for i in range(self.control_points[param].shape[0]):
            self.control_points[param][i, 1] += self.param.deformations[param][
                i]

    def generate_spline(self, param):
        """
        Generate the B-spline interpolations, using the information: `degree`,
        `npoints` from the parameter file, as well as the computed 2D
        coordinates of the control points.

        :param str param: parameter corresponding to the parametric curve.
            possible values are `chord`, `pitch`, `rake`, `skew`, `camber`
        """
        self._check_param(param=param)
        self._check_control_points(param=param)

        self.spline[param] = scipy_bspline(
            cv=self.control_points[param],
            npoints=self.param.npoints[param],
            degree=self.param.degree[param])

    def compute_deformed_parameters(self, param, tol=1e-3):
        """
        This method uses the spline npoints interpolation of the parametric
        curve to extract the parameters corresponding to the radial
        distribution of the original undeformed array. Therefore the resulting
        deformed parameters should be arrays of same length like that of the
        original parameters.

        :param str param: parameter corresponding to the parametric curve.
            possible values are `chord`, `pitch`, `rake`, `skew`, `camber`
        :param float tol: tolerance required to find the B-spline estimation
            within the neighborhood of each of the radii sections. It is
            important to specify the value carefully as it depends on the order
            of the original array values, as well as the number of points for
            the spline interpolations. Default value is 1e-3
        """
        self._check_param(param=param)
        self._check_spline(param=param)

        for i, val in enumerate(self.param.radii):
            index = np.where(np.fabs(self.spline[param][:, 0] - val) < tol)[0]
            if len(index) == 0:
                raise ValueError(
                    'Could not compute deformed parameter "' + param +
                    '" at radius "' + str(val) +
                    '". Either increase the tolerance for that parameter, or'\
                    ' increase the spline npoints in the parameter file.'
                )
            if index.shape[0] > 1:
                # In case more neighbors are found, then take first value only.
                index = index[0]
            self.deformed_parameters[param][i] = self.spline[param][index, 1]

    def compute_all(self,
                    rbf_points=1000,
                    tol_chord=1e-3,
                    tol_pitch=1e-3,
                    tol_rake=1e-3,
                    tol_skew=1e-3,
                    tol_camber=1e-3):
        """
        Computes everything:
            - control points 2D coordinates
            - deformed control points
            - spline npoints interpolations
            - deformed parameters of the original arrays

        The previous procedure is applied for all the parameters: `chord`,
        `pitch`, `rake`, `skew`, `camber`

        :param int rbf_points: if greater than zero then the Wendland C2 radial
            basis function is used to interpolate the original arrays for the
            parametric curve, so that the control points are computed according
            to the interpolated arrays. Needless to mention that longer arrays
            would produce better estimation of the control points optimum
            coordinates. In order to turn off the rbf interpolation then
            specify either 0 or -1 (Also a None value can be used too). Default
            value is 1000
        :param float tol_chord: tolerance used to extract the chord radial
            distribution for the deformed B-spline interpolation. Default value
            is 1e-3
        :param float tol_pitch: tolerance used to extract the pitch radial
            distribution for the deformed B-spline interpolation. Default value
            is 1e-3
        :param float tol_rake: tolerance used to extract the rake radial
            distribution for the deformed B-spline interpolation. Default value
            is 1e-3
        :param float tol_skew: tolerance used to extract the skew radial
            distribution for the deformed B-spline interpolation. Default value
            is 1e-3
        :param float tol_camber: tolerance used to extract the camber radial
            distribution for the deformed B-spline interpolation. Default value
            is 1e-3

        """
        tols = {
            'chord': tol_chord,
            'pitch': tol_pitch,
            'rake': tol_rake,
            'skew': tol_skew,
            'camber': tol_camber
        }
        params = ['chord', 'pitch', 'rake', 'skew', 'camber']
        for param in params:
            self.compute_control_points(param=param, rbf_points=rbf_points)
            self.update_control_points(param=param)
            self.generate_spline(param=param)
            self.compute_deformed_parameters(param=param, tol=tols[param])

    def plot(self,
             param,
             original=True,
             ctrl_points=True,
             spline=True,
             rbf=False,
             rbf_points=500,
             deformed=False,
             outfile=None):
        """
        Plot the parametric curve. Several options can be specified.

        :param str param: parameter corresponding to the parametric curve
            needs to be plotted. possible values are `chord`, `pitch`, `rake`,
            `skew`, `camber`
        :param bool original: if True, then plot the original points of the
            parameter at the radii sections. Default value is True
        :param bool ctrl_points: if True, then plot the control points of
            that parametric curve. Default value is True
        :param bool spline: If True, then plot the B-spline interpolation of
            the parametric curve. Default value is True
        :param bool rbf: if True, then plot the radial basis functions
            interpolation of the parametric curve. Default value is True
        :param int rbf_points: number of points used for the rbf interpolation,
            if the flag `rbf` is set True. Beware that this argument does not
            have the same function of that when computing the control points,
            although both uses the radial basis function interpolation with
            the Wendland basis. Default value is 500
        :param bool deformed: if True, then plot the deformed points of the
            parameter radial distribution, estimated using the B-spline
            interpolations within a given tolerance. Default value is False
        :param str outfile: if string is passed, then the plot is saved
            with that name. If the value is None, then the plot is shown on
            the screen. Default value is None
        """
        self._check_param(param=param)

        plt.figure()

        if original:
            plt.plot(
                self.param.radii,
                self.param.parameters[param],
                'o',
                label='original points')

        if ctrl_points:
            self._check_control_points(param=param)
            plt.plot(
                self.control_points[param][:, 0],
                self.control_points[param][:, 1],
                '*-',
                label='control points')

        if spline:
            self._check_spline(param=param)
            plt.plot(
                self.spline[param][:, 0],
                self.spline[param][:, 1],
                label='spline')

        if rbf:
            xx = np.linspace(
                self.param.radii[0], self.param.radii[-1], num=rbf_points)
            yy = np.zeros(rbf_points)
            reconstruct_f(
                original_input=self.param.radii,
                original_output=self.param.parameters[param],
                rbf_input=xx,
                rbf_output=yy,
                basis='beckert_wendland_c2_basis',
                radius=2.0)
            plt.plot(xx, yy, label='rbf')

        if deformed:
            self._check_deformed(param=param)
            plt.plot(
                self.param.radii,
                self.deformed_parameters[param],
                '+',
                label='deformed points')

        plt.grid(linestyle='dotted')
        plt.title(param + ' curve')
        plt.legend()

        if outfile:
            if not isinstance(outfile, str):
                raise ValueError('Output file name must be string.')
            plt.savefig(outfile)
        else:
            plt.show()

    def export_param_file(self, outfile='parameters_mod.prm'):
        """
        Export a new parameter file with the new deformed parameters, while
        all other values are kept the same as in the original parameter file
        with the undeformed parameters. In the new parameter file (i.e. with
        deformed parameters) the deformations arrays become array of zeros.

        :param str outfile: file name to be written out
        """

        prm = ParamFile()
        prm.radii = self.param.radii
        params = ['chord', 'pitch', 'rake', 'skew', 'camber']
        for param in params:
            # If the original parameter file specifies zero deformations for
            # some parameter, then the method writes out deformed parameters
            # a with radial distributionfrom the original parameter file;
            # otherwise it writes out the computed deformed parameters
            if np.all(self.param.deformations[param] == 0):
                prm.parameters[param] = self.param.parameters[param]
            else:
                prm.parameters[param] = self.deformed_parameters[param]
            prm.nbasis[param] = self.param.nbasis[param]
            prm.degree[param] = self.param.nbasis[param]
            prm.npoints[param] = self.param.npoints[param]
            prm.deformations[param] = np.zeros(self.param.nbasis[param])

        prm.write_parameters(filename=outfile)

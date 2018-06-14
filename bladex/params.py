"""
Module to read and write a parameter file
which can be used for parameters deformations.
"""
try:
    import configparser as configparser
except ImportError:
    import ConfigParser as configparser
import os
import numpy as np


class ParamFile(object):
    """
    Read and Write a parameter file

    :cvar array_like radii: contains radii values of the blade sectional
        profiles, starting from the hub. Default value is None
    :cvar dict parameters: dictionary that contains the radial distribution
        of parameters `chord`, `pitch`, `rake`, `skew`, `camber` at specific
        blade sections. Each element of the dictionary is an array_like of
        length equals to that of the array radii. Both parameters `chord` and
        `camber` descibes the chord length and the camber of the 2D foil
        representing the blade section. Possible dictionary keys are `chord`,
        `pitch`, `rake`, `skew`, `camber`. Default values are None
    :cvar dict nbasis: dictionary that contains number of control points for
        each parameter. Possible dictionary keys are: `chord`, `pitch`, `rake`,
        `skew`, `camber`
    :cvar dict degree: dictionary that contains degree of the BSpline to be
        constructed according to the parameter radial distribution. Possible
        dictionary keys are: `chord`, `pitch`, `rake`, `skew`, `camber`
    :cvar dict npoints: dictionary that contains number of points to be
        evaluated using the BSpline interpolation, for each of the parameter
        curves. Possible dictionary keys for the parameters are: `chord`,
        `pitch`, `rake`, `skew`, `camber`
    :cvar dict deformations: dictionary that contains the control points
        Y-deformations of the parameter BSpline curve. Possible dictionary
        keys are: `chord`, `pitch`, `rake`, `skew`, `camber`
    """

    def __init__(self):
        self.radii = None
        self.parameters = {
            'chord': None,
            'pitch': None,
            'rake': None,
            'skew': None,
            'camber': None
        }
        self.nbasis = {
            'chord': 10,
            'pitch': 10,
            'rake': 10,
            'skew': 10,
            'camber': 10
        }
        self.degree = {
            'chord': 3,
            'pitch': 3,
            'rake': 3,
            'skew': 3,
            'camber': 3
        }
        self.npoints = {
            'chord': 500,
            'pitch': 500,
            'rake': 500,
            'skew': 500,
            'camber': 500
        }
        self.deformations = {
            'chord': np.zeros(self.nbasis['chord']),
            'pitch': np.zeros(self.nbasis['chord']),
            'rake': np.zeros(self.nbasis['chord']),
            'skew': np.zeros(self.nbasis['chord']),
            'camber': np.zeros(self.nbasis['chord'])
        }

    def _check_params(self):
        """
        Private method that is called while writing a parameter file.

        1. The method checks if the user specifies a radii array, otherwise it
        raises exception.

        2. In case no values assigned to any of the remaining parameter arrays
        then an array of zeros is assigned, in which its length is equal
        to the radii array.

        3. Any array that is not numpy is converted to be one. Finally, in case
        the user specifies the parameter array values but not with same length,
        then an exception is raised.
        """
        if self.radii is None:
            raise ValueError('Array radii can not have a None value.')

        if not isinstance(self.radii, np.ndarray):
            self.radii = np.asarray(self.radii)

        for param in ['chord', 'pitch', 'rake', 'skew', 'camber']:
            # If any parameter is not inserted then assign that parameter
            # with array of zeros. This is useful in case the user is
            # interested in only one or few parameters, while not interested
            # in (or does have information about) the remaining ones.
            if self.parameters[param] is None:
                self.parameters[param] = np.zeros(self.radii.shape[0])

            # In case inserted parameters are not numpy arrays
            if not isinstance(self.parameters[param], np.ndarray):
                self.parameters[param] = np.asarray(self.parameters[param])

            # check the case if user inserts inhomogeneous radial distributions
            if not self.parameters[param].shape == self.radii.shape:
                raise ValueError(
                    'Array ' + param + ' must have same shape of array radii.')

            # If inserted deformations array not correspond to nbasis
            if not self.nbasis[param] == len(self.deformations[param]):
                raise ValueError(
                    param + ' deformations must correspond to nbasis.')

    def read_parameters(self, filename='parameters.prm'):
        """
        Reads in the parameters file and fill the self structure.

        :param str filename: parameters file to be read in. Default value is
            parameters.prm
        """
        if not isinstance(filename, str):
            raise TypeError('filename must be a string')

        # Checks if the parameters file exists. If not it writes a default
        # parameters file with zero deformations for all parameters at uniform
        # radial sections.
        if not os.path.isfile(filename):
            self.radii = np.arange(0.3, 1.1, 0.1)
            self.write_parameters(filename=filename)

        config = configparser.RawConfigParser()
        config.read(filename)

        radii = config.get('Original parameters', 'Radial sections')
        lines = radii.split('\n')
        self.radii = np.zeros(len(lines))
        for j, line in enumerate(lines):
            if len(line.split()) > 1:
                raise ValueError(
                    'Radial sections must have single value at each section.')
            self.radii[j] = float(line)

        params = ['chord', 'pitch', 'rake', 'skew', 'camber']
        for param in params:
            parameters = config.get('Original parameters',
                                    'Radial distribution of ' + param)
            lines = parameters.split('\n')
            self.parameters[param] = np.zeros(len(lines))
            for j, line in enumerate(lines):
                if len(line.split()) > 1:
                    raise ValueError(
                        param +
                        ' radial distribution must have single value at each'\
                        ' radial section.'
                    )
                self.parameters[param][j] = float(line)

        if not (len(self.radii) ==
                len(self.parameters['chord']) ==
                len(self.parameters['pitch']) ==
                len(self.parameters['rake']) ==
                len(self.parameters['skew']) ==
                len(self.parameters['camber'])):
            raise ValueError(
                'Arrays "radii", "chord", "pitch", "rake", "skew", "camber"'\
                ' must have same length.'
            )

        section_all = [
            'Chord B-Spline', 'Pitch B-Spline', 'Rake B-Spline',
            'Skew B-Spline', 'Camber B-Spline'
        ]

        for section, param in zip(section_all, params):
            self.degree[param] = config.getint(section, 'spline degree')
            self.npoints[param] = config.getint(section, 'spline npoints')
            self.nbasis[param] = config.getint(section,
                                               'number of control points')

            deformations = config.get(section, 'control points Y-deformations')
            if bool(deformations) is False:
                raise ValueError('control points Y-deformations in section [' +
                                 section + '] must be non-empty.')
            lines = deformations.split('\n')
            self.deformations[param] = np.zeros(len(lines))
            for j, line in enumerate(lines):
                if len(line.split()) > 1:
                    raise ValueError(
                        'You can pass only one value for each control point',
                        ' Y-deformation of the ' + param)
                self.deformations[param][j] = float(line)
            if not len(self.deformations[param]) == self.nbasis[param]:
                raise ValueError(param + ' has nbasis not equal deformations.')

    def write_parameters(self, filename='parameters.prm'):
        """
        This method writes a parameters file (.prm) called `filename` and fills
        it with all the parameters class members. Default value is
        parameters.prm.

        :param str filename: parameters file to be written out.
        :param bool param_pptc: if True, then the parameter arrays are replaced
            with that of the benchmark PPTC propeller.
        """
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")

        self._check_params()

        output_string = ""

        output_string += '\n[Original parameters]\n'
        output_string += '# This section describes the radial distributions'\
                        ' of the parameters:\n'
        output_string += '# "chord lengths", "pitch", "rake", "skew angles",'\
                         ' and "camber"\n'
        output_string += '# at given radial sections.\n'

        output_string += '\nRadial sections = '
        offset = 1
        for radius in self.radii:
            output_string += offset * ' ' + str(radius) + '\n'
            offset = 19

        params = ['chord', 'pitch', 'rake', 'skew', 'camber']
        gaps = [0, 0, 1, 1, -1]

        for param, gap in zip(params, gaps):
            offset = 1 + gap
            output_string += '\nRadial distribution of ' + param + ' = '
            for parameter in self.parameters[param]:
                output_string += offset * ' ' + str(parameter) + '\n'
                offset = 32

        sections = [
            '[Chord B-Spline]', '[Pitch B-Spline]', '[Rake B-Spline]',
            '[Skew B-Spline]', '[Camber B-Spline]'
        ]

        for section, param in zip(sections, params):
            output_string += '\n\n' + section + '\n'
            output_string += '# This section describes the B-Spline'\
            ' construction of the RADII -- ' + param.upper() + ' curve.\n'
            output_string += '\n# degree of the B-Spline curve\n'
            output_string += 'spline degree: {}\n'.format(
                str(self.degree[param]))
            output_string += '\n# number of points to be evaluated with the'\
             ' B-Spline interpolation\n'
            output_string += 'spline npoints: {}\n'.format(
                str(self.npoints[param]))
            output_string += '\n# number of the control points\n'
            output_string += 'number of control points: {}\n'.format(
                str(self.nbasis[param]))
            output_string += '\n# Y-deformations of the control points.\n'
            output_string += 'control points Y-deformations = '
            offset = 1
            for i in range(self.nbasis[param]):
                output_string += offset * ' ' + str(
                    self.deformations[param][i]) + '\n'
                offset = 33

        with open(filename, 'w') as f:
            f.write(output_string)

    def __str__(self):
        """
        This method prints all the parameters on the screen. Its purpose is
        for debugging.
        """
        string = ''
        string += '\nradii = {}\n'.format(self.radii)
        params = ['chord', 'pitch', 'rake', 'skew', 'camber']
        for param in params:
            string += '\n\n' + param + ' = {}\n'.format(self.parameters[param])
            string += '\n' + param + ' degree = {}\n'.format(
                self.degree[param])
            string += param + ' npoints = {}\n'.format(
                self.npoints[param])
            string += param + ' nbasis = {}\n'.format(self.nbasis[param])
            string += param + ' control points deformations =\n'
            string += '{}\n'.format(self.deformations[param])
        return string

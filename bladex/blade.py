"""
Module for the blade bottom-up parametrized construction.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .intepolatedface import InterpolatedFace

class Blade(object):
    """
    Bottom-up parametrized blade construction.

    Given the following parameters of a propeller blade:

        - :math:`(X, Y)` coordinates of the blade cylindrical sections after
          being expanded in 2D to create airfoils.

        - Radial distance :math:`(r_i)` from the propeller axis of rotation
          to each cylindrical section.

        - Pitch angle :math:`(\\varphi)`, for each cylindrical section.

        - Rake :math:`(k)`, in distance units, for each cylindrical section.

        - Skew angle :math:`(\\theta_s)`, for each cylindrical section.

    then, a bottom-up construction procedure is performed by applying series of
    transformation operations on the airfoils according to the provided
    parameters, to end up with a 3D CAD model of the blade, which can be
    exported into IGES format. Also surface or volume meshes can be obtained.

    Useful definitions on the propeller geometry:

        - Blade cylindrical section: the cross section of a blade cut by a
          cylinder whose centerline is the propeller axis of rotation.
          We may also refer as "radial section".

        - Pitch :math:`(P)`: the linear distance that a propeller would move in
          one revolution with no slippage. The geometric pitch angle
          :math:`(\\varphi)` is the angle between the pitch reference line
          and a line perpendicular to the propeller axis of rotation.

        .. math::
            tan (\\varphi) = \\frac{\\text{pitch}}
            {\\text{propeller circumference}} = \\frac{P}{2 \\pi r}

        - Rake: the fore or aft slant of the blade with respect to a line
          perpendicular to the propeller axis of rotation.

        - Skew: the transverse sweeping of a blade such that viewing the blades
          from fore or aft would show an asymmetrical shape.

    References:

    - Carlton, J. Marine propellers and propulsion. Butterworth-Heinemann, 2012.
      http://navalex.com/downloads/Michigan_Wheel_Propeller_Geometry.pdf

    - J. Babicz. Wartsila Encyclopedia of Ship Technology. 2nd ed. Wartsila
      Corporation. 2015.

    .. _transformation_operations:

    Transformation operations according to the provided parameters:

    .. figure:: ../../readme/transformations.png
       :scale: 75 %
       :alt: transformations

       Airfoil 2D transformations corresponding to the pitch, rake, and skew of
       the blade expanded cylindrical section.

    --------------------------

    :param array_like sections: 1D array, each element is an object of the
        BaseProfile class at specific radial section.
    :param array_like radii: 1D array, contains the radii values of the
        sectional profiles.
    :param array_like chord_lengths: 1D array, contains the value of the
        airfoil's chord length for each radial section of the blade.
    :param array_like pitch: 1D array, contains the local pitch values
        (in unit length) for each radial section of the blade.
    :param array_like rake: 1D array, contains the local rake values for each
        radial section of the blade.
    :param array_like skew_angles: 1D array, contains the skew angles
        (in degrees) for each radial section of the blade.

    Note that, each of the previous array_like parameters must be consistent
    with the other parameters in terms of the radial ordering of the blade
    sections. In particular, an array_like elements must follow the radial
    distribution of the blade sections starting from the blade root and ends up
    with the blade tip since the blade surface generator depends on that order.

    Finally, beware that the profiles class objects in the array 'sections'
    undergo several transformations that affect their coordinates. Therefore
    the array must be specific to each blade class instance. For example, if
    we generate 12 sectional profiles using NACA airfoils and we need to use
    them in two different blade classes, then we should instantiate two class
    objects for the profiles, as well as the blade. The following example
    explains the fault and the correct implementations (assuming we already
    have the arrays radii, chord, pitch, rake, skew):

    INCORRECT IMPLEMENTATION:

    >>> sections = [bladex.profiles.NacaProfile(digits='0012', n_points=240,
                    cosine_spacing=True) for i in range(12)]
    >>> blade_1 = Blade(
                    sections=sections,
                    radii=radii,
                    chord_lengths=chord,
                    pitch=pitch,
                    rake=rake,
                    skew_angles=skew)
    >>> blade_1.apply_transformations()
    >>> blade_2 = Blade(
                    sections=sections,
                    radii=radii,
                    chord_lengths=chord,
                    pitch=pitch,
                    rake=rake,
                    skew_angles=skew)
    >>> blade_2.apply_transformations()

    The previous implementation would lead into erroneous blade coordinates due
    to the transformed data in the array sections

    CORRECT IMPLEMENTATION:

    >>> sections_1 = [bladex.profiles.NacaProfile(digits='0012', n_points=240,
                      cosine_spacing=True) for i in range(12)]
    >>> sections_2 = [bladex.profiles.NacaProfile(digits='0012', n_points=240,
                      cosine_spacing=True) for i in range(12)]
    >>> blade_1 = Blade(
                    sections=sections_1,
                    radii=radii,
                    chord_lengths=chord,
                    pitch=pitch,
                    rake=rake,
                    skew_angles=skew)
    >>> blade_1.apply_transformations()
    >>> blade_2 = Blade(
                    sections=sections_2,
                    radii=radii,
                    chord_lengths=chord,
                    pitch=pitch,
                    rake=rake,
                    skew_angles=skew)
    >>> blade_2.apply_transformations()
    """

    def __init__(self, sections, radii, chord_lengths, pitch, rake,
                 skew_angles):
        # Data are given in absolute values
        self.sections = sections
        self.n_sections = len(sections)
        self.radii = radii
        self.chord_lengths = chord_lengths
        self.pitch = pitch
        self.rake = rake
        self.skew_angles = skew_angles
        self._check_params()

        self.conversion_factor = 1000  # to convert units if necessary
        self.reset()

    def reset(self):
        """
        Reset the blade coordinates and generated faces.
        """
        self.blade_coordinates_up = []
        self.blade_coordinates_down = []

        self.upper_face = None
        self.lower_face = None
        self.tip_face = None
        self.root_face = None

    def build(self, reflect=True):
        """
        Generate a bottom-up constructed propeller blade without applying any
        transformations on the airfoils.

        The method directly constructs the blade CAD model by interpolating
        the given 3D coordinates of the blade sections.

        """
        self.apply_transformations(reflect=reflect)

        blade_coordinates_up = self.blade_coordinates_up * self.conversion_factor
        blade_coordinates_down = self.blade_coordinates_down * self.conversion_factor

        self.upper_face = InterpolatedFace(blade_coordinates_up).face
        self.lower_face = InterpolatedFace(blade_coordinates_down).face
        self.tip_face = InterpolatedFace(np.stack([
            blade_coordinates_up[-1],
            blade_coordinates_down[-1]
        ])).face
        self.root_face = InterpolatedFace(np.stack([
            blade_coordinates_up[0],
            blade_coordinates_down[0]
        ])).face

    def _check_params(self):
        """
        Private method to check if all the blade arguments are numpy.ndarrays
        with the same shape.
        """
        if not isinstance(self.sections, np.ndarray):
            self.sections = np.asarray(self.sections)
        if not isinstance(self.radii, np.ndarray):
            self.radii = np.asarray(self.radii)
        if not isinstance(self.chord_lengths, np.ndarray):
            self.chord_lengths = np.asarray(self.chord_lengths)
        if not isinstance(self.pitch, np.ndarray):
            self.pitch = np.asarray(self.pitch)
        if not isinstance(self.rake, np.ndarray):
            self.rake = np.asarray(self.rake)
        if not isinstance(self.skew_angles, np.ndarray):
            self.skew_angles = np.asarray(self.skew_angles)

        if not (self.sections.shape == self.radii.shape ==
                self.chord_lengths.shape == self.pitch.shape == self.rake.shape
                == self.skew_angles.shape):
            raise ValueError('Arrays {sections, radii, chord_lengths, pitch, '\
            'rake, skew_angles} do not have the same shape.')

    @property
    def pitch_angles(self):
        """
        Return the pitch angle from the linear pitch for all blade sections.

        :return: pitch angle in radians
        :rtype: numpy.ndarray
        """
        return np.arctan(self.pitch / (2.0 * np.pi * self.radii))

    @property
    def induced_rake(self):
        """
        Returns the induced rake from skew for all the blade sections, according
        to :ref:`mytransformation_operations`.

        :return: induced rake from skew
        :rtype: numpy.ndarray
        """
        return self.radii * np.radians(self.skew_angles) * np.tan(
            self.pitch_angles)

    def _planar_to_cylindrical(self):
        """
        Private method that transforms the 2D planar airfoils into 3D
        cylindrical sections.

        The cylindrical transformation is defined by the following formulas:

            - :math:`x = x_{i} \\qquad \\forall x_i \\in X`

            - :math:`y = r \\sin\\left( \\frac{y_i}{r} \\right) \\qquad
              \\forall y_i \\in Y`

            - :math:`z = r \\cos\\left( \\frac{y_i}{r} \\right) \\qquad
              \\forall y_i \\in Y`

        After transformation, the method also fills the numpy.ndarray
        "blade_coordinates_up" and "blade_coordinates_down" with the new
        :math:`(X, Y, Z)` coordinates.
        """

        self.blade_coordinates_down = []
        self.blade_coordinates_up = []

        for section, radius in zip(self.sections[::-1], self.radii[::-1]):
            theta_up = section.yup_coordinates / radius
            theta_down = section.ydown_coordinates / radius

            y_section_up = radius * np.sin(theta_up)
            y_section_down = radius * np.sin(theta_down)

            z_section_up = radius * np.cos(theta_up)
            z_section_down = radius * np.cos(theta_down)

            self.blade_coordinates_up.append(
                np.array([section.xup_coordinates, y_section_up, z_section_up]))
            self.blade_coordinates_down.append(
                np.array(
                    [section.xdown_coordinates, y_section_down,
                     z_section_down]))

        self.blade_coordinates_down = np.stack(self.blade_coordinates_down)
        self.blade_coordinates_up = np.stack(self.blade_coordinates_up)

    def apply_transformations(self, reflect=True):
        """
        Generate a bottom-up constructed propeller blade based on the airfoil
        transformations, see :ref:`mytransformation_operations`.

        The order of the transformation operations is as follows:

            1. Translate airfoils by reference points into origin.

            2. Scale X, Y coordinates by a factor of the chord length. Also
               reflect the airfoils if necessary.

            3. Rotate the airfoils counter-clockwise according to the local
               pitch angles. Beware of the orientation system.

            4. Translate airfoils along X-axis by a magnitude of the local
               rake. Perform another translation for the skew-induced rake.

            5. Translate airfoils along Y-axis by a magnitude of the skewness.

            6. Transform the 2D airfoils into cylindrical sections, by laying
               each foil on a cylinder of radius equals to the section radius,
               and the cylinder axis is the propeller axis of rotation.

        :param bool reflect: if true, then reflect the coordinates of all the
            airfoils about both X-axis and Y-axis. Default value is True.

        We note that the implemented transformation operations with the current
        Cartesian coordinate system shown in :ref:`mytransformation_operations`
        assumes a right-handed propeller. In case of a desired left-handed
        propeller the user can either change the code for the negative
        Z-coordinates in the cylindrical transformation (i.e.
        `_planar_to_cylindrical` private method), or manipulating the
        orientation of the generated CAD with respect to the hub.
        """
        for i in range(self.n_sections):
            # Translate reference point into origin
            self.sections[i].translate(-self.sections[i].reference_point)

            if reflect:
                self.sections[i].reflect()

            # Scale the unit chord to actual length.
            self.sections[i].scale(self.chord_lengths[i])

            # Rotate according to the pitch angle.
            # Since the current orientation system is not standard (It is
            # left-handed Cartesian orientation system, where Y-axis points
            # downwards and X-axis points to the right), the standard rotation
            # matrix yields clockwise rotation.
            self.sections[i].rotate(
                rad_angle=np.pi / 2.0 - self.pitch_angles[i])

            # Translation due to skew.
            self.sections[i].translate(
                [0, -self.radii[i] * np.radians(self.skew_angles[i])])

            # Translate due to total rake.
            self.sections[i].translate(
                [-(self.rake[i] + self.induced_rake[i]), 0])

        self._planar_to_cylindrical()

    def rotate(self, deg_angle=None, rad_angle=None, axis='x'):
        """
        3D counter clockwise rotation about the specified axis of the Cartesian
        coordinate system, which is the axis of rotation of the propeller hub.

        The rotation matrix, :math:`R(\\theta)`, is used to perform rotation
        in the 3D Euclidean space about the specified axis, which is
        -- by default -- the x axis.

        when the axis of rotation is the x-axis :math: `R(\\theta)` is defined
        by:

        .. math::
             \\left(\\begin{matrix} 1 & 0 & 0 \\\\
             0 & cos (\\theta) & - sin (\\theta) \\\\
             0 & sin (\\theta) & cos (\\theta) \\end{matrix}\\right)

        Given the coordinates of point :math:`P` such that

        .. math::
            P = \\left(\\begin{matrix} x \\\\
            y \\\\ z \\end{matrix}\\right),

        Then, the rotated coordinates will be:

        .. math::
            P^{'} = \\left(\\begin{matrix} x^{'} \\\\
                     y^{'} \\\\ z^{'} \\end{matrix}\\right)
                  = R (\\theta) \\cdot P

        :param float deg_angle: angle in degrees. Default value is None
        :param float rad_angle: angle in radians. Default value is None
        :param string axis: cartesian axis of rotation. Default value is 'x'
        :raises ValueError: if both rad_angle and deg_angle are inserted,
            or if neither is inserted

        """
        if len(self.blade_coordinates_up) == 0:
            raise ValueError('You must apply transformations before rotation.')

        # Check rotation angle
        if deg_angle is not None and rad_angle is not None:
            raise ValueError(
                'You have to pass either the angle in radians or in degrees,' \
                ' not both.')

        if rad_angle is not None:
            cosine = np.cos(rad_angle)
            sine = np.sin(rad_angle)
        elif deg_angle is not None:
            cosine = np.cos(np.radians(deg_angle))
            sine = np.sin(np.radians(deg_angle))
            rad_angle = deg_angle * np.pi / 180
        else:
            raise ValueError(
                'You have to pass either the angle in radians or in degrees.')

        # Rotation is always about the X-axis, which is the center if the hub
        # according to the implemented transformation procedure
        if axis == 'x':
            rot_matrix = np.array([1, 0, 0, 0, cosine, -sine, 0, sine,
                                cosine]).reshape((3, 3))

        elif axis=='y':
            rot_matrix = np.array([cosine, 0, -sine, 0, 1, 0, sine, 0,
                            cosine]).reshape((3, 3))

        elif axis=='z':
            rot_matrix = np.array([cosine, -sine, 0, sine, cosine, 0,
                0, 0, 1]).reshape((3, 3))
        else:
            raise ValueError('Axis must be either x, y, or z.')

        self.blade_coordinates_up = np.einsum('ij, kjl->kil',
            rot_matrix, self.blade_coordinates_up)
        self.blade_coordinates_down = np.einsum('ij, kjl->kil',
            rot_matrix, self.blade_coordinates_down)

        # TODO: working but ugly
        for id, face in enumerate([self.upper_face, self.lower_face,
                     self.tip_face, self.root_face]):
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
            from OCC.Core.gp import gp_Pnt
            from OCC.Core.gp import gp_Dir
            from OCC.Core.gp import gp_Ax1
            from OCC.Core.gp import gp_Trsf
            
            origin = gp_Pnt(0, 0, 0)
            if axis == 'y':
                direction = gp_Dir(0, 1, 0)
            elif axis == 'z':
                direction = gp_Dir(0, 0, 1)
            elif axis == 'x':
                direction = gp_Dir(1, 0, 0)
            else:
                raise ValueError('Axis must be either x, y, or z.')
            ax1 = gp_Ax1(origin, direction)
            trsf = gp_Trsf()
            trsf.SetRotation(ax1, rad_angle)

            brep_tr = BRepBuilderAPI_Transform(face, trsf, True, True)
            face = brep_tr.Shape()
            if id == 0:
                self.upper_face = face
            elif id == 1:
                self.lower_face = face
            elif id == 2:
                self.tip_face = face
            elif id == 3:
                self.root_face = face

    def scale(self, factor):
        """
        Scale the blade coordinates by a specified factor.

        :param float factor: scaling factor
        """
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        self.blade_coordinates_up *= factor
        self.blade_coordinates_down *= factor

        for id, face in enumerate([self.upper_face, self.lower_face,
                     self.tip_face, self.root_face]):
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
            from OCC.Core.gp import gp_Pnt
            from OCC.Core.gp import gp_Dir
            from OCC.Core.gp import gp_Ax1
            from OCC.Core.gp import gp_Trsf
            
            origin = gp_Pnt(0, 0, 0)
            trsf = gp_Trsf()
            trsf.SetScale(origin, factor)

            brep_tr = BRepBuilderAPI_Transform(face, trsf, True, True)
            face = brep_tr.Shape()
            if id == 0:
                self.upper_face = face
            elif id == 1:
                self.lower_face = face
            elif id == 2:
                self.tip_face = face
            elif id == 3:
                self.root_face = face

    def plot(self, elev=None, azim=None, ax=None, outfile=None):
        """
        Plot the generated blade sections.

        :param int elev: set the view elevation of the axes. This can be used
            to rotate the axes programatically. 'elev' stores the elevation
            angle in the z plane. If elev is None, then the initial value is
            used which was specified in the mplot3d.Axes3D constructor. Default
            value is None
        :param int azim: set the view azimuth angle of the axes. This can be
            used to rotate the axes programatically. 'azim' stores the azimuth
            angle in the x,y plane. If azim is None, then the initial value is
            used which was specified in the mplot3d.Axes3D constructor. Default
            value is None
        :param matplotlib.axes ax: allows to pass the instance of figure axes
            to the current plot. This is useful when the user needs to plot the
            coordinates of several blade objects on the same figure (see the
            example below). If nothing is passed then the method plots on a new
            figure axes. Default value is None
        :param string outfile: save the plot if a filename string is provided.
            Default value is None

        EXAMPLE:
        Assume we already have the arrays radii, chord, pitch, rake, skew for
        10 blade sections.

        >>> sections_1 = np.asarray([blade.NacaProfile(digits='0012')
                            for i in range(10)])
        >>> blade_1 = blade.Blade(sections=sections,
                                  radii=radii,
                                  chord_lengths=chord,
                                  pitch=pitch,
                                  rake=rake,
                                  skew_angles=skew)
        >>> blade_1.apply_transformations()

        >>> sections_2 = np.asarray([blade.NacaProfile(digits='0012')
                            for i in range(10)])
        >>> blade_2 = blade.Blade(sections=sections,
                                  radii=radii,
                                  chord_lengths=chord,
                                  pitch=pitch,
                                  rake=rake,
                                  skew_angles=skew)
        >>> blade_2.apply_transformations()
        >>> blade_2.rotate(rot_angle_deg=72)

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(projection='3d')
        >>> blade_1.plot(ax=ax)
        >>> blade_2.plot(ax=ax)

        On the other hand, if we need to plot for a single blade object,
        we can just ignore such parameter, and the method will internally
        create a new instance for the figure axes, i.e.

        >>> sections = np.asarray([blade.NacaProfile(digits='0012')
                            for i in range(10)])
        >>> blade = blade.Blade(sections=sections,
                                radii=radii,
                                chord_lengths=chord,
                                pitch=pitch,
                                rake=rake,
                                skew_angles=skew)
        >>> blade.apply_transformations()
        >>> blade.plot()
        """
        if len(self.blade_coordinates_up) == 0:
            raise ValueError('You must build the blade before plotting.')
        if ax:
            ax = ax
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        ax.set_aspect('auto')

        for i in range(self.n_sections):
            pts_up = self.blade_coordinates_up[i]
            pts_down = self.blade_coordinates_down[i]
            
            ax.plot(*pts_up),
            ax.plot(*pts_down)

        plt.axis('auto')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('radii axis')
        ax.xaxis.label.set_color('red')
        ax.yaxis.label.set_color('red')
        ax.zaxis.label.set_color('red')
        ax.view_init(elev=elev, azim=azim)

        if outfile:
            plt.savefig(outfile)

    @staticmethod
    def _import_occ_libs():
        """
        Private static method to import specific modules from the OCC package.
        """
        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
        from OCC.Core.gp import gp_Pnt
        from OCC.Core.TColgp import TColgp_HArray1OfPnt
        from OCC.Core.GeomAPI import GeomAPI_Interpolate
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex,\
             BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire,\
             BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid

        # Set the imported modules as global variables to be used out of scope
        global BRepOffsetAPI_ThruSections, gp_Pnt, TColgp_HArray1OfPnt,\
               GeomAPI_Interpolate, BRepBuilderAPI_MakeVertex,\
               BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire,\
               BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing

    def _write_blade_errors(self, upper_face, lower_face, errors):
        """
        Private method to write the errors between the generated foil points in
        3D space from the parametric transformations, and their projections on
        the generated blade faces from the OCC algorithm.

        :param string upper_face: if string is passed then the method generates
            the blade upper surface using the BRepOffsetAPI_ThruSections
            algorithm, then exports the generated CAD into .iges file holding
            the name <upper_face_string>.iges
        :param string lower_face: if string is passed then the method generates
            the blade lower surface using the BRepOffsetAPI_ThruSections
            algorithm, then exports the generated CAD into .iges file holding
            the name <lower_face_string>.iges
        :param string errors: if string is passed then the method writes out
            the distances between each discrete point used to construct the
            blade and the nearest point on the CAD that is perpendicular to
            that point
        """
        from OCC.Core.gp import gp_Pnt
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
        from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

        output_string = '\n'
        with open(errors + '.txt', 'w') as f:
            if upper_face:
                output_string += '########## UPPER FACE ##########\n\n'
                output_string += 'N_section\t\tN_point\t\t\tX_crds\t\t\t\t'
                output_string += 'Y_crds\t\t\t\t\tZ_crds\t\t\t\t\tDISTANCE'
                output_string += '\n\n'
                for i in range(self.n_sections):
                    alength = len(self.blade_coordinates_up[i][0])
                    for j in range(alength):
                        vertex = BRepBuilderAPI_MakeVertex(
                            gp_Pnt(
                                1000 * self.blade_coordinates_up[i][0][j],
                                1000 * self.blade_coordinates_up[i][1][j], 1000
                                * self.blade_coordinates_up[i][2][j])).Vertex()
                        projection = BRepExtrema_DistShapeShape(
                            self.generated_upper_face, vertex)
                        projection.Perform()
                        output_string += str(
                            i) + '\t\t\t' + str(j) + '\t\t\t' + str(
                                1000 *
                                self.blade_coordinates_up[i][0][j]) + '\t\t\t'
                        output_string += str(
                            1000 * self.blade_coordinates_up[i][1]
                            [j]) + '\t\t\t' + str(
                                1000 * self.blade_coordinates_up[i][2]
                                [j]) + '\t\t\t' + str(projection.Value())
                        output_string += '\n'

            if lower_face:
                output_string += '########## LOWER FACE ##########\n\n'
                output_string += 'N_section\t\tN_point\t\t\tX_crds\t\t\t\t'
                output_string += 'Y_crds\t\t\t\t\tZ_crds\t\t\t\t\tDISTANCE'
                output_string += '\n\n'
                for i in range(self.n_sections):
                    alength = len(self.blade_coordinates_down[i][0])
                    for j in range(alength):
                        vertex = BRepBuilderAPI_MakeVertex(
                            gp_Pnt(
                                1000 * self.blade_coordinates_down[i][0][j],
                                1000 * self.blade_coordinates_down[i][1][j],
                                1000 *
                                self.blade_coordinates_down[i][2][j])).Vertex()
                        projection = BRepExtrema_DistShapeShape(
                            self.generated_lower_face, vertex)
                        projection.Perform()
                        output_string += str(
                            i) + '\t\t\t' + str(j) + '\t\t\t' + str(
                                1000 *
                                self.blade_coordinates_down[i][0][j]) + '\t\t\t'
                        output_string += str(
                            1000 * self.blade_coordinates_down[i][1]
                            [j]) + '\t\t\t' + str(
                                1000 * self.blade_coordinates_down[i][2]
                                [j]) + '\t\t\t' + str(projection.Value())
                        output_string += '\n'
            f.write(output_string)

    def generate_solid(self):
        """
        Generate a solid blade assembling the upper face, lower face, tip and
        root using the BRepBuilderAPI_MakeSolid algorithm.

        :param int max_deg: Define the maximal U degree of generated surface.
            Default value is 1
        :raises RuntimeError: if the assembling of the solid blade is not
            completed successfully
        """
        from OCC.Display.SimpleGui import init_display
        from OCC.Core.TopoDS import TopoDS_Shell
        import OCC.Core.TopoDS
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing, \
                BRepBuilderAPI_MakeSolid

        faces = [
            self.upper_face, self.lower_face, self.tip_face, self.root_face
        ]

        sewer = BRepBuilderAPI_Sewing(1e-2)
        for face in faces:
            sewer.Add(face)
        sewer.Perform()

        result_shell = sewer.SewedShape()
        solid_maker = BRepBuilderAPI_MakeSolid()
        solid_maker.Add(OCC.Core.TopoDS.topods.Shell(result_shell))

        if not solid_maker.IsDone():
            raise RuntimeError('Unsuccessful assembling of solid blade')
        result_solid = solid_maker.Solid()

       	return result_solid

    def export_stl(self, filename, linear_deflection=0.1):
        """
        Generate and export the .STL file for the entire blade.
        This method requires PythonOCC (7.4.0) to be installed.
        """
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Extend.DataExchange import write_stl_file
        from OCC.Core.StlAPI import StlAPI_Writer

        sewer = BRepBuilderAPI_Sewing(1e-2)
        sewer.Add(self.upper_face)
        sewer.Add(self.lower_face)
        sewer.Add(self.root_face)
        sewer.Add(self.tip_face)
        sewer.Perform()
        sewed_shape = sewer.SewedShape()

        triangulation = BRepMesh_IncrementalMesh(sewed_shape, linear_deflection, True)
        triangulation.Perform()

        writer = StlAPI_Writer()
        writer.SetASCIIMode(False)
        writer.Write(sewed_shape, filename)

    def _generate_leading_edge_curves(self):
        """
        Private method to generate curves that follow the leading edge of the blade
        (top and bottom surfaces).
        """
        self._import_occ_libs()
        
        # Extract points at leftmost 5% for upper and lower surfaces
        upper_points = []
        lower_points = []
        
        for i in range(self.n_sections):
            min_x = np.min(self.sections[i].xdown_coordinates)
            max_x = np.max(self.sections[i].xdown_coordinates)
            delta_x = max_x - min_x
            
            target_x = min_x + 0.95 * delta_x
             
            idx = np.abs(self.sections[i].xdown_coordinates - target_x).argmin()
            
            # Create points for upper and lower curves
            upper_points.append(gp_Pnt(
                1000 * self.blade_coordinates_up[i][0][idx],
                1000 * self.blade_coordinates_up[i][1][idx],
                1000 * self.blade_coordinates_up[i][2][idx]
            ))
            
            lower_points.append(gp_Pnt(
                1000 * self.blade_coordinates_down[i][0][idx],
                1000 * self.blade_coordinates_down[i][1][idx],
                1000 * self.blade_coordinates_down[i][2][idx]
            ))

        # Create arrays of points for interpolation
        upper_array = TColgp_HArray1OfPnt(1, len(upper_points))
        lower_array = TColgp_HArray1OfPnt(1, len(lower_points))
        
        for i, (up, low) in enumerate(zip(upper_points, lower_points)):
            upper_array.SetValue(i + 1, up)
            lower_array.SetValue(i + 1, low)

        # Create interpolated curves
        upper_curve = GeomAPI_Interpolate(upper_array, False, 1e-9)
        lower_curve = GeomAPI_Interpolate(lower_array, False, 1e-9)
        
        upper_curve.Perform()
        lower_curve.Perform()
        
        # Convert to edges
        self.upper_le_edge = BRepBuilderAPI_MakeEdge(upper_curve.Curve()).Edge()
        self.lower_le_edge = BRepBuilderAPI_MakeEdge(lower_curve.Curve()).Edge()

    def export_iges(self, filename, include_le_curves=False):
        """
        Generate and export the .IGES file for the entire blade.
        This method requires PythonOCC (7.4.0) to be installed.
        """
        from OCC.Core.IGESControl import IGESControl_Writer

        if include_le_curves:
            self._generate_leading_edge_curves()
        
        iges_writer = IGESControl_Writer()
        iges_writer.AddShape(self.upper_face)
        iges_writer.AddShape(self.lower_face)
        iges_writer.AddShape(self.root_face)
        iges_writer.AddShape(self.tip_face)
        
        if include_le_curves:
            iges_writer.AddShape(self.upper_le_edge)
            iges_writer.AddShape(self.lower_le_edge)
        
        iges_writer.Write(filename)

    @staticmethod
    def _check_string(filename):
        """
        Private method to check if the parameter type is string

        :param string filename: filename of the generated .iges surface
        """
        if not isinstance(filename, str):
            raise TypeError('IGES filename must be a valid string.')

    @staticmethod
    def _check_errors(upper_face, lower_face):
        """
        Private method to check if either the blade upper face or lower face
        is passed in the generate_iges method. Otherwise it raises an exception

        :param string upper_face: blade upper face.
        :param string lower_face: blade lower face.
        """
        if not (upper_face or lower_face):
            raise ValueError(
                'Either upper_face or lower_face must not be None.')

    def _abs_to_norm(self, D_prop):
        """
        Private method to normalize the blade parameters.

        :param float D_prop: propeller diameter
        """
        self.radii = self.radii * 2. / D_prop
        self.chord_lengths = self.chord_lengths / D_prop
        self.pitch = self.pitch / D_prop
        self.rake = self.rake / D_prop

    def _norm_to_abs(self, D_prop):
        """
        Private method that converts the normalized blade parameters into the
        actual values.

        :param float D_prop: propeller diameter
        """
        self.radii = self.radii * D_prop / 2.
        self.chord_lengths = self.chord_lengths * D_prop
        self.pitch = self.pitch * D_prop
        self.rake = self.rake * D_prop

    def export_ppg(self,
                   filename='data_out.ppg',
                   D_prop=0.25,
                   D_hub=0.075,
                   n_blades=5,
                   params_normalized=False):
        """
        Export the generated blade parameters and sectional profiles into
        .ppg format.

        :param string filename: name of the exported file. Default is
            'data/data_out.ppg'
        :param float D_prop: propeller diameter
        :param float D_hub: hub diameter
        :param float n_blades: number of blades
        :param bool params_normalized: since the standard .ppg format contains
            the blade parameters in the normalized form, therefore the user
            needs to inform whether the provided parameters (from the class
            Blade) are normalized or not. By default the argument is set to
            False, which assumes the user provides the blade parameters in
            their actual values, i.e. not normalized, hence a normalization
            operation needs to be applied so as to follow the .ppg standard
            format.
        """
        thickness = np.zeros(self.n_sections)
        camber = np.zeros(self.n_sections)
        for i, section in enumerate(self.sections):
            # Evaluate maximum profile thickness and camber for each section.
            # We assume at the current step, that sectional profiles already
            # have the coordinates (x_up,x_down) normalized by chord length (C)
            # and subsequently (y_up,y_down) are also scaled. This implies that
            # the computed thickness and camber are given in their normalized
            # form, i.e. thickness=t/C and camber=f/C.
            thickness[i] = section.max_thickness()
            camber[i] = section.max_camber()

        if params_normalized is False:
            # Put the parameters (radii, chord, pitch, rake) in the normalized
            # form.
            self._abs_to_norm(D_prop=D_prop)

        output_string = ""
        output_string += 'propeller id       = SVA\n'
        output_string += 'propeller diameter = ' + str(D_prop) + '\n'
        output_string += 'hub diameter       = ' + str(D_hub) + '\n'
        output_string += 'number of blades   = ' + str(n_blades) + '\n'
        output_string += "'Elica PPTC workshop'\n"
        output_string += 'number of radial sections         = ' + str(
            self.n_sections) + '\n'
        output_string += 'number of radial sections         = ' + str(
            self.n_sections) + '\n'
        output_string += 'number of sectional profiles      = ' + str(
            self.n_sections) + '\n'
        output_string += 'description of sectional profiles = BNF\n'
        output_string += '            r/R            c/D      skew[deg]'\
                         '         rake/D            P/D            t/C'\
                         '            f/C\n'
        for i in range(self.n_sections):
            output_string += ' ' + str("%.8e" % self.radii[i]) + ' ' + str(
                "%.8e" % self.chord_lengths[i]) + ' ' + str(
                    "%.8e" % self.skew_angles[i]) + ' ' + str(
                        "%.8e" % self.rake[i])
            output_string += ' ' + str("%.8e" % self.pitch[i]) + ' ' + str(
                "%.8e" % thickness[i]) + ' ' + str("%.8e" % camber[i]) + '\n'

        for i in range(self.n_sections):
            output_string += str("%.8e" % self.radii[i]) + '  ' + str(
                len(self.sections[i].xup_coordinates)) + '\n'

            for value in self.sections[i].xup_coordinates:
                output_string += ' ' + str("%.8e" % value)
            output_string += ' \n'
            for value in self.sections[i].yup_coordinates:
                output_string += ' ' + str("%.8e" % value)
            output_string += ' \n'
            for value in self.sections[i].ydown_coordinates:
                output_string += ' ' + str("%.8e" % value)
            output_string += ' \n'

        hub_offsets = np.asarray(
            [[-3.0, 0.305], [-0.57, 0.305], [-0.49, 0.305], [-0.41, 0.305],
             [-0.33, 0.305], [-0.25, 0.305], [-0.17, 0.305], [0.23, 0.305],
             [0.31, 0.285], [0.39, 0.2656], [0.47, 0.2432], [0.55, 0.2124],
             [0.63, 0.1684], [0.71, 0.108], [0.79, 0.0]])

        output_string += 'number of Hub offsets = ' + str(
            len(hub_offsets)) + '\n'

        for i, offset in enumerate(hub_offsets):
            if i == len(hub_offsets) - 1:
                output_string += str("%.8e" % offset[0]) + ' ' + str(
                    "%.8e" % hub_offsets[i][1])
                
                continue
            output_string += str("%.8e" % offset[0]) + ' ' + str(
                "%.8e" % offset[1]) + '\n'

        with open(filename, 'w') as f:
            f.write(output_string)

        if params_normalized is False:
            # Revert back normalized parameters into actual values.
            self._norm_to_abs(D_prop=D_prop)

    def __str__(self):
        """
        This method prints all the parameters on the screen. Its purpose is
        for debugging.
        """
        string = ''
        string += 'Blade number of sections = {}'.format(self.n_sections)
        string += '\nBlade radii sections = {}'.format(self.radii)
        string += '\nChord lengths of the sectional profiles'\
                  ' = {}'.format(self.chord_lengths)
        string += '\nRadial distribution of the pitch (in unit lengths)'\
                  ' = {}'.format(self.pitch)
        string += '\nRadial distribution of the rake (in unit length)'\
                  ' = {}'.format(self.rake)
        string += '\nRadial distribution of the skew angles'\
                  ' (in degrees) = {}'.format(self.skew_angles)
        string += '\nPitch angles (in radians) for the'\
                  ' sections = {}'.format(self.pitch_angles)
        string += '\nInduced rake from skew (in unit length)'\
                  ' for the sections = {}'.format(self.induced_rake)
        return string

    def display(self):
        """
        Display the propeller with shaft.
        """
        from OCC.Display.SimpleGui import init_display
        display, start_display = init_display()[:2]
        display.DisplayShape(self.upper_face, update=True)
        display.DisplayShape(self.lower_face, update=True)
        display.DisplayShape(self.root_face, update=True)
        display.DisplayShape(self.tip_face, update=True)
        start_display()
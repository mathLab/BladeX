"""
Module for the blade bottom-up parametrized construction.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

        self.pitch_angles = self._compute_pitch_angle()
        self.induced_rake = self._induced_rake_from_skew()

        self.blade_coordinates_up = []
        self.blade_coordinates_down = []

        self.generated_upper_face = None
        self.generated_lower_face = None
        self.generated_tip = None
        self.generated_root = None

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

    def _compute_pitch_angle(self):
        """
        Private method that computes the pitch angle from the linear pitch for
        all blade sections.

        :return: pitch angle in radians
        :rtype: numpy.ndarray
        """
        return np.arctan(self.pitch / (2.0 * np.pi * self.radii))

    def _induced_rake_from_skew(self):
        """
        Private method that computes the induced rake from skew for all the
        blade sections, according to :ref:`mytransformation_operations`.

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

    def rotate(self, deg_angle=None, rad_angle=None):
        """
        3D counter clockwise rotation about the X-axis of the Cartesian
        coordinate system, which is the axis of rotation of the propeller hub.

        The rotation matrix, :math:`R(\\theta)`, is used to perform rotation
        in the 3D Euclidean space about the X-axis, which is -- by default --
        the propeller axis of rotation.

        :math:`R(\\theta)` is defined by:

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
        :raises ValueError: if both rad_angle and deg_angle are inserted,
            or if neither is inserted

        """
        if not self.blade_coordinates_up:
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
        else:
            raise ValueError(
                'You have to pass either the angle in radians or in degrees.')

        # Rotation is always about the X-axis, which is the center if the hub
        # according to the implemented transformation procedure
        rot_matrix = np.array([1, 0, 0, 0, cosine, -sine, 0, sine,
                               cosine]).reshape((3, 3))

        for i in range(self.n_sections):
            coord_matrix_up = np.vstack((self.blade_coordinates_up[i][0],
                                         self.blade_coordinates_up[i][1],
                                         self.blade_coordinates_up[i][2]))
            coord_matrix_down = np.vstack((self.blade_coordinates_down[i][0],
                                           self.blade_coordinates_down[i][1],
                                           self.blade_coordinates_down[i][2]))

            new_coord_matrix_up = np.dot(rot_matrix, coord_matrix_up)
            new_coord_matrix_down = np.dot(rot_matrix, coord_matrix_down)

            self.blade_coordinates_up[i][0] = new_coord_matrix_up[0]
            self.blade_coordinates_up[i][1] = new_coord_matrix_up[1]
            self.blade_coordinates_up[i][2] = new_coord_matrix_up[2]

            self.blade_coordinates_down[i][0] = new_coord_matrix_down[0]
            self.blade_coordinates_down[i][1] = new_coord_matrix_down[1]
            self.blade_coordinates_down[i][2] = new_coord_matrix_down[2]

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
        >>> ax = fig.gca(projection=Axes3D.name)
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
        if not self.blade_coordinates_up:
            raise ValueError('You must apply transformations before plotting.')
        if ax:
            ax = ax
        else:
            fig = plt.figure()
            ax = fig.gca(projection=Axes3D.name)
        ax.set_aspect('auto')

        for i in range(self.n_sections):
            ax.plot(self.blade_coordinates_up[i][0],
                    self.blade_coordinates_up[i][1],
                    self.blade_coordinates_up[i][2])
            ax.plot(self.blade_coordinates_down[i][0],
                    self.blade_coordinates_down[i][1],
                    self.blade_coordinates_down[i][2])

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

    def _generate_upper_face(self, max_deg):
        """
        Private method to generate the blade upper face.

        :param int max_deg: Define the maximal U degree of generated surface
        """
        self._import_occ_libs()
        # Initializes ThruSections algorithm for building a shell passing
        # through a set of sections (wires). The generated faces between
        # the edges of every two consecutive wires are smoothed out with
        # a precision criterion = 1e-10
        generator = BRepOffsetAPI_ThruSections(False, False, 1e-10)
        generator.SetMaxDegree(max_deg)
        # Define upper edges (wires) for the face generation
        for i in range(self.n_sections):
            npoints = len(self.blade_coordinates_up[i][0])
            vertices = TColgp_HArray1OfPnt(1, npoints)
            for j in range(npoints):
                vertices.SetValue(
                    j + 1,
                    gp_Pnt(1000 * self.blade_coordinates_up[i][0][j],
                           1000 * self.blade_coordinates_up[i][1][j],
                           1000 * self.blade_coordinates_up[i][2][j]))
            # Initializes an algorithm for constructing a constrained
            # BSpline curve passing through the points of the blade i-th
            # section, with tolerance = 1e-9
            bspline = GeomAPI_Interpolate(vertices, False, 1e-9)
            bspline.Perform()
            edge = BRepBuilderAPI_MakeEdge(bspline.Curve()).Edge()
            if i == 0:
                bound_root_edge = edge
            # Add BSpline wire to the generator constructor
            generator.AddWire(BRepBuilderAPI_MakeWire(edge).Wire())
        # Returns the shape built by the shape construction algorithm
        generator.Build()
        # Returns the Face generated by each edge of the first section
        self.generated_upper_face = generator.GeneratedFace(bound_root_edge)

    def _generate_lower_face(self, max_deg):
        """
        Private method to generate the blade lower face.

        :param int max_deg: Define the maximal U degree of generated surface
        """
        self._import_occ_libs()
        # Initializes ThruSections algorithm for building a shell passing
        # through a set of sections (wires). The generated faces between
        # the edges of every two consecutive wires are smoothed out with
        # a precision criterion = 1e-10
        generator = BRepOffsetAPI_ThruSections(False, False, 1e-10)
        generator.SetMaxDegree(max_deg)
        # Define upper edges (wires) for the face generation
        for i in range(self.n_sections):
            npoints = len(self.blade_coordinates_down[i][0])
            vertices = TColgp_HArray1OfPnt(1, npoints)
            for j in range(npoints):
                vertices.SetValue(
                    j + 1,
                    gp_Pnt(1000 * self.blade_coordinates_down[i][0][j],
                           1000 * self.blade_coordinates_down[i][1][j],
                           1000 * self.blade_coordinates_down[i][2][j]))
            # Initializes an algorithm for constructing a constrained
            # BSpline curve passing through the points of the blade i-th
            # section, with tolerance = 1e-9
            bspline = GeomAPI_Interpolate(vertices, False, 1e-9)
            bspline.Perform()
            edge = BRepBuilderAPI_MakeEdge(bspline.Curve()).Edge()
            if i == 0:
                bound_root_edge = edge
            # Add BSpline wire to the generator constructor
            generator.AddWire(BRepBuilderAPI_MakeWire(edge).Wire())
        # Returns the shape built by the shape construction algorithm
        generator.Build()
        # Returns the Face generated by each edge of the first section
        self.generated_lower_face = generator.GeneratedFace(bound_root_edge)

    def _generate_tip(self, max_deg):
        """
        Private method to generate the surface that closing the blade tip.

        :param int max_deg: Define the maximal U degree of generated surface
        """
        self._import_occ_libs()

        generator = BRepOffsetAPI_ThruSections(False, False, 1e-10)
        generator.SetMaxDegree(max_deg)
        # npoints_up == npoints_down
        npoints = len(self.blade_coordinates_down[-1][0])
        vertices_1 = TColgp_HArray1OfPnt(1, npoints)
        vertices_2 = TColgp_HArray1OfPnt(1, npoints)
        for j in range(npoints):
            vertices_1.SetValue(
                j + 1,
                gp_Pnt(1000 * self.blade_coordinates_down[-1][0][j],
                       1000 * self.blade_coordinates_down[-1][1][j],
                       1000 * self.blade_coordinates_down[-1][2][j]))

            vertices_2.SetValue(
                j + 1,
                gp_Pnt(1000 * self.blade_coordinates_up[-1][0][j],
                       1000 * self.blade_coordinates_up[-1][1][j],
                       1000 * self.blade_coordinates_up[-1][2][j]))

        # Initializes an algorithm for constructing a constrained
        # BSpline curve passing through the points of the blade last
        # section, with tolerance = 1e-9
        bspline_1 = GeomAPI_Interpolate(vertices_1, False, 1e-9)
        bspline_1.Perform()

        bspline_2 = GeomAPI_Interpolate(vertices_2, False, 1e-9)
        bspline_2.Perform()

        edge_1 = BRepBuilderAPI_MakeEdge(bspline_1.Curve()).Edge()
        edge_2 = BRepBuilderAPI_MakeEdge(bspline_2.Curve()).Edge()

        # Add BSpline wire to the generator constructor
        generator.AddWire(BRepBuilderAPI_MakeWire(edge_1).Wire())
        generator.AddWire(BRepBuilderAPI_MakeWire(edge_2).Wire())
        # Returns the shape built by the shape construction algorithm
        generator.Build()
        # Returns the Face generated by each edge of the first section
        self.generated_tip = generator.GeneratedFace(edge_1)

    def _generate_root(self, max_deg):
        """
        Private method to generate the surface that closing the blade at the root.

        :param int max_deg: Define the maximal U degree of generated surface
        """
        self._import_occ_libs()

        generator = BRepOffsetAPI_ThruSections(False, False, 1e-10)
        generator.SetMaxDegree(max_deg)
        # npoints_up == npoints_down
        npoints = len(self.blade_coordinates_down[0][0])
        vertices_1 = TColgp_HArray1OfPnt(1, npoints)
        vertices_2 = TColgp_HArray1OfPnt(1, npoints)
        for j in range(npoints):
            vertices_1.SetValue(
                j + 1,
                gp_Pnt(1000 * self.blade_coordinates_down[0][0][j],
                       1000 * self.blade_coordinates_down[0][1][j],
                       1000 * self.blade_coordinates_down[0][2][j]))

            vertices_2.SetValue(
                j + 1,
                gp_Pnt(1000 * self.blade_coordinates_up[0][0][j],
                       1000 * self.blade_coordinates_up[0][1][j],
                       1000 * self.blade_coordinates_up[0][2][j]))

        # Initializes an algorithm for constructing a constrained
        # BSpline curve passing through the points of the blade last
        # section, with tolerance = 1e-9
        bspline_1 = GeomAPI_Interpolate(vertices_1, False, 1e-9)
        bspline_1.Perform()

        bspline_2 = GeomAPI_Interpolate(vertices_2, False, 1e-9)
        bspline_2.Perform()

        edge_1 = BRepBuilderAPI_MakeEdge(bspline_1.Curve()).Edge()
        edge_2 = BRepBuilderAPI_MakeEdge(bspline_2.Curve()).Edge()

        # Add BSpline wire to the generator constructor
        generator.AddWire(BRepBuilderAPI_MakeWire(edge_1).Wire())
        generator.AddWire(BRepBuilderAPI_MakeWire(edge_2).Wire())
        # Returns the shape built by the shape construction algorithm
        generator.Build()
        # Returns the Face generated by each edge of the first section
        self.generated_root = generator.GeneratedFace(edge_1)

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

    def generate_iges(self,
                      upper_face=None,
                      lower_face=None,
                      tip=None,
                      root=None,
                      max_deg=1,
                      display=False,
                      errors=None):
        """
        Generate and export the .iges CAD for the blade upper face, lower face,
        tip and root. This method requires PythonOCC (7.4.0) to be installed.

        :param string upper_face: if string is passed then the method generates
            the blade upper surface using the BRepOffsetAPI_ThruSections
            algorithm, then exports the generated CAD into .iges file holding
            the name <upper_face_string>.iges. Default value is None
        :param string lower_face: if string is passed then the method generates
            the blade lower surface using the BRepOffsetAPI_ThruSections
            algorithm, then exports the generated CAD into .iges file holding
            the name <lower_face_string>.iges. Default value is None
        :param string tip: if string is passed then the method generates
            the blade tip using the BRepOffsetAPI_ThruSections algorithm
            in order to close the blade at the tip, then exports the generated
            CAD into .iges file holding the name <tip_string>.iges.
            Default value is None
        :param string root: if string is passed then the method generates
            the blade root using the BRepOffsetAPI_ThruSections algorithm
            in order to close the blade at the root, then exports the generated
            CAD into .iges file holding the name <tip_string>.iges. 
            Default value is None
        :param int max_deg: Define the maximal U degree of generated surface.
            Default value is 1
        :param bool display: if True, then display the generated CAD. Default
            value is False
        :param string errors: if string is passed then the method writes out
            the distances between each discrete point used to construct the
            blade and the nearest point on the CAD that is perpendicular to
            that point. Default value is None

        We note that the blade object must have its radial sections be arranged
        in order from the blade root to the blade tip, so that generate_iges
        method can build the CAD surface that passes through the corresponding
        airfoils. Also to be able to identify and close the blade tip and root.
        """

        from OCC.Core.IGESControl import IGESControl_Writer
        from OCC.Display.SimpleGui import init_display

        if max_deg <= 0:
            raise ValueError('max_deg argument must be a positive integer.')

        if upper_face:
            self._check_string(filename=upper_face)
            self._generate_upper_face(max_deg=max_deg)
            # Write IGES
            iges_writer = IGESControl_Writer()
            iges_writer.AddShape(self.generated_upper_face)
            iges_writer.Write(upper_face + '.iges')

        if lower_face:
            self._check_string(filename=lower_face)
            self._generate_lower_face(max_deg=max_deg)
            # Write IGES
            iges_writer = IGESControl_Writer()
            iges_writer.AddShape(self.generated_lower_face)
            iges_writer.Write(lower_face + '.iges')

        if tip:
            self._check_string(filename=tip)
            self._generate_tip(max_deg=max_deg)
            # Write IGES
            iges_writer = IGESControl_Writer()
            iges_writer.AddShape(self.generated_tip)
            iges_writer.Write(tip + '.iges')

        if root:
            self._check_string(filename=root)
            self._generate_root(max_deg=max_deg)
            # Write IGES
            iges_writer = IGESControl_Writer()
            iges_writer.AddShape(self.generated_root)
            iges_writer.Write(root + '.iges')

        if errors:
            # Write out errors between discrete points and constructed faces
            self._check_string(filename=errors)
            self._check_errors(upper_face=upper_face, lower_face=lower_face)

            self._write_blade_errors(
                upper_face=upper_face, lower_face=lower_face, errors=errors)

        if display:
            display, start_display, add_menu, add_function_to_menu = init_display(
            )

            ## DISPLAY FACES
            if upper_face:
                display.DisplayShape(self.generated_upper_face, update=True)
            if lower_face:
                display.DisplayShape(self.generated_lower_face, update=True)
            if tip:
                display.DisplayShape(self.generated_tip, update=True)
            if root:
                display.DisplayShape(self.generated_root, update=True)
            start_display()

    def generate_solid(self,
                             max_deg=1,
                             display=False,
                             errors=None):
        """
        Generate a solid blade assembling the upper face, lower face, tip and
        root using the BRepBuilderAPI_MakeSolid algorithm.
        This method requires PythonOCC (7.4.0) to be installed.

        :param int max_deg: Define the maximal U degree of generated surface.
            Default value is 1
        :param bool display: if True, then display the generated CAD. Default
            value is False
        :param string errors: if string is passed then the method writes out
            the distances between each discrete point used to construct the
            blade and the nearest point on the CAD that is perpendicular to
            that point. Default value is None
        :raises RuntimeError: if the assembling of the solid blade is not
            completed successfully
        """
        from OCC.Display.SimpleGui import init_display
        from OCC.Core.TopoDS import TopoDS_Shell
        import OCC.Core.TopoDS

        if max_deg <= 0:
            raise ValueError('max_deg argument must be a positive integer.')

        self._generate_upper_face(max_deg=max_deg)
        self._generate_lower_face(max_deg=max_deg)
        self._generate_tip(max_deg=max_deg)
        self._generate_root(max_deg=max_deg)

        if errors:
            # Write out errors between discrete points and constructed faces
            self._check_string(filename=errors)
            self._check_errors(upper_face=upper_face, lower_face=lower_face)

            self._write_blade_errors(
                upper_face=upper_face, lower_face=lower_face, errors=errors)

        if display:
            display, start_display, add_menu, add_function_to_menu = init_display(
            )

            ## DISPLAY FACES
            display.DisplayShape(self.generated_upper_face, update=True)
            display.DisplayShape(self.generated_lower_face, update=True)
            display.DisplayShape(self.generated_tip, update=True)
            display.DisplayShape(self.generated_root, update=True)
            start_display()

        sewer = BRepBuilderAPI_Sewing(1e-2)
        sewer.Add(self.generated_upper_face)
        sewer.Add(self.generated_lower_face)
        sewer.Add(self.generated_tip)
        sewer.Add(self.generated_root)
        sewer.Perform()
        result_shell = sewer.SewedShape()
        solid_maker = BRepBuilderAPI_MakeSolid()
        solid_maker.Add(OCC.Core.TopoDS.topods_Shell(result_shell))
        if not solid_maker.IsDone():
            raise RuntimeError('Unsuccessful assembling of solid blade')
        result_solid = solid_maker.Solid()
       	return result_solid

    def generate_stl(self, upper_face=None,
                      lower_face=None,
                      tip=None,
                      root=None,
                      max_deg=1,
                      display=False,
                      errors=None):
        """
        Generate and export the .STL files for upper face, lower face, tip
        and root. This method requires PythonOCC (7.4.0) to be installed.

        :param string upper_face: if string is passed then the method generates
            the blade upper surface using the BRepOffsetAPI_ThruSections
            algorithm, then exports the generated CAD into .stl file holding
            the name <upper_face_string>.stl. Default value is None
        :param string lower_face: if string is passed then the method generates
            the blade lower surface using the BRepOffsetAPI_ThruSections
            algorithm, then exports the generated CAD into .stl file holding
            the name <lower_face_string>.stl. Default value is None
        :param string tip: if string is passed then the method generates
            the blade tip using the BRepOffsetAPI_ThruSections algorithm
            in order to close the blade at the tip, then exports the generated
            CAD into .stl file holding the name <tip_string>.stl. 
            Default value is None
        :param string root: if string is passed then the method generates
            the blade root using the BRepOffsetAPI_ThruSections algorithm
            in order to close the blade at the root, then exports the generated 
            CAD into .stl file holding the name <tip_string>.stl. 
            Default value is None
        :param int max_deg: Define the maximal U degree of generated surface.
            Default value is 1
        :param bool display: if True, then display the generated CAD. Default
            value is False
        :param string errors: if string is passed then the method writes out
            the distances between each discrete point used to construct the
            blade and the nearest point on the CAD that is perpendicular to
            that point. Default value is None

        We note that the blade object must have its radial sections be arranged
        in order from the blade root to the blade tip, so that generate_stl
        method can build the CAD surface that passes through the corresponding
        airfoils. Also to be able to identify and close the blade tip and root.
        """

        from OCC.Extend.DataExchange import write_stl_file
        from OCC.Display.SimpleGui import init_display

        if max_deg <= 0:
            raise ValueError('max_deg argument must be a positive integer.')

        if upper_face:
            self._check_string(filename=upper_face)
            self._generate_upper_face(max_deg=max_deg)
            # Write STL
            write_stl_file(self.generated_upper_face, upper_face + '.stl')

        if lower_face:
            self._check_string(filename=lower_face)
            self._generate_lower_face(max_deg=max_deg)
            # Write STL
            write_stl_file(self.generated_lower_face, lower_face + '.stl')

        if tip:
            self._check_string(filename=tip)
            self._generate_tip(max_deg=max_deg)
            # Write STL
            write_stl_file(self.generated_tip, tip + '.stl')

        if root:
            self._check_string(filename=root)
            self._generate_root(max_deg=max_deg)
            # Write STL
            write_stl_file(self.generated_root, root + '.stl')

        if errors:
            # Write out errors between discrete points and constructed faces
            self._check_string(filename=errors)
            self._check_errors(upper_face=upper_face, lower_face=lower_face)

            self._write_blade_errors(
                upper_face=upper_face, lower_face=lower_face, errors=errors)

        if display:
            display, start_display, add_menu, add_function_to_menu = init_display(
            )

            ## DISPLAY FACES
            if upper_face:
                display.DisplayShape(self.generated_upper_face, update=True)
            if lower_face:
                display.DisplayShape(self.generated_lower_face, update=True)
            if tip:
                display.DisplayShape(self.generated_tip, update=True)
            if root:
                display.DisplayShape(self.generated_root, update=True)
            start_display()



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

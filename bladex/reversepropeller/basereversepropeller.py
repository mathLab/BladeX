import sys
import numpy as np
import csv
from bladex import CustomProfile, Blade, Propeller, Shaft
from .reversepropellerinterface import ReversePropellerInterface
from OCC.Core.IGESControl import (IGESControl_Reader)
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex,\
             BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeSolid, \
             BRepBuilderAPI_Sewing, BRepBuilderAPI_NurbsConvert
from OCC.Core.BRep import BRep_Tool
import OCC.Core.TopoDS
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE, TopAbs_SHELL
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Vec, gp_Pln
from OCC.Core.TColgp import  TColgp_Array1OfPnt
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.BRepAdaptor import (BRepAdaptor_Curve,
                                  BRepAdaptor_CompCurve)
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.Geom import Geom_Plane
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 


class BaseReversePropeller(ReversePropellerInterface):
    """
    Base class for the extraction of the parameters of a blade and reconstruction of the parametrized
    blade and of the entire propeller.

    Depending on the source code of the generated blade, different child classes can be chosen.


    Given the IGES file of the blade, the following parameters are extracted:

        - :math:`(X, Y)` coordinates of the blade cylindrical sections after
          being expanded in 2D to create airfoils.

        - Chord length, for each cylindrical section.

        - Pitch :math:`(P)`, for each cylindrical section.

        - Rake :math:`(k)`, in distance units, for each cylindrical section.

        - Skew angle :math:`(\\theta_s)`, for each cylindrical section, expressed
          in degrees.

    The parameters can be saved in a csv file with the method
    'save_global_parameters' and used to reconstruct the blade, which is saved
    in a IGES file (method 'reconstruct_blade').
    The coordinates of each section can be extracted with the method
    'reconstruct_sections'.
    Given the shaft, the whole propeller can be reconstructed and saved in a
    IGES file with the method 'reconstruct_propeller'.

    --------------------------

    :param filename: path to the IGES file of the blade.
    :param list radii_list: list which contains the radii values of the
        sectional profiles.
    :param num_points_top_bottom: number of points used to interpolate each
        sectional profile.
    """
    def __init__(self, filename, radii_list, num_points_top_bottom):

        self.iges_file = filename  #filename is the path to the file iges
        self.radii_list = radii_list  #radii at which we want to measure properties
        self.section_points_number = 1200  #number of points for each section profile
        self.airfoil_points_number = 1000
        self.num_points_top_bottom = num_points_top_bottom  #number of points used to reconstruct the profile, equal for top and lower parts of the profile
        # self.num_sections = len(self.radii_list)
        # Initialize things we want to compute from the blade IGES file
        self.pitch_angles_list = []
        self.pitch_list = []
        self.skew_angles_list = []
        self.skew_list = []
        self.rake_list = []
        self.chord_length_list = []
        self.blade_solid = None
        self.blade_compound = None
        self.tolerance_solid = 5e-2
        self._extract_solid_from_file()
        num_sections = len(radii_list)
        self.xup = np.zeros((num_sections, self.num_points_top_bottom))
        self.xdown = np.zeros((num_sections, self.num_points_top_bottom))
        self.yup = np.zeros((num_sections, self.num_points_top_bottom))
        self.ydown = np.zeros((num_sections, self.num_points_top_bottom))
        # Lists of OCC objects used for plotting in display_* methods
        self.cylinder_lateral_faces_list = []
        self.section_wires_list = []
        self.camber_wires_list = []

    def _extract_solid_from_file(self):
        """
        Private method that reads the IGES file of the blade and constructs
        the correspondent solid object
        """
        # Extract the blade from the IGES file; build it as a solid with OCC
        iges_reader = IGESControl_Reader()
        iges_reader.ReadFile(self.iges_file)
        iges_reader.TransferRoots()
        sewer = BRepBuilderAPI_Sewing(self.tolerance_solid)
        # Case where we have Faces and not closed surface. 
        # This is the case for BladeX generated blade
        if iges_reader.Shape().ShapeType() == 4:
            self.blade_compound = iges_reader.OneShape()
            exp = TopExp_Explorer(self.blade_compound, TopAbs_FACE)
            while exp.More():
                sewer.Add(exp.Current())
                exp.Next()
        else:
            self.blade_compound = iges_reader.Shape()
            sewer.Add(self.blade_compound)
        sewer.Perform()
        result_sewed_blade = sewer.SewedShape()
        blade_solid_maker = BRepBuilderAPI_MakeSolid()
        if result_sewed_blade.ShapeType() == 0:
            exp = TopExp_Explorer(result_sewed_blade, TopAbs_SHELL)
            while exp.More():
                shell = topods.Shell(exp.Current())
                blade_solid_maker.Add(shell)
                exp.Next()
        else:
            blade_solid_maker.Add(OCC.Core.TopoDS.topods.Shell(result_sewed_blade))
        if (blade_solid_maker.IsDone()):
            self.blade_solid = blade_solid_maker.Solid()
        else:
            self.blade_solid = result_sewed_blade

    def _build_cylinder(self, radius):
        """
        Private method that builds the cylinder which intersects the blade
        at a specific cylindrical section.
        Argument 'radius' is the radius value corresponding to the cylindrical
        section taken into account. This method is applied to all the radii in
        list 'radii_list' given as input.
        """
        # Base point such that cylinder intersect all section
        bbox = Bnd_Box()
        brepbndlib.Add(self.blade_solid, bbox)
        xmin, _, _, xmax, _, _ = bbox.Get()

        axis = gp_Ax2(gp_Pnt(xmin-0.2*abs(xmin), 0.0, 0.0),
                      gp_Dir(1.0, 0.0, 0.0))
        self.cylinder = BRepPrimAPI_MakeCylinder(axis, radius, 1.2*(xmax-xmin)).Shape()
        faceCount = 0
        faces_explorer = TopExp_Explorer(self.cylinder, TopAbs_FACE)

        while faces_explorer.More():
            face = OCC.Core.TopoDS.topods.Face(faces_explorer.Current())
            faceCount += 1
            # Convert the cylinder faces into Non Uniform Rational Basis-Splines geometry (NURBS)
            nurbs_converter = BRepBuilderAPI_NurbsConvert(face)
            nurbs_converter.Perform(face)
            nurbs_face_converter = nurbs_converter.Shape()
            nurbs_face = OCC.Core.TopoDS.topods.Face(nurbs_face_converter)

            surface = BRep_Tool.Surface(OCC.Core.TopoDS.topods.Face(nurbs_face))
            self.bounds = 0.0
            self.bounds = surface.Bounds()

            # Compute the normal curve to the surfaces, specifying the bounds considered,
            # the maximum order of the derivative we want to compute, the linear tolerance
            normal = GeomLProp_SLProps(surface,
                                       (self.bounds[0] + self.bounds[1]) / 2.0,
                                       (self.bounds[2] + self.bounds[3]) / 2.0,
                                       1, self.linear_tolerance).Normal()
            if (normal * axis.Direction() == 0):
                self.cylinder_lateral_face = face
                self.cylinder_lateral_surface = surface
                self.cylinder_lateral_faces_list.append(surface)
            faces_explorer.Next()

    def _extract_parameters_and_transform_profile(self, radius):
        """
        Private method which extracts the parameters (pitch, rake, skew ,...) related
        to a specific section of the blade, and then transforms the camber points
        initialized in _initial_camber_points_plane and the prile points initialized
        in _initial_airfoil_points_plane according to the parameters found.
        This method trasforms the leading and trailing edges initialed in
        _initial_leading_trailing_edges and the mid chord point as well.

        Basically all points and edges of the profile section are transformed
        to fulfill the properties of the specific sectionwe are considering.
        """
        self.pitch_angle = np.arctan2(
            self.leading_edge_point_on_plane[1] -
            self.trailing_edge_point_on_plane[1],
            self.leading_edge_point_on_plane[0] -
            self.trailing_edge_point_on_plane[0])
        self.skew = self.mid_chord_point_on_plane[1]
        self.rake_induced_by_skew = self.skew / np.tan(self.pitch_angle)

        self.airfoil_points_on_plane[:, 1:2] -= self.skew
        self.camber_points_on_plane[:, 1:2] -= self.skew
        self.leading_edge_point_on_plane[1] -= self.skew
        self.trailing_edge_point_on_plane[1] -= self.skew
        self.mid_chord_point_on_plane[1] -= self.skew


        self.airfoil_points_on_plane[:, 0:1] -= self.rake_induced_by_skew
        self.camber_points_on_plane[:, 0:1] -= self.rake_induced_by_skew
        self.leading_edge_point_on_plane[0] -= self.rake_induced_by_skew
        self.trailing_edge_point_on_plane[0] -= self.rake_induced_by_skew
        self.mid_chord_point_on_plane[0] -= self.rake_induced_by_skew


        self.rake = self.mid_chord_point_on_plane[0]

        self.airfoil_points_on_plane[:, 0:1] -= self.rake
        self.camber_points_on_plane[:, 0:1] -= self.rake
        self.leading_edge_point_on_plane[0] -= self.rake
        self.trailing_edge_point_on_plane[0] -= self.rake
        self.mid_chord_point_on_plane[0] -= self.rake

        self.rotation_matrix = np.zeros((2, 2))
        self.rotation_matrix[0][0] = np.cos(-self.pitch_angle)
        self.rotation_matrix[0][1] = -np.sin(-self.pitch_angle)
        self.rotation_matrix[1][0] = np.sin(-self.pitch_angle)
        self.rotation_matrix[1][1] = np.cos(-self.pitch_angle)

        self.airfoil_points_on_plane = self.airfoil_points_on_plane.dot(
            self.rotation_matrix.transpose())
        self.camber_points_on_plane = self.camber_points_on_plane.dot(
            self.rotation_matrix.transpose())
        self.leading_edge_point_on_plane = self.leading_edge_point_on_plane.dot(
            self.rotation_matrix.transpose())
        self.trailing_edge_point_on_plane = self.trailing_edge_point_on_plane.dot(
            self.rotation_matrix.transpose())
        self.mid_chord_point_on_plane = self.mid_chord_point_on_plane.dot(
            self.rotation_matrix.transpose())


        self.chord_length = ((self.leading_edge_point_on_plane[0] -
                              self.trailing_edge_point_on_plane[0])**2 +
                             (self.leading_edge_point_on_plane[1] -
                              self.trailing_edge_point_on_plane[1])**2)**.5

        self.airfoil_points_on_plane[:, 0:2] /= self.chord_length
        self.camber_points_on_plane[:, 0:2] /= self.chord_length
        self.leading_edge_point_on_plane[0:1] /= self.chord_length
        self.trailing_edge_point_on_plane[0:1] /= self.chord_length
        self.mid_chord_point_on_plane[0:1] /= self.chord_length

        self.airfoil_points_on_plane[:, 0:1] *= -1.0
        self.camber_points_on_plane[:, 0:1] *= -1.0
        self.leading_edge_point_on_plane[0] *= -1.0
        self.trailing_edge_point_on_plane[0] *= -1.0
        self.mid_chord_point_on_plane[0] *= -1.0

        self.airfoil_points_on_plane[:, 0:1] += 0.5
        self.camber_points_on_plane[:, 0:1] += 0.5
        self.leading_edge_point_on_plane[0] += 0.5
        self.trailing_edge_point_on_plane[0] += 0.5
        self.mid_chord_point_on_plane[0] += 0.5

        self.camber_points_on_plane = np.matrix(self.camber_points_on_plane)
        self.camber_points_on_plane[0, 0] = 0.0
        self.camber_points_on_plane[-1, 0] = 1.0

        self.camber_points_on_plane = np.sort(
            self.camber_points_on_plane.view('float64,float64'),
            order=['f0'],
            axis=0).view(np.float64)
        self.f_camber = interp1d(
            np.squeeze(np.asarray(self.camber_points_on_plane[:, 0])),
            np.squeeze(np.asarray(self.camber_points_on_plane[:, 1])),
            kind='cubic')
        
    def save_global_parameters(self, filename_csv):
        """
        Method that writes all blade properties and points of the sections
        in a csv file named filename_csv
        """

        # open the file in the write mode
        with open(filename_csv, 'w') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow(("Radii list: ", self.radii_list))
            writer.writerow(("Pitch angles list: ", self.pitch_angles_list))
            writer.writerow(("Pitch list: ", self.pitch_list))
            writer.writerow(("Skew angles list: ", self.skew_angles_list))
            writer.writerow(("Rake list: ", self.rake_list))
            writer.writerow(("Chord length list: ", self.chord_length_list))
            for j in range(len(self.radii_list)):
                writer.writerow(("x section" + str(j) + " upper coordinates: ",
                                 self.xup[j, :]))
                writer.writerow(("x section" + str(j) + " lower coordinates: ",
                                 self.xdown[j, :]))
                writer.writerow(("y section" + str(j) + " upper coordinates: ",
                                 self.yup[j, :]))
                writer.writerow(("y section" + str(j) + " lower coordinates: ",
                                 self.ydown[j, :]))

    def reconstruct_sections(self):
        """
        Method that reconstructs single sections of the blade starting from the
        points computed from the original IGES file.
        If sections are not enough to reconstruct the blade, one can just export sections
        and then reconstruct the blade in a dedicated script.
        """

        for j in range(self.num_sections):
            self.recons_sections[j] = CustomProfile(
                xup=self.xup[j, :],
                yup=self.yup[j, :],
                xdown=self.xdown[j, :],
                ydown=self.ydown[j, :])
        return self.recons_sections

    def reconstruct_blade(self):
        """
        Method that reconstructs the blade starting from the sections
        computed from the original IGES file, and then reconstruct the 4 parts
        of the blade (upper, lower face, tip, root).
        """

        for j in range(self.num_sections):
            self.recons_sections[j] = CustomProfile(
                xup=self.xup[j, :],
                yup=self.yup[j, :],
                xdown=self.xdown[j, :],
                ydown=self.ydown[j, :])
        radii = np.array(self.radii_list) / 1000
        self.recons_blade = Blade(sections=self.recons_sections,
                                  radii=radii,
                                  chord_lengths=self.chord_length_list,
                                  pitch=self.pitch_list,
                                  rake=self.rake_list,
                                  skew_angles=self.skew_angles_list)
        self.recons_blade.apply_transformations()
        self.recons_blade.generate_iges_blade('iges_reconstructed.iges')

    def reconstruct_propeller(self, propeller_iges, shaft_iges, n_blades):
        """
        Method which reconstructs the whole propeller with shaft starting from
        the iges file of the shaft and the number of blades. The whole propeller
        is saved in an IGES file named filename_iges (must be given as input).
        """
        shaft_path = shaft_iges
        prop_shaft = Shaft(shaft_path)
        prop = Propeller(prop_shaft, self.recons_blade, n_blades)
        prop.generate_iges(propeller_iges)
        prop.display()

    def display_cylinder(self, radius):
        """
        Method that displays the cylinder and the blade intersecting each others.
        If the radius is not in self.radii_list, error is raised
        """
        if radius not in self.radii_list:
            raise ValueError("The radius must be among the ones passed in the constructor of the object. The unit is mm.")
        index_radius = np.where(self.radii_list == radius)[0][0]
        display, start_display, add_menu, add_function_to_menu = init_display()
        display.DisplayShape(self.blade_solid, update=True)
        display.DisplayShape(self.cylinder_lateral_faces_list[index_radius], update=True)
        display.DisplayShape(self.section_wires_list[index_radius], update=True)
        start_display()

    def display_section(self, radius):
        """
        Method that displays a section profile over the cylinder with its camber line.
        """
        if radius not in self.radii_list:
            raise ValueError("The radius must be among the ones passed in the constructor of the object. The unit is mm.")
        index_radius = np.where(self.radii_list == radius)[0][0]
        display, start_display, add_menu, add_function_to_menu = init_display()
        display.DisplayShape(self.cylinder_lateral_faces_list[index_radius], update=True)
        display.DisplayShape(self.section_wires_list[index_radius], update=True)
        display.DisplayShape(self.camber_wires_list[index_radius], color="GREEN", update=True)
        start_display()

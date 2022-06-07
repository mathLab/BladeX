"""
Module for the extraction of the parameters of a blade and for the 
approximated reconstruction of the blade and the whole propeller.
"""

import os, errno
import os.path
import matplotlib.pyplot as plt
import numpy as np
import csv
from bladex import NacaProfile, CustomProfile, Blade, Propeller, Shaft
from OCC.Core.IGESControl import (IGESControl_Reader, IGESControl_Writer,
                                  IGESControl_Controller_Init)
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex,\
             BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeSolid, \
             BRepBuilderAPI_Sewing, BRepBuilderAPI_Transform, BRepBuilderAPI_NurbsConvert, \
             BRepBuilderAPI_MakeFace
from OCC.Core.BRep import (BRep_Tool, BRep_Builder, BRep_Tool_Curve,
                           BRep_Tool_CurveOnSurface)
import OCC.Core.TopoDS
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.BRepAlgo import BRepAlgo_Section
from OCC.Core.TopTools import TopTools_ListOfShape, TopTools_MapOfShape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.gp import gp_Pnt, gp_Pnt2d, gp_Dir, gp_Ax1, gp_Ax2, gp_Trsf, gp_Pln, gp_Vec, gp_Lin
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.TColgp import TColgp_HArray1OfPnt, TColgp_Array1OfPnt
from OCC.Core.GeomAPI import GeomAPI_Interpolate, GeomAPI_IntCS, GeomAPI_ProjectPointOnSurf
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_HCurve
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.BRep import BRep_Tool
from OCC.Core.IntTools import IntTools_FClass2d
from OCC.Core.TopAbs import TopAbs_IN
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.TopoDS import topods, TopoDS_Edge, TopoDS_Compound
from subprocess import call
from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCC.Core.Adaptor3d import Adaptor3d_Curve, Adaptor3d_HCurve
from OCC.Core.Geom import Geom_Line
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepGProp import (brepgprop_LinearProperties,
                                brepgprop_SurfaceProperties,
                                brepgprop_VolumeProperties)
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.BRepAdaptor import (BRepAdaptor_Curve, BRepAdaptor_HCurve,
                                  BRepAdaptor_CompCurve, BRepAdaptor_HCompCurve)
from OCC.Core.GCPnts import GCPnts_UniformDeflection
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import (GeomAbs_C0, GeomAbs_G1, GeomAbs_C1, GeomAbs_G2,
                              GeomAbs_C2, GeomAbs_C3, GeomAbs_CN)
from OCC.Core.GeomProjLib import geomprojlib
from OCC.Core.TopExp import topexp
from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def distance(P1, P2):
    """
    distance between two points
    """
    return ((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2)**0.5


def optimized_path(coords, start=None):
    if start is None:
        start = coords[0]
    pass_by = coords
    path = [start]
    pass_by.remove(start)
    while pass_by:
        nearest = min(pass_by, key=lambda x: distance(path[-1], x))
        path.append(nearest)
        pass_by.remove(nearest)
    return path


def point_inside_polygon(x, y, poly, include_edges=True):
    '''
    Test if point (x,y) is inside polygon poly.

    poly is N-vertices polygon defined as 
    [(x1,y1),...,(xN,yN)] or [(x1,y1),...,(xN,yN),(x1,y1)]
    (function works fine in both cases)

    Geometrical idea: point is inside polygon if horisontal beam
    to the right from point crosses polygon even number of times. 
    Works fine for non-convex polygons.
    '''
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if p1y == p2y:
            if y == p1y:
                if min(p1x, p2x) <= x <= max(p1x, p2x):
                    # point is on horisontal edge
                    inside = include_edges
                    break
                elif x < min(p1x,
                             p2x):  # point is to the left from current edge
                    inside = not inside
        else:  # p1y!= p2y
            if min(p1y, p2y) <= y <= max(p1y, p2y):
                xinters = (y - p1y) * (p2x - p1x) / float(p2y - p1y) + p1x

                if x == xinters:  # point is right on the edge
                    inside = include_edges
                    break

                if x < xinters:  # point is to the left from current edge
                    inside = not inside

        p1x, p1y = p2x, p2y

    return inside


class ReversePropeller(object):
    """
    Extraction of the parameters of a blade and reconstruction of the parametrized
    blade and of the entire propeller.

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
        self.coords = []
        self.num_sections = len(self.radii_list)
        self.recons_sections = [0] * self.num_sections
        self.start = None
        self.x = []
        self.y = []
        self.poly = []
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
        self.xup = np.zeros((self.num_sections, self.num_points_top_bottom))
        self.xdown = np.zeros((self.num_sections, self.num_points_top_bottom))
        self.yup = np.zeros((self.num_sections, self.num_points_top_bottom))
        self.ydown = np.zeros((self.num_sections, self.num_points_top_bottom))
        # At each radius, i.e., for each section, the cylinder with that radius is built
        # and the faces of the cylinder are isolated.

        ind_sec = 0
        for radius in self.radii_list:
            self.cylinder = None
            self.bounds = None
            self.cylinder_lateral_face = None
            self.cylinder_lateral_surface = None
            self.linear_tolerance = 1e-3
            self.conversion_unit = 1000
            self._build_cylinder(radius)
            self._build_intersection_cylinder_blade(radius)
            self.leading_edge_point = []
            self.trailing_edge_point = []
            self.edge_3d = None
            self.camber_curve_edge = None
            self.first_edge = None
            self.last_edge = None
            self.firstSegment = None
            self.lastSegment = None
            self.camber_points_on_plane = None
            self.leading_edge_point_on_plane = []
            self.trailing_edge_point_on_plane = []
            # Compute a number section_points_number of points on the parametric curve
            # at a equal relative distance (taken on the curvilinear ascissa)
            self.param_plane_points = []
            self.orig_param_plane_points = []
            for i in range(self.section_points_number):
                self.i = i
                self._camber_points(radius)
            self._voronoi_points(radius)
            self._camber_curve(radius)
            self._initial_leading_trailing_edges_plane(radius)
            self._initial_camber_points_plane(radius)
            self._initial_airfoil_points_plane(radius)
            self._extract_parameters_and_transform_profile(radius)
            self._airfoil_top_and_bottom_points()
            self.xup[ind_sec, :] = self.ascissa
            self.xdown[ind_sec, :] = self.ascissa
            self.yup[ind_sec, :] = self.int_airfoil_top
            self.ydown[ind_sec, :] = self.int_airfoil_bottom
            ind_sec = ind_sec + 1

    def _extract_solid_from_file(self):
        """
        Private method that reads the IGES file of the blade and constructs
        the correspondent solid object
        """
        # Extract the blade from the IGES file; build it as a solid with OCC
        iges_reader = IGESControl_Reader()
        iges_reader.ReadFile(self.iges_file)
        iges_reader.TransferRoots()
        self.blade_compound = iges_reader.Shape()
        sewer = BRepBuilderAPI_Sewing(self.tolerance_solid)
        sewer.Add(self.blade_compound)
        sewer.Perform()
        result_sewed_blade = sewer.SewedShape()
        blade_solid_maker = BRepBuilderAPI_MakeSolid()
        blade_solid_maker.Add(OCC.Core.TopoDS.topods_Shell(result_sewed_blade))
        if (blade_solid_maker.IsDone()):
            self.blade_solid = blade_solid_maker.Solid()
        else:
            self.blade_solid = result_sewed_blade

    def _build_cylinder(self, radius):
        """
        Private method that builds the cylinder which  intersects the blade at a 
        specific cylindrical section. 
        Argument 'radius' is the radius value corresponding to the cylindrical section
        taken into account.
        This method is applied to all the radii in list 'radii_list' given as input.
        """
        axis = gp_Ax2(gp_Pnt(-0.5*self.conversion_unit, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0))
        self.cylinder = BRepPrimAPI_MakeCylinder(axis, radius, 1*self.conversion_unit).Shape()
        faceCount = 0
        faces_explorer = TopExp_Explorer(self.cylinder, TopAbs_FACE)

        while faces_explorer.More():
            face = OCC.Core.TopoDS.topods_Face(faces_explorer.Current())
            faceCount += 1
            # Convert the cylinder faces into Non Uniform Rational Basis-Splines geometry (NURBS)
            nurbs_converter = BRepBuilderAPI_NurbsConvert(face)
            nurbs_converter.Perform(face)
            nurbs_face_converter = nurbs_converter.Shape()
            nurbs_face = OCC.Core.TopoDS.topods_Face(nurbs_face_converter)

            surface = BRep_Tool.Surface(OCC.Core.TopoDS.topods_Face(nurbs_face))
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
            faces_explorer.Next()

    def _build_intersection_cylinder_blade(self, radius):
        """
        Private method that constructs the section lines which are the intersections 
        between the cylinder at a fixed radius and the blade, and the camber points.
        """
        # Construction of the section lines between two shapes (in this case the
        # blade and the lateral face of the cylinder)
        section_builder = BRepAlgo_Section(self.blade_solid,
                                           self.cylinder_lateral_face, False)
        # Define and build the parametric 2D curve (pcurve) for the section lines defined above
        section_builder.ComputePCurveOn2(True)
        section_builder.Build()
        self.section = section_builder.Shape()
        edgeExplorer = TopExp_Explorer(self.section, TopAbs_EDGE)
        wire_maker = BRepBuilderAPI_MakeWire()

        edgeCount = 0
        self.total_section_length = 0.0
        edgeList = TopTools_ListOfShape()
        while edgeExplorer.More():
            edgeCount = edgeCount + 1
            edge = topods.Edge(edgeExplorer.Current())
            edgeList.Append(edge)
            edgeExplorer.Next()
        wire_maker.Add(edgeList)
        self.wire = wire_maker.Wire()
        # Create a 3D curve from a wire
        self.curve_adaptor = BRepAdaptor_CompCurve(
            OCC.Core.TopoDS.topods_Wire(self.wire))
        # Length of the curve section (ascissa curvilinea)
        self.total_section_length = GCPnts_AbscissaPoint.Length(
            self.curve_adaptor)

    def _camber_points(self, radius):
        """
        Private method which computes the single points of the camber curve and
        collects them in param_plane_points and in orig_plane_points.

        """
        firstParam = self.curve_adaptor.FirstParameter()
        rel_distance = float(self.i) / float(
            self.section_points_number) * self.total_section_length
        point_generator = GCPnts_AbscissaPoint(1e-7, self.curve_adaptor,
                                               rel_distance, firstParam)
        param = point_generator.Parameter()
        self.point = self.curve_adaptor.Value(param)
        # Compute the angle for polar reference system
        theta = np.arctan2(self.point.Y(), self.point.Z())
        if theta < 0:
            theta += 2.0 * np.pi
        # Param_plane_points is the list of all points parametrized in cylindric coordinates,
        # i.e., the coordinate along X (the axis of the cylinder) and the
        # radius multiplying the polar angle.
        self.orig_param_plane_points.append([self.point.X(), radius * theta])
        self.param_plane_points.append([self.point.X(), radius * theta])

    def _voronoi_points(self, radius):
        """
        Private method which computes the points of the Voronoi map for each 
        section of the blade.

        """
        # update the bounds of the section
        self.bounds = [0.0, 0.0, 0.0, 0.0]
        for p in self.param_plane_points:
            self.bounds[0] = min(self.bounds[0], p[0])
            self.bounds[1] = max(self.bounds[1], p[0])
            self.bounds[2] = min(self.bounds[2], p[1])
            self.bounds[3] = max(self.bounds[3], p[1])
        # Create the Voronoi diagram for the 2D points
        vor = Voronoi(self.param_plane_points)
        vor_points = []
        min_u = 1e8
        self.start = vor.vertices[0]
        # Find in an iterative way the Voronoi points
        for i in range(len(vor.vertices)):
            p = [vor.vertices[i][0], vor.vertices[i][1]]
            if (point_inside_polygon(p[0],
                                     p[1],
                                     self.param_plane_points,
                                     include_edges=True)):
                if (p[0] < min_u):
                    min_u = p[0]
                    self.start = p
                vor_points.append(p)
        optimized_vor_points = optimized_path(vor_points, self.start)
        us = []
        vs = []
        uss = []
        vss = []
        self.vor_us = []
        self.vor_vs = []
        for p in self.orig_param_plane_points:
            us.append(p[0])
            vs.append(p[1])
        for p in self.param_plane_points:
            uss.append(
                p[0]
            )  # coordinates of the points of the original parametric curve
            vss.append(p[1])
        for p in optimized_vor_points:
            self.vor_us.append(
                p[0])  # coordinates of the points of the Voronoi map
            self.vor_vs.append(
                p[1]
            )  #vor_us includes the X coordinates and vor_vs the Z coordinates
        us.append(self.orig_param_plane_points[0][0])
        vs.append(self.orig_param_plane_points[0][1])

    def _camber_curve(self, radius):
        """
        Private method which constructs the camber curve by interpolating the
        points found with the method _camber_points. Also the leading and the 
        trailing edge are found and assembled with the camber line in a unique 
        object.
        """
        # Create objects for the points of the Voronoi map, construct Splines
        # to construct the camber line, the upper and lower part of each section profile.
        Pnts = TColgp_Array1OfPnt(1, len(self.vor_us))
        for i in range(len(self.vor_us)):
            # from cartesian to cylindric coordinates
            Pnts.SetValue(
                i + 1,
                gp_Pnt(self.vor_us[i], radius * np.sin(self.vor_vs[i] / radius),
                       radius * np.cos(self.vor_vs[i] / radius)))

        # Build the spline curve of degree 3 from Voronoi points, which are those on the camber line
        # with tolerance 1e-1
        spline_builder = GeomAPI_PointsToBSpline(Pnts, 3, 3, GeomAbs_C2, 1e-1)
        camber_curve = spline_builder.Curve()
        # BUild the camber line from 3D edges
        self.edge_3d = BRepBuilderAPI_MakeEdge(camber_curve).Edge()
        camber_curve_adaptor = BRepAdaptor_Curve(
            OCC.Core.TopoDS.topods_Edge(self.edge_3d))
        # Compute the length of the camber curve for each section
        camber_curve_length = GCPnts_AbscissaPoint.Length(camber_curve_adaptor)
        relative_tolerance = 2.5e-3
        absolute_tolerance = relative_tolerance * camber_curve_length
        firstPoint = gp_Pnt(0.0, 0.0, 0.0)
        lastPoint = gp_Pnt(0.0, 0.0, 0.0)
        firstTangent = gp_Vec(0.0, 0.0, 0.0)
        lastTangent = gp_Vec(0.0, 0.0, 0.0)
        # Compute the first and last points and the related derivative of the camber curve w.r.t.
        camber_curve_adaptor.D1(camber_curve_adaptor.FirstParameter(),
                                firstPoint, firstTangent)
        camber_curve_adaptor.D1(camber_curve_adaptor.LastParameter(), lastPoint,
                                lastTangent)
        dummyFirstPoint = gp_Pnt(0.0, 0.0, 0.0)
        dummyLastPoint = gp_Pnt(0.0, 0.0, 0.0)
        usedFirstTangent = gp_Vec(0.0, 0.0, 0.0)
        usedLastTangent = gp_Vec(0.0, 0.0, 0.0)
        camber_curve_adaptor.D1(
            camber_curve_adaptor.FirstParameter() + 0.005 *
            (camber_curve_adaptor.LastParameter() -
             camber_curve_adaptor.FirstParameter()), dummyFirstPoint,
            usedFirstTangent)
        camber_curve_adaptor.D1(
            camber_curve_adaptor.FirstParameter() + 0.985 *
            (camber_curve_adaptor.LastParameter() -
             camber_curve_adaptor.FirstParameter()), dummyLastPoint,
            usedLastTangent)
        firstTangent = gp_Dir(usedFirstTangent)
        lastTangent = gp_Dir(usedLastTangent)
        firstPoint = dummyFirstPoint
        lastPoint = dummyLastPoint
        self.camber_curve_edge = BRepBuilderAPI_MakeEdge(
            camber_curve,
            camber_curve_adaptor.FirstParameter() + 0.005 *
            (camber_curve_adaptor.LastParameter() -
             camber_curve_adaptor.FirstParameter()),
            camber_curve_adaptor.FirstParameter() + 0.985 *
            (camber_curve_adaptor.LastParameter() -
             camber_curve_adaptor.FirstParameter())).Edge()
        # Set of the leading edges
        self.firstSegment = BRepBuilderAPI_MakeEdge(
            firstPoint,
            gp_Pnt(
                firstPoint.X() - 10.0 * firstTangent.X() * absolute_tolerance,
                firstPoint.Y() - 10.0 * firstTangent.Y() * absolute_tolerance,
                firstPoint.Z() -
                10.0 * firstTangent.Z() * absolute_tolerance)).Edge()

        # Set of trailing edges
        self.lastSegment = BRepBuilderAPI_MakeEdge(
            lastPoint,
            gp_Pnt(lastPoint.X() + 10.0 * lastTangent.X() * absolute_tolerance,
                   lastPoint.Y() + 10.0 * lastTangent.Y() * absolute_tolerance,
                   lastPoint.Z() +
                   10.0 * lastTangent.Z() * absolute_tolerance)).Edge()
        # Visiting the vertices of the chord line and storing the leading edges
        vertexExplorer = TopExp_Explorer(self.wire, TopAbs_VERTEX)
        found = False
        while vertexExplorer.More():
            vertex = topods.Vertex(vertexExplorer.Current())
            point = BRep_Tool.Pnt(vertex)
            if (point.Distance(firstPoint) <
                    camber_curve_length * relative_tolerance):
                self.leading_edge_point = point
                found = True
            vertexExplorer.Next()
        if found == False:
            distSS = BRepExtrema_DistShapeShape(self.wire, self.firstSegment)
            minDist = 1e8
            for k in range(distSS.NbSolution()):
                if firstPoint.Distance(distSS.PointOnShape1(k + 1)) < minDist:
                    minDist = firstPoint.Distance(distSS.PointOnShape1(k + 1))
                    self.leading_edge_point = distSS.PointOnShape1(k + 1)

        # Visiting the vertices of the chord line and storing the trailing edges
        found = False
        while vertexExplorer.More():
            vertex = topods.Vertex(vertexExplorer.Current())
            point = BRep_Tool.Pnt(vertex)
            if (point.Distance(lastPoint) <
                    camber_curve_length * relative_tolerance):
                self.trailing_edge_point = point
                found = True
            vertexExplorer.Next()
        if found == False:
            distSS = BRepExtrema_DistShapeShape(self.wire, self.lastSegment)
            minDist = 1e8

            for k in range(distSS.NbSolution()):
                if lastPoint.Distance(distSS.PointOnShape1(k + 1)) < minDist:
                    minDist = lastPoint.Distance(distSS.PointOnShape1(k + 1))
                    self.trailing_edge_point = distSS.PointOnShape1(k + 1)

        if (self.trailing_edge_point.X() > self.leading_edge_point.X()):
            serv_point = self.trailing_edge_point
            self.trailing_edge_point = self.leading_edge_point
            self.leading_edge_point = serv_point
            serv_point = lastPoint
            lastPoint = firstPoint
            firstPoint = serv_point

        # Edges
        self.first_edge = BRepBuilderAPI_MakeEdge(self.trailing_edge_point,
                                                  lastPoint).Edge()
        self.last_edge = BRepBuilderAPI_MakeEdge(self.leading_edge_point,
                                                 firstPoint).Edge()

        # Storing the "full" camber line (camber line = leading and trailing edges)
        # in a unique object with OCC wire
        wire_maker = BRepBuilderAPI_MakeWire()
        wire_maker.Add(self.first_edge)
        wire_maker.Add(self.camber_curve_edge)
        wire_maker.Add(self.last_edge)

        full_camber_wire = wire_maker.Wire()
        self.full_camber_curve_adaptor = BRepAdaptor_CompCurve(
            OCC.Core.TopoDS.topods_Wire(full_camber_wire))
        self.full_camber_length = GCPnts_AbscissaPoint.Length(
            self.full_camber_curve_adaptor)

    def _initial_leading_trailing_edges_plane(self, radius):
        """
        Private method which computes the coordinates of the leading and trailing
        edges of each section, as were in the initial blade. Thenm transformations 
        will be applied in order to plot the points of the profiles on a plane
        and to map the X coordinates in the interval [0,1].
        """

        leading_edge_theta = np.arctan2(self.leading_edge_point.Y(),
                                        self.leading_edge_point.Z())
        if leading_edge_theta < 0:
            leading_edge_theta += 2.0 * np.pi
        self.leading_edge_point_on_plane = np.array(
            [self.leading_edge_point.X(), radius * leading_edge_theta])

        trailing_edge_theta = np.arctan2(self.trailing_edge_point.Y(),
                                         self.trailing_edge_point.Z())
        if trailing_edge_theta < 0:
            trailing_edge_theta += 2.0 * np.pi
        self.trailing_edge_point_on_plane = np.array(
            [self.trailing_edge_point.X(), radius * trailing_edge_theta])

    def _initial_camber_points_plane(self, radius):
        """
        Private method that defines the points of the camber line projected on
        plane, in 2D coordinates.
        """
        camber_points_number = int(round(self.airfoil_points_number / 2))

        self.camber_points_on_plane = np.zeros((camber_points_number, 2))

        for i in range(camber_points_number):
            firstParam = self.full_camber_curve_adaptor.FirstParameter()
            rel_distance = float(i) / float(camber_points_number -
                                            1) * self.full_camber_length
            param = GCPnts_AbscissaPoint(1e-7, self.full_camber_curve_adaptor,
                                         rel_distance, firstParam).Parameter()
            point = self.full_camber_curve_adaptor.Value(param)
            theta = np.arctan2(point.Y(), point.Z())
            if theta < 0:
                theta += 2.0 * np.pi
            self.camber_points_on_plane[i][0] = point.X()
            self.camber_points_on_plane[i][1] = radius * theta

    def _initial_airfoil_points_plane(self, radius):
        """
        Private method that evaluates the airfoil points on a plane (the points of
        the profile of a single section, which will be then distinguished into 
        upper and lower part). Those points are defined in a plane, in 2D coordinates.
        """

        self.airfoil_points_on_plane = np.zeros((self.airfoil_points_number, 2))

        for i in range(self.airfoil_points_number):
            firstParam = self.curve_adaptor.FirstParameter()
            rel_distance = float(i) / float(self.airfoil_points_number -
                                            1) * self.total_section_length
            param = GCPnts_AbscissaPoint(1e-7, self.curve_adaptor, rel_distance,
                                         firstParam).Parameter()
            point = self.curve_adaptor.Value(param)
            theta = np.arctan2(point.Y(), point.Z())
            if theta < 0:
                theta += 2.0 * np.pi

            self.airfoil_points_on_plane[i][0] = point.X()
            self.airfoil_points_on_plane[i][1] = radius * theta

        # Compute the mid point of the chord line
        self.mid_chord_point_on_plane = (
            self.leading_edge_point_on_plane +
            self.trailing_edge_point_on_plane) / 2.0

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

        rotation_matrix = np.zeros((2, 2))
        rotation_matrix[0][0] = np.cos(-self.pitch_angle)
        rotation_matrix[0][1] = -np.sin(-self.pitch_angle)
        rotation_matrix[1][0] = np.sin(-self.pitch_angle)
        rotation_matrix[1][1] = np.cos(-self.pitch_angle)

        self.airfoil_points_on_plane = self.airfoil_points_on_plane.dot(
            rotation_matrix.transpose())
        self.camber_points_on_plane = self.camber_points_on_plane.dot(
            rotation_matrix.transpose())
        self.leading_edge_point_on_plane = self.leading_edge_point_on_plane.dot(
            rotation_matrix.transpose())
        self.trailing_edge_point_on_plane = self.trailing_edge_point_on_plane.dot(
            rotation_matrix.transpose())
        self.mid_chord_point_on_plane = self.mid_chord_point_on_plane.dot(
            rotation_matrix.transpose())

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

        # Save the properties we wanted for each section
        self.pitch_angles_list.append(self.pitch_angle)
        self.pitch_list.append(
            abs(2 * np.pi * radius / np.tan(self.pitch_angle)) / 1000.0)
        self.skew_angles_list.append((self.skew / radius - np.pi) * 180 / np.pi)
        self.skew_list.append((self.skew - np.pi * radius) / 1000.0)
        total_rake = self.rake + self.rake_induced_by_skew
        self.rake = total_rake - (self.skew - np.pi * radius) / np.tan(
            self.pitch_angle)
        self.rake = -self.rake / 1000.0
        self.rake_list.append(self.rake)
        self.chord_length_list.append(self.chord_length / 1000.0)
        self.camber_points_on_plane = np.sort(
            self.camber_points_on_plane.view('float64,float64'),
            order=['f0'],
            axis=0).view(np.float64)
        self.f_camber = interp1d(
            np.squeeze(np.asarray(self.camber_points_on_plane[:, 0])),
            np.squeeze(np.asarray(self.camber_points_on_plane[:, 1])),
            kind='cubic')

    def _airfoil_top_and_bottom_points(self):
        """
        Private method that finds the points of the airfoil belonging to the upper 
        and lower profiles of each section. 
        """
        self.airfoil_top = []
        self.airfoil_top.append(self.leading_edge_point_on_plane)
        self.airfoil_top.append(self.trailing_edge_point_on_plane)
        self.airfoil_bottom = []
        self.airfoil_bottom.append(self.leading_edge_point_on_plane)
        self.airfoil_bottom.append(self.trailing_edge_point_on_plane)
        for i in range(len(self.airfoil_points_on_plane)):
            # points close to the leading and trailing edges are not considered for
            # the construction of the upper and lower profiles of sections
            if ((self.airfoil_points_on_plane[i, 0] > 0.005)
                    and (self.airfoil_points_on_plane[i, 0] < 0.995)):
                if (self.airfoil_points_on_plane[i, 1] > self.f_camber(
                        min(max(self.airfoil_points_on_plane[i, 0], 0.0),
                            1.0))):
                    self.airfoil_top.append([
                        self.airfoil_points_on_plane[i, 0],
                        self.airfoil_points_on_plane[i, 1]
                    ])
                else:
                    self.airfoil_bottom.append([
                        self.airfoil_points_on_plane[i, 0],
                        self.airfoil_points_on_plane[i, 1]
                    ])
        # Storing the upper and lower points in matrices and rescaling in the interval [0,1]
        self.airfoil_top = np.matrix(self.airfoil_top)
        self.airfoil_top = np.sort(self.airfoil_top.view('float64,float64'),
                                   order=['f0'],
                                   axis=0).view(np.float64)
        self.airfoil_bottom = np.matrix(self.airfoil_bottom)
        self.airfoil_bottom = np.sort(
            self.airfoil_bottom.view('float64,float64'), order=['f0'],
            axis=0).view(np.float64)

        self.airfoil_top[0, 0] = 0.0
        self.airfoil_top[-1, 0] = 1.0
        self.airfoil_bottom[0, 0] = 0.0
        self.airfoil_bottom[-1, 0] = 1.0

        self.ascissa = np.linspace(0,
                                   1,
                                   num=self.num_points_top_bottom,
                                   endpoint=True)
        self.f_top = interp1d(np.squeeze(np.asarray(self.airfoil_top[:, 0])),
                              np.squeeze(np.asarray(self.airfoil_top[:, 1])),
                              kind='cubic')
        self.f_bottom = interp1d(
            np.squeeze(np.asarray(self.airfoil_bottom[:, 0])),
            np.squeeze(np.asarray(self.airfoil_bottom[:, 1])),
            kind='cubic')
        # Plot points of camber line, trailing and leading edges, upper and lower profiles
        # of each blade section
        self.int_airfoil_top = self.f_top(self.ascissa)
        self.int_airfoil_top[0] = 0.0
        self.int_airfoil_top[-1] = 0.0
        self.int_airfoil_bottom = self.f_bottom(self.ascissa)
        self.int_airfoil_bottom[0] = 0.0
        self.int_airfoil_bottom[-1] = 0.0

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
            for j in range(self.num_sections):
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
            self.recons_sections[j] = CustomProfile(xup=self.xup[j, :],
                                                    yup=self.yup[j, :],
                                                    xdown=self.xdown[j, :],
                                                    ydown=self.ydown[j, :],
                                                    chord_len=self.chord_length_list[j])
        return self.recons_sections

    def reconstruct_blade(self):
        """
        Method that reconstructs the blade starting from the sections
        computed from the original IGES file, and then reconstruct the 4 parts
        of the blade (upper, lower face, tip, root).
        """

        for j in range(self.num_sections):
            self.recons_sections[j] = CustomProfile(xup=self.xup[j, :],
                                                    yup=self.yup[j, :],
                                                    xdown=self.xdown[j, :],
                                                    ydown=self.ydown[j, :],
                                                    chord_len=self.chord_length_list[j])
        radii = np.array(self.radii_list) / 1000
        self.recons_blade = Blade(sections=self.recons_sections,
                                  radii=radii,
                                  chord_lengths=self.chord_length_list,
                                  pitch=self.pitch_list,
                                  rake=self.rake_list,
                                  skew_angles=self.skew_angles_list)
        self.recons_blade.apply_transformations()
        self.recons_blade.generate_iges(upper_face='rec_uface',
                                        lower_face='rec_lface',
                                        tip='rec_tip',
                                        root='rec_root',
                                        max_deg=1,
                                        display=True,
                                        errors=None)

    def reconstruct_propeller(self, propeller_iges, shaft_iges, n_blades):
        """
        Method which reconstruct the whole propeller with shaft starting from 
        the iges file of the shaft and the number of blades. The whole propeller 
        is saved in an IGES file named filename_iges (must be given as input).
        """
        shaft_path = shaft_iges
        prop_shaft = Shaft(shaft_path)
        prop = Propeller(prop_shaft, self.recons_blade, n_blades)
        prop.generate_iges(propeller_iges)
        prop.display()

    def display_cylinders(self):
        """
        Method that displays the cylinders and the blade, for each section.
        """
        display, start_display, add_menu, add_function_to_menu = init_display()
        for radius in self.radii_list:
            # Display the blade and the wire corresponding to the intersection with the
            # lateral face of the cylinder computed for each section radius
            display.DisplayShape(self.blade_solid, update=True)
            display.DisplayShape(self.wire, update=True)

    def display_sections(self):
        """
        Method that displays the sections profiles.
        """
        display, start_display, add_menu, add_function_to_menu = init_display()
        for radius in self.radii_list:
           
            display.DisplayShape(BRepBuilderAPI_MakeVertex(
                self.leading_edge_point).Vertex(),
                                  update=True,
                                  color="ORANGE")
            display.DisplayShape(BRepBuilderAPI_MakeVertex(
                self.trailing_edge_point).Vertex(),
                                  update=True,
                                  color="GREEN")
            # Display the camber points, leading and trailing edges projected on plane
            display.DisplayShape(self.camber_curve_edge,
                                  update=True,
                                  color="GREEN")
            display.DisplayShape(self.edge_3d, update=True, color="RED")
            display.DisplayShape(self.first_edge, update=True, color="BLUE1")
            display.DisplayShape(self.last_edge, update=True, color="BLUE1")
            display.DisplayShape(self.firstSegment, update=True, color="ORANGE")
            display.DisplayShape(self.lastSegment, update=True, color="ORANGE")
            # Update the blade, cylinder and section when considering a different radius
            display.DisplayShape(self.blade_compound, update=True)
            display.DisplayShape(self.cylinder_lateral_face, update=True)
            display.DisplayShape(self.section, update=True)

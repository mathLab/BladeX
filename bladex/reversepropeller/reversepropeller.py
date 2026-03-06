"""
Module for the extraction of the parameters of a blade and for the
approximated reconstruction of the blade and the whole propeller.
"""

import numpy as np
from bladex import CustomProfile, Blade, Propeller, Shaft
from .basereversepropeller import BaseReversePropeller
from OCC.Core.IGESControl import (IGESControl_Reader)
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex,\
             BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeSolid, \
             BRepBuilderAPI_Sewing, BRepBuilderAPI_NurbsConvert
from OCC.Core.BRep import BRep_Tool
import OCC.Core.TopoDS
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Vec
from OCC.Core.TColgp import  TColgp_Array1OfPnt
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
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
from OCC.Display.SimpleGui import init_display

from scipy.spatial import Voronoi
from scipy.interpolate import interp1d

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
    """
    Test if point (x,y) is inside polygon poly.

    poly is N-vertices polygon defined as
    [(x1,y1),...,(xN,yN)] or [(x1,y1),...,(xN,yN),(x1,y1)]
    (function works fine in both cases)

    Geometrical idea: point is inside polygon if horisontal beam
    to the right from point crosses polygon even number of times.
    Works fine for non-convex polygons.
    """
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


class ReversePropeller(BaseReversePropeller):
    """

    :param filename: path to the IGES file of the blade.
    :param list radii_list: list which contains the radii values of the
        sectional profiles.
    :param num_points_top_bottom: number of points used to interpolate each
        sectional profile.
    """
    def __init__(self, filename, radii_list, num_points_top_bottom):

        super().__init__(filename, radii_list, num_points_top_bottom)
        ind_sec = 0
        for radius in self.radii_list:
            self.cylinder = None
            self.bounds = None
            self.cylinder_lateral_face = None
            self.cylinder_lateral_surface = None
            self.linear_tolerance = 1e-3
            self.conversion_unit = 1000
            self._build_cylinder(radius)
            self._build_intersection_cylinder_blade()
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
            self._camber_points(radius)
            self._voronoi_points(radius)
            self._camber_curve(radius)
            self._initial_leading_trailing_edges_plane(radius)
            self._initial_camber_points_plane(radius)
            self._initial_airfoil_points_plane(radius)
            self._extract_parameters_and_transform_profile(radius)
            self._store_properties(radius)
            self._airfoil_top_and_bottom_points()
            self.xup[ind_sec, :] = self.ascissa
            self.xdown[ind_sec, :] = self.ascissa
            self.yup[ind_sec, :] = self.int_airfoil_top
            self.ydown[ind_sec, :] = self.int_airfoil_bottom
            ind_sec = ind_sec + 1


    # def _build_intersection_cylinder_blade(self):
    #     """
    #     Private method that constructs the section lines which are the intersections
    #     between the cylinder at a fixed radius and the blade, and the camber points.
    #     """
    #     # Construction of the section lines between two shapes (in this case the
    #     # blade and the lateral face of the cylinder)
    #     section_builder = BRepAlgoAPI_Section(self.blade_solid,
    #                                        self.cylinder_lateral_face, False)
    #     # Define and build the parametric 2D curve (pcurve) for the section lines defined above
    #     section_builder.ComputePCurveOn2(True)
    #     section_builder.Build()
    #     self.section = section_builder.Shape()
    #     edgeExplorer = TopExp_Explorer(self.section, TopAbs_EDGE)
    #     wire_maker = BRepBuilderAPI_MakeWire()

    #     edgeCount = 0
    #     edgeList = TopTools_ListOfShape()
    #     while edgeExplorer.More():
    #         edgeCount = edgeCount + 1 # Numbering from 1 in OCC
    #         edge = topods.Edge(edgeExplorer.Current())
    #         edgeList.Append(edge)
    #         edgeExplorer.Next()
    #     wire_maker.Add(edgeList)
    #     self.wire = wire_maker.Wire()
    #     self.section_wires_list.append(self.wire)
    #     self.curve_adaptor = BRepAdaptor_CompCurve(
    #         OCC.Core.TopoDS.topods.Wire(self.wire))
    #     # Length of the curve section (ascissa curvilinea)
    #     self.total_section_length = GCPnts_AbscissaPoint.Length(
    #         self.curve_adaptor)


    def _camber_points(self, radius):
        """
        Private method which computes the single points of the camber curve and
        collects them in param_plane_points and in orig_plane_points.
        """
        firstParam = self.curve_adaptor.FirstParameter()
        for i in range(self.section_points_number):
            rel_distance = float(i) / float(
                self.section_points_number) * self.total_section_length
            point_generator = GCPnts_AbscissaPoint(1e-7, self.curve_adaptor,
                                                   rel_distance, firstParam)
            param = point_generator.Parameter()
            point = self.curve_adaptor.Value(param)
            # Compute the angle for polar reference system
            theta = np.arctan2(point.Y(), point.Z())
            if theta < 0:
                theta += 2.0 * np.pi
            # Param_plane_points is the list of all points parametrized in cylindric coordinates,
            # i.e., the coordinate along X (the axis of the cylinder) and the
            # radius multiplying the polar angle.
            self.orig_param_plane_points.append([point.X(), radius * theta])
            self.param_plane_points.append([point.X(), radius * theta])

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
        # Find in an iterative way the Voronoi points
        for i in range(len(vor.vertices)):
            p = [vor.vertices[i][0], vor.vertices[i][1]]
            if (point_inside_polygon(p[0],
                                     p[1],
                                     self.param_plane_points,
                                     include_edges=True)):
                vor_points.append(p)
        optimized_vor_points = sorted(vor_points, key=lambda x: x[0])
        #vor_us includes the X coordinates and vor_vs the Z coordinates
        self.vor_us = []
        self.vor_vs = []
        for p in optimized_vor_points:
            # coordinates of the points of the Voronoi map
            self.vor_us.append(p[0])
            self.vor_vs.append(p[1])

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
        # Build the camber line from 3D edges
        self.edge_3d = BRepBuilderAPI_MakeEdge(camber_curve).Edge()
        camber_curve_adaptor = BRepAdaptor_Curve(
            OCC.Core.TopoDS.topods.Edge(self.edge_3d))
        # Converting camber edge into wire for plotting purposes
        wire_maker = BRepBuilderAPI_MakeWire()
        wire_maker.Add(self.edge_3d)
        self.camber_wires_list.append(wire_maker.Wire())
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
            OCC.Core.TopoDS.topods.Wire(full_camber_wire))
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

    def _store_properties(self, radius):
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


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


class ReversePropellerBladeX(BaseReversePropeller):
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
            self.param_plane_points = []
            self.orig_param_plane_points = []
            # points = []
            self._camber_curve(radius)
            # for i in range(len(self.vor_us)):
            #     x, y, z = self.vor_us[i], radius*np.sin(self.vor_vs[i]/radius), radius*np.cos(self.vor_vs[i]/radius)
            #     pnt = gp_Pnt(float(x), float(y), float(z))
            #     points.append(BRepBuilderAPI_MakeVertex(pnt).Vertex())
            # self._camber_curve(radius)
            if False:
                display, start_display, add_menu, add_function_to_menu = init_display()
                # display.DisplayShape(self.blade_solid, update=True)
                display.DisplayShape(self.cylinder, update=True)
                display.DisplayShape(self.wire_top, color="RED", update=True)
                display.DisplayShape(self.wire_bottom, color="BLUE", update=True)
                # display.DisplayShape(self.chord_wire, color="ORANGE", update=True)
                display.DisplayShape(self.full_camber_wire, color="GREEN", update=True)
                display.DisplayShape(self.leading_edge, color="BLACK", update=True)
                display.DisplayShape(self.trailing_edge, color="MAGENTA", update=True)
                # display.DisplayShape(self.chord_plane, update=True)
                display.DisplayShape(self.chord_point, update=True)
                display.DisplayShape(self.trimmed_edge, color="BLACK", update=True)
                # display.DisplayShape(points, update=True)
                start_display()
            self._initial_leading_trailing_edges_plane(radius)
            self._initial_camber_points_plane(radius)
            self._initial_airfoil_points_plane(radius)
            self._airfoil_top_and_bottom_points_before_transformations(radius)
            self._extract_parameters_and_transform_profile(radius)
            self._store_properties(radius)
            self._transform_top_and_bottom()
            self._airfoil_top_and_bottom_points_after_transformations()
            self.xup[ind_sec, :] = self.ascissa
            self.xdown[ind_sec, :] = self.ascissa
            self.yup[ind_sec, :] = self.int_airfoil_top
            self.ydown[ind_sec, :] = self.int_airfoil_bottom
            ind_sec = ind_sec + 1


    def _build_intersection_cylinder_blade(self):
        super()._build_intersection_cylinder_blade()
        wire_maker_top = BRepBuilderAPI_MakeWire()
        wire_maker_bottom = BRepBuilderAPI_MakeWire()
        edgeList_top = TopTools_ListOfShape()
        edgeList_bottom = TopTools_ListOfShape()
        edgeCount = 0
        edgeExplorer = TopExp_Explorer(self.section, TopAbs_EDGE)
        while edgeExplorer.More():
            edgeCount = edgeCount + 1 # Numbering from 1 in OCC
            edge = edgeExplorer.Current()
            if edgeCount % 2 == 1:
                edgeList_top.Append(edge)
            else:
                edgeList_bottom.Append(edge)
            edgeExplorer.Next()
        
        # Top part of section
        wire_maker_top.Add(edgeList_top)
        self.wire_top = wire_maker_top.Wire()
        self.curve_adaptor_top = BRepAdaptor_CompCurve(
            OCC.Core.TopoDS.topods.Wire(self.wire_top))
        self.total_section_top_length = GCPnts_AbscissaPoint.Length(
            self.curve_adaptor_top)
        # Bottom part
        wire_maker_bottom.Add(edgeList_bottom)
        self.wire_bottom = wire_maker_bottom.Wire()
        self.curve_adaptor_bottom = BRepAdaptor_CompCurve(
            OCC.Core.TopoDS.topods.Wire(self.wire_bottom))
        self.total_section_bottom_length = GCPnts_AbscissaPoint.Length(
            self.curve_adaptor_bottom)

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
    #     wire_maker = BRepBuilderAPI_MakeWire()
    #     wire_maker_top = BRepBuilderAPI_MakeWire()
    #     wire_maker_bottom = BRepBuilderAPI_MakeWire()

    #     edgeList = TopTools_ListOfShape()
    #     edgeList_top = TopTools_ListOfShape()
    #     edgeList_bottom = TopTools_ListOfShape()
    #     edgeExplorer = TopExp_Explorer(self.section, TopAbs_EDGE)
    #     edgeCount = 0
    #     while edgeExplorer.More():
    #         edgeCount = edgeCount + 1 # Numbering from 1 in OCC
    #         edge = edgeExplorer.Current()
    #         edgeList.Append(edge)
    #         if edgeCount % 2 == 1:
    #             edgeList_top.Append(edge)
    #         else:
    #             edgeList_bottom.Append(edge)
    #         edgeExplorer.Next()
        
    #     # Total sectional curve
    #     wire_maker.Add(edgeList)
    #     self.wire = wire_maker.Wire()
    #     self.section_wires_list.append(self.wire)
    #     self.curve_adaptor = BRepAdaptor_CompCurve(
    #         OCC.Core.TopoDS.topods.Wire(self.wire))
    #     self.total_section_length = GCPnts_AbscissaPoint.Length(
    #         self.curve_adaptor)
    #     # Top part
    #     wire_maker_top.Add(edgeList_top)
    #     self.wire_top = wire_maker_top.Wire()
    #     self.curve_adaptor_top = BRepAdaptor_CompCurve(
    #         OCC.Core.TopoDS.topods.Wire(self.wire_top))
    #     self.total_section_top_length = GCPnts_AbscissaPoint.Length(
    #         self.curve_adaptor_top)
    #     # Bottom part
    #     wire_maker_bottom.Add(edgeList_bottom)
    #     self.wire_bottom = wire_maker_bottom.Wire()
    #     self.curve_adaptor_bottom = BRepAdaptor_CompCurve(
    #         OCC.Core.TopoDS.topods.Wire(self.wire_bottom))
    #     self.total_section_bottom_length = GCPnts_AbscissaPoint.Length(
    #         self.curve_adaptor_bottom)


    def _camber_curve(self, radius):
        """
        Computation of the camber points. We get the chord, move along it and fint the intersection
        of the othogonal plane to the chord-curvilinear absissa and the top-bottom wires
        """
        self.trailing_edge = self.curve_adaptor_top.Value(0.0)
        self.leading_edge = self.curve_adaptor_top.Value(self.curve_adaptor_top.LastParameter())
        # Equation of generic geodesic from leading to trailing edge:
        # x = lambda*s + mu, y = R*sin(a*s+b), z = R*cos(a*s+b)
        # with s parametrization of curve and lambda, mu, a, b to be found
        # for s=0 we habe leading edge and s=1 we have trailing edge
        mu = self.leading_edge.X()
        lam = self.trailing_edge.X() - mu
        b = np.arcsin(self.leading_edge.Y()/radius)
        a = np.arcsin(self.trailing_edge.Y()/radius) - b
        # Thus, the new a chord point is, given any s \in [0,1]
        # gp_Pnt(lam*s + mu, radius * np.sin(a*s+b), radius * np.cos(a*s+b)))

        # Now, for N chord points, we compute the tangent vector to it,
        # define the plane passing to the point and with normal direction equal to the tangent
        # and compute the intersecting curve between cylinder and plane. The midpoint is the camber point
        self.n_camber_points = 500 
        n_plot_point = int (self.n_camber_points/2)
        s_values = np.linspace(0, 1, self.n_camber_points)
        self.camber_Pnts = TColgp_Array1OfPnt(1, len(s_values))
        for i, s in enumerate(s_values):
            if i == 0:
                self.camber_Pnts.SetValue(i+1, self.leading_edge)           
            elif i == len(s_values)-1:
                self.camber_Pnts.SetValue(i+1, self.trailing_edge) 
            else:
                chord_point = gp_Pnt(lam*s + mu, radius * np.sin(a*s+b), radius * np.cos(a*s+b))
                # Equation of tangent vector is: x = lam, y = a*R*cos(a*s+b), z = -a*R*sin(a*s+b)
                tangent_vector = gp_Vec(lam, a*radius*np.cos(a*s+b), -a*radius*np.sin(a*s+b))
                chord_plane = gp_Pln(chord_point, gp_Dir(tangent_vector))   
                chord_plane_face = BRepBuilderAPI_MakeFace(chord_plane, -1e8, 1e8, -1e8, 1e8).Face()
                if i == n_plot_point:
                    self.chord_plane = chord_plane_face
                    # self.chord_plane = chord_plane
                    self.chord_point = chord_point
                # Now we intersect the plane with the cylinder
                section_pc = BRepAlgoAPI_Section(self.cylinder_lateral_face, chord_plane_face)
                section_pc.Build()
                curve_shape = section_pc.Shape()
                exp = TopExp_Explorer(curve_shape, TopAbs_EDGE)
                curve_edge = exp.Current()
                # Find the intersection of the curve with top and bottom faces of the blade's section
                int_top = BRepAlgoAPI_Section(curve_edge, self.wire_top)
                int_top.Build()
                int_bottom = BRepAlgoAPI_Section(curve_edge, self.wire_bottom)
                int_bottom.Build()
                exp = TopExp_Explorer(int_top.Shape(), TopAbs_VERTEX)
                from OCC.Core.TopoDS import TopoDS_Vertex
                while exp.More():
                    point_top = exp.Current()
                    if isinstance(int_top, TopoDS_Vertex):
                        break
                    exp.Next()
                exp = TopExp_Explorer(int_bottom.Shape(), TopAbs_VERTEX)
                while exp.More():
                    point_bottom= exp.Current()
                    if isinstance(int_bottom, TopoDS_Vertex):
                        break
                    exp.Next()
                # Now we trim the curve to only have inside the two intersection points
                curve_handle, u1, u2 = BRep_Tool.Curve(curve_edge)
                from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
                proj = GeomAPI_ProjectPointOnCurve(BRep_Tool.Pnt(point_top), curve_handle)
                u_top = proj.LowerDistanceParameter()
                proj = GeomAPI_ProjectPointOnCurve(BRep_Tool.Pnt(point_bottom), curve_handle)
                u_bot = proj.LowerDistanceParameter()
                u_min = min(u_top, u_bot)
                u_max = max(u_top, u_bot)
                trimmed_edge = BRepBuilderAPI_MakeEdge(curve_handle, u_min, u_max).Edge()
                if i == n_plot_point:
                    wire_maker = BRepBuilderAPI_MakeWire()
                    wire_maker.Add(trimmed_edge)
                    self.trimmed_edge= wire_maker.Wire()
                trimmed_adaptor = BRepAdaptor_Curve(trimmed_edge)
                u0 = trimmed_adaptor.FirstParameter()
                u1 = trimmed_adaptor.LastParameter()
                total_length = GCPnts_AbscissaPoint.Length(trimmed_adaptor, u0, u1)
                absc = GCPnts_AbscissaPoint(trimmed_adaptor, 0.5*total_length, u0)
                u_mid = absc.Parameter()
                self.camber_Pnts.SetValue(i+1, trimmed_adaptor.Value(u_mid))           
        # Building an edge from points
        spline_builder = GeomAPI_PointsToBSpline(self.camber_Pnts, 3, 3, GeomAbs_C2, 1e-1)
        camber_curve = spline_builder.Curve()
        # Build the camber line from 3D edges
        self.edge_3d = BRepBuilderAPI_MakeEdge(camber_curve).Edge()
        # Building wire from edge
        wire_maker = BRepBuilderAPI_MakeWire()
        wire_maker.Add(self.edge_3d)

        self.full_camber_wire = wire_maker.Wire()
        self.full_camber_curve_adaptor = BRepAdaptor_CompCurve(
            OCC.Core.TopoDS.topods.Wire(self.full_camber_wire))
        self.full_camber_length = GCPnts_AbscissaPoint.Length(
            self.full_camber_curve_adaptor)
        self.camber_wires_list.append(self.full_camber_wire)

    def _initial_leading_trailing_edges_plane(self, radius):
        """
        Private method which computes the coordinates of the leading and trailing
        edges of each section, as were in the initial blade. Thenm transformations
        will be applied in order to plot the points of the profiles on a plane
        and to map the X coordinates in the interval [0,1].
        """

        self.trailing_edge_point = self.trailing_edge
        self.leading_edge_point = self.leading_edge
        leading_edge_theta = np.arctan2(self.leading_edge_point.Y(),
                                        self.leading_edge_point.Z())
        self.leading_edge_point_on_plane = np.array(
            [self.leading_edge_point.X(), radius * leading_edge_theta])

        trailing_edge_theta = np.arctan2(self.trailing_edge_point.Y(),
                                         self.trailing_edge_point.Z())
        self.trailing_edge_point_on_plane = np.array(
            [self.trailing_edge_point.X(), radius * trailing_edge_theta])

    def _initial_camber_points_plane(self, radius):
        """
        Private method that defines the points of the camber line projected on
        plane, in 2D coordinates.
        """
        self.camber_points_on_plane = np.zeros((self.num_points_top_bottom, 2))

        for i in range(self.num_points_top_bottom):
            point = self.camber_Pnts.Value(i+1)
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

    def _airfoil_top_and_bottom_points_before_transformations(self, radius):
        """
        Private method that finds the points of the airfoil belonging to the upper
        and lower profiles of each section.
        """
        self.airfoil_top = []
        u0_top = self.curve_adaptor_top.FirstParameter()
        u1_top = self.curve_adaptor_top.LastParameter()
        total_length = GCPnts_AbscissaPoint.Length(self.curve_adaptor_top, u0_top, u1_top)
        for i in range(self.num_points_top_bottom):
            point_generator = GCPnts_AbscissaPoint(1e-7, self.curve_adaptor_top,
                                                   float(i)/(self.num_points_top_bottom-1)*total_length, u0_top)
            param = point_generator.Parameter()
            point = self.curve_adaptor_top.Value(param)
            theta = np.arctan2(point.Y(), point.Z())
            self.airfoil_top.append([point.X(), radius*theta])
        self.airfoil_bottom = []
        u0_bottom = self.curve_adaptor_bottom.FirstParameter()
        u1_bottom = self.curve_adaptor_bottom.LastParameter()
        total_length = GCPnts_AbscissaPoint.Length(self.curve_adaptor_bottom, u0_bottom, u1_bottom)
        for i in range(self.num_points_top_bottom):
            point_generator = GCPnts_AbscissaPoint(1e-7, self.curve_adaptor_bottom,
                                                   float(i)/(self.num_points_top_bottom-1)*total_length, u0_bottom)
            param = point_generator.Parameter()
            point = self.curve_adaptor_bottom.Value(param)
            theta = np.arctan2(point.Y(), point.Z())
            self.airfoil_bottom.append([point.X(), radius*theta])

        self.airfoil_top = np.array(self.airfoil_top)
        self.airfoil_bottom = np.array(self.airfoil_bottom)

    def _store_properties(self, radius):
        # Save the properties we wanted for each section
        self.pitch_angles_list.append(self.pitch_angle)
        self.pitch_list.append(
            abs(2 * np.pi * radius / np.tan(self.pitch_angle)) / 1000.0)
        self.skew_angles_list.append(-(self.skew / radius) * 180 / np.pi)
        self.skew_list.append(self.skew / 1000.0)
        total_rake = self.rake + self.rake_induced_by_skew
        rake = total_rake - (self.skew) / np.tan(
            self.pitch_angle)
        rake = -rake / 1000.0
        self.rake_list.append(rake)
        self.chord_length_list.append(self.chord_length / 1000.0)

    def _transform_top_and_bottom(self):
        """
        Method that transforms the top and bottom coordinates from physical coordinates
        to 2D planar with chord aligned to x-axis
        skew, pitch, rake etc. have been already compute in _extract_parameters_and_transform_profile()
        """
        self.airfoil_top[:, 1] -= self.skew
        self.airfoil_bottom[:, 1] -= self.skew

        self.airfoil_top[:, 0] -= self.rake_induced_by_skew
        self.airfoil_bottom[:, 0] -= self.rake_induced_by_skew

        self.airfoil_top[:, 0] -= self.rake
        self.airfoil_bottom[:, 0] -= self.rake

        self.airfoil_top = self.airfoil_top.dot(
            self.rotation_matrix.transpose())
        self.airfoil_bottom = self.airfoil_bottom.dot(
            self.rotation_matrix.transpose())

        self.airfoil_top /= self.chord_length
        self.airfoil_bottom /= self.chord_length

        self.airfoil_top[:, 0] *= -1.0
        self.airfoil_bottom[:, 0] *= -1.0

        self.airfoil_top[:, 0] += 0.5
        self.airfoil_bottom[:, 0] += 0.5

        # Final check on y components to see which one is really top
        if self.airfoil_top[np.abs(self.airfoil_top[:, 1]).argmax(), 1] < 0:
            self.airfoil_top[:, 1] *= -1
            self.airfoil_bottom[:, 1] *= -1

    def _airfoil_top_and_bottom_points_after_transformations(self):
        # Now we create two interpolating functions fitting bottom e up points
        # In this way we can retrieve as many bottom/up points as we want
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


"""
Module for the blade bottom-up parametrized construction.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_HArray1OfPnt
from OCC.Core.GeomAPI import GeomAPI_Interpolate
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex,\
        BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire,\
        BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid

class InterpolatedFace:

    def __init__(self, pts, max_deg=3, tolerance=1e-10):

        print(pts.shape)
        if pts.ndim not in [2, 3]:
            raise ValueError("pts must be a 2D or 3D array.")
        
        if pts.ndim == 2:
            pts = pts[None, :, :]

        if pts.shape[1] != 3:
            raise ValueError("Each point must have  3 coordinates for X, Y, Z.")

        self.max_deg = max_deg
        self.tolerance = tolerance

        generator = BRepOffsetAPI_ThruSections(False, False, tolerance)
        generator.SetMaxDegree(max_deg)

        for id_section, section in enumerate(pts):
            vertices = TColgp_HArray1OfPnt(1, section.shape[1])
            for id_pt, pt in enumerate(section.T, start=1):
                vertices.SetValue(id_pt, gp_Pnt(*pt))

            # Initializes an algorithm for constructing a constrained
            # BSpline curve passing through the points of the blade last
            # section
            bspline = GeomAPI_Interpolate(vertices, False, tolerance)
            bspline.Perform()

            edge = BRepBuilderAPI_MakeEdge(bspline.Curve()).Edge()

            if id_section == 0:
                root_edge = edge

            # Add BSpline wire to the generator constructor
            generator.AddWire(BRepBuilderAPI_MakeWire(edge).Wire())

        generator.Build()
        self.face = generator.GeneratedFace(root_edge)
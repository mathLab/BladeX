"""
Shared CAD export utilities.

Everything needed to turn OCC faces into IGES/STL files:
compounds, sewing, solid assembly, tessellation and file writing. Keeping
these operations in one place avoids each class reinventing its own variant.
"""
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeSolid,
                                     BRepBuilderAPI_Sewing)
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.IGESControl import IGESControl_Writer
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.TopAbs import TopAbs_COMPOUND, TopAbs_SHELL
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Iterator, topods


def make_compound(shapes):
    """Group a list of shapes into a single TopoDS_Compound."""
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)

    for shape in shapes:
        if shape is None:
            raise ValueError("make_compound: shape is None")
        builder.Add(comp, shape)

    return comp


def sew_shapes(shapes, tolerance=1e-2):
    """Sew a list of faces/shapes into a single shape (shell or compound)."""
    sewer = BRepBuilderAPI_Sewing(tolerance)
    for shape in shapes:
        if shape is None:
            raise ValueError(
                "sew_shapes: shape is None (call Blade.build first)")
        sewer.Add(shape)
    sewer.Perform()
    return sewer.SewedShape()


def to_solid(shape):
    """Convert `shape` to a solid when it contains exactly one shell.

    If the sewing did not produce a single shell (degenerate geometry, open
    edges, ...) the input shape is returned unchanged: IGES and STL accept it,
    while the forced cast to TopoDS_Shell used before raised
    Standard_TypeMismatch and made the export fail.
    """
    shell = _single_shell(shape)
    if shell is None:
        return shape

    maker = BRepBuilderAPI_MakeSolid()
    maker.Add(shell)
    if not maker.IsDone():
        return shape
    return maker.Solid()


def _single_shell(shape):
    """Return the shell of `shape` when there is exactly one, None otherwise.

    Compounds are inspected recursively, so a compound wrapping a compound
    that wraps a single shell still resolves to that shell. Any other
    sub-shape (a face, an edge, a second body) makes the shape ineligible, so
    that nothing is silently dropped when converting to a solid.
    """
    if shape.ShapeType() == TopAbs_SHELL:
        return topods.Shell(shape)

    if shape.ShapeType() != TopAbs_COMPOUND:
        return None

    shells = []
    iterator = TopoDS_Iterator(shape)
    while iterator.More():
        child = iterator.Value()
        if child.ShapeType() == TopAbs_SHELL:
            shells.append(topods.Shell(child))
        elif child.ShapeType() == TopAbs_COMPOUND:
            nested = _single_shell(child)
            if nested is None:
                return None
            shells.append(nested)
        else:
            return None
        iterator.Next()
    return shells[0] if len(shells) == 1 else None


def mesh(shape,
         linear_deflection=0.05,
         angular_deflection=0.3,
         is_relative=True):
    """Tessellate `shape` with the incremental mesh algorithm."""
    triangulation = BRepMesh_IncrementalMesh(shape, linear_deflection,
                                             is_relative, angular_deflection,
                                             True)
    triangulation.Perform()


def write_stl(shape,
              filename,
              linear_deflection=0.05,
              angular_deflection=0.3,
              is_relative=True):
    """Write a binary STL file and check the writer status."""
    mesh(shape, linear_deflection, angular_deflection, is_relative)

    writer = StlAPI_Writer()
    writer.SetASCIIMode(False)
    if not writer.Write(shape, filename):
        raise RuntimeError("Unsuccessful writing of STL file: " + filename)


def write_iges(shape, filename):
    """Write an IGES file and check the writer status."""
    writer = IGESControl_Writer()
    writer.AddShape(shape)
    if not writer.Write(filename):
        raise RuntimeError("Unsuccessful writing of IGES file: " + filename)

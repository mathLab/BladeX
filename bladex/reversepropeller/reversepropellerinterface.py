"""
Interface module that provides essential tools for the reversepropeller.
"""

from abc import ABC, abstractmethod


class ReversePropellerInterface(ABC):
    """
    Interface for reverse problem propeller blade.

    Given an iges file that has the blade's geometry, we solve an inverse problem
    to retrieve the properties of the blades both in terms of: 
        - sectional properties (chord length, thickness, camber);
        - sections location in the space (skew angle, pitch and rake)
    """
    @abstractmethod
    def _extract_solid_from_file(self):
        """
        Abstract method that extract the solid object from the IGES file
        """
        pass

    @abstractmethod
    def _build_cylinder(self, radius):
        """
        Abstract method that, given a radius, it builds the cylinder with specified radius
        and main axis along x direction. This cylinder will be intersected with the blade
        """
        pass

    @abstractmethod
    def _build_intersection_cylinder_blade(self):
        """
        Abstract method that computes the intersection between the blade and the cylinder
        created by _build_cylinder() method
        """
        pass
    
    @abstractmethod
    def _camber_curve(self, radius):
        """
        Abstract method that retrieves the camber curve once the intersection is computed
        """
        pass

    @abstractmethod
    def save_global_parameters(self, filename_csv):
        """
        Abstract method to store the retrieved properties and section into a csv file
        """


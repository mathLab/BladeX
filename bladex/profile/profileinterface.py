"""
Interface module that provides essential tools and transformations on airfoils.
"""

from abc import ABC, abstractmethod


class ProfileInterface(ABC):
    """
    Interface for profile of the propeller blade.

    Each sectional profile is a 2D airfoil that is split into two parts: the
    upper and lower parts. The coordinates of each part is represented by two
    arrays corresponding to the X and Y components in the 2D coordinate system.
    Such coordinates can be either generated using NACA functions, or be
    inserted directly by the user as custom profiles.
    """
    @abstractmethod
    def generate_parameters(self, convention='british'):
        """
        Abstract method that generates the airfoil parameters based on the
        given coordinates.

        The method generates the airfoil's chord length, maximum camber and
        maximum thickness, along with their corresponding percentages. The
        method is called automatically when the airfoil coordinates are
        inserted by the user.

        """
        pass

    @abstractmethod
    def generate_coordinates(self):
        """
        Abstract method that generates the airfoil coordinates based on the
        given parameters.

        The method generates the airfoil's upper and lower surfaces
        coordinates. The method is called automatically when the airfoil
        parameters are inserted by the user.

        """
        pass

    @abstractmethod
    def plot(self):
        """
        Abstract method that plots the airfoil coordinates.

        The method creates a 2D plot of the airfoil's upper and lower
        surfaces using Matplotlib.

        """
        pass
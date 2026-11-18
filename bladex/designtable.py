import pandas as pd
import numpy as np

class DesignTable:
    """Class representing the Design Table"""

    def __init__(self, df: pd.DataFrame, skew_unit: str = 'deg', propeller_radius: float = None):
            """
            args:
            df (pd.DataFrame): design table with adimensional radii
            skew_unit (str): unit of the skew column in df, either 'm' or 'deg'
            propeller_radius (float): propeller radius R in meters. Required if skew_unit == 'deg'
            """
            self.admissible_names = ['chord', 'skew', 'pitch', 'rake', 'thickness', 'camber']
            self.propeller_radius = propeller_radius
            self.table = df.copy()
            if 'skew' in self.table.columns:
                self.table['skew'] = self._skew_to_meters(skew_unit)

    @classmethod
    def from_csv(cls, csv: str, skew_unit: str = 'deg', propeller_radius: float = None):
            """Initialize class from csv file. Skew unit can be 'deg' or 'm'."""
            df = pd.read_csv(csv)
            return cls(df, skew_unit, propeller_radius)

    def _skew_to_degrees(self, unit):
        """Converts skew values from meters to degrees"""
        skew = np.asarray(self.table['skew'], dtype=float)
        if unit == 'deg':
            return skew
        if unit == 'm':
            if self.propeller_radius is None:
                raise ValueError("propeller_radius is required to convert skew from meters to degrees.")
            r_dim = np.asarray(self.table['radii'], dtype=float) * self.propeller_radius
            return 180 * skew * np.cos(self.table['pitch']) / (r_dim * np.pi)
        raise ValueError(f"Unknown skew unit '{unit}'. Use 'm' or 'deg'.")

    def radii(self):
        """Returns the radii of the design table"""
        return self.table['radii'].tolist()

    def properties(self):
        """Returns the properties from the design table. Radii are not included"""
        return self.table.drop(columns=["radii"])

    def add_property(self, property_name, property):
        """Adds a property to the design table.
        
        args:
            property_name (str): the name of the property
            property (array_like): the property values. Must have same size of radii
        """
        if property_name not in self.admissible_names:
            raise ValueError('The property name is not among the admissible ones. ' \
                             'The admissible names are: {}'.format(self.admissible_names))
        if property_name in self.properties().keys():
            raise ValueError("This property is already in the design table. " \
                             "If you want to modify it, use the change_property() method.")
        self.table[property_name] = property

    def change_property(self, property_name, property):
        """Changes a property in the design table.
        
        args:
            property_name (str): the name of the property
            property (array_like): the property values. Must have same size of radii
        """
        if property_name not in self.admissible_names:
            raise ValueError('The property name is not among the admissible ones. ' \
                             'The admissible names are: {}'.format(self.admissible_names))
        if property_name not in self.properties().keys():
            raise ValueError("This property is not in the design table. " \
                             "If you want to add it, use the add_property() method.")
        self.table[property_name] = property

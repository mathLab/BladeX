import numpy as np
from bladex import Blade, CustomProfile, NacaProfile

# Create sections with NACAProfile
sections = np.asarray([NacaProfile(digits='4412', n_points=50) for i in range(10)])

# Create sections with CustomProfile extracting coordinates from NACAProfile
sections_custom = []

for sec in sections:
    sec_custom = CustomProfile(xup=sec.xup_coordinates,
        yup=sec.yup_coordinates,
        xdown=sec.xdown_coordinates,
        ydown=sec.ydown_coordinates)
    sections_custom.append(sec_custom)
    sec_custom.generate_parameters(convention='american')

sections_custom = np.asarray(sections_custom)

# Define blade parameters
radii = np.arange(1.0, 11.0, 1.0)
chord_lengths = np.array([0.05, 2.5, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
pitch = np.arange(1., 11.)
rake = np.arange(0.1, 1.1, 0.1)
skew_angles = np.arange(1., 21, 2.)

# Create blade with custom sections
blade = Blade(sections=sections_custom,
        radii=radii,
        chord_lengths=chord_lengths,
        pitch=pitch,
        rake=rake,
        skew_angles=skew_angles)

# Tranform coordinates from planar to cylindrical coordinates
blade.apply_transformations(reflect=True)

blade.generate_iges(upper_face = 'up', lower_face = 'low', tip = 'tip', root =        'root', display = True)

# deformation of parameters
def_pitch = 1
def_chord_len = 1.3
def_thickness = 2
def_camber = 1.7

# Generate a deformed blade deforming parameters
sections_deformed = []

for sec in sections_custom:
    sec_deformed = CustomProfile(chord_perc=sec.chord_percentage,
#            chord_len=sec.chord_len,
            thickness_max=sec.thickness_max*def_thickness,
            camber_max=sec.camber_max*def_camber,
            thickness_perc=sec.thickness_percentage,
            camber_perc=sec.camber_percentage)

    sections_deformed.append(sec_deformed)
    sec_deformed.generate_coordinates(convention='american')

blade_deformed = Blade(sections=sections_deformed,
        radii=radii,
        chord_lengths=chord_lengths*def_chord_len,
        pitch=pitch*def_pitch,
        rake=rake,
        skew_angles=skew_angles)

blade_deformed.apply_transformations(reflect=True)
blade_deformed.generate_iges(upper_face = 'up_def', lower_face = 'low_def', tip = 'tip_def', root = 'root_def', display = True)



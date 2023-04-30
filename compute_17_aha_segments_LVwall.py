""" Compute 17-AHA segmentation of a given LV wall mask.
The output is the LV wall volume with the corresponding 17-AHA labels. The image can then be probed to get values on meshes etc.
If dilate_wall = True, compute also a dilated version of the wall to help 'probing' LV endo or LV epi (extremest) meshes

Use official definition of the 17-AHA model described here: https://www.pmod.com/files/download/v34/doc/pcardp/3615.html
17-Segment Model (AHA). Left Ventricle Segmentation Procedure:
    The left ventricle is divided into equal thirds perpendicular to the long axis of the heart. This generates three
    circular sections of the left ventricle named basal, mid-cavity, and apical.
    Only slices containing myocardium in all 360' are included.
    - The basal part is divided into six segments of 60' each. (...)
    - Similarly the mid-cavity part is divided into six 60' segments (...)
    - Only four segments of 90' each are used for the apex because of the myocardial tapering.
    - The apical cap represents the true muscle at the extreme tip of the ventricle where there is no longer
    cavity present. This segment is called the apex.

Usage example:
python compute_17_aha_segments_LVwall.py --path example_pat0/ --mask_lvendo_sax ct-lvendo-sax.mha --mask_lvwall_sax ct-lvwall-sax.mha --mask_rvepi_sax ct-rvepi-sax.mha --mask_lvepi_sax ct-lvepi-sax.mha
"""

from aux_functions import *
import numpy as np
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, metavar='PATH', help='Data path')
parser.add_argument('--mask_lvendo_sax', type=str, help='Input LV endo mask name')
parser.add_argument('--mask_lvwall_sax', type=str, help='Input LV wall mask name')
parser.add_argument('--mask_rvepi_sax', type=str, help='Input RV epi mask name')
parser.add_argument('--mask_lvepi_sax', type=str, help='Input LV epi mask name')
args = parser.parse_args()


dilate_wall = True       # set to False if you don't plan to probe a mesh to get AHA labels on its surface. Faster.

t = time.time()

# inputs
lvendo_sax_filename = args.path + args.mask_lvendo_sax
lvwall_sax_filename = args.path + args.mask_lvwall_sax
rvepi_sax_filename = args.path + args.mask_rvepi_sax
lvepi_sax_filename = args.path + args.mask_lvepi_sax

# outputs
lvwall_aha_filename = args.path + args.mask_lvwall_sax[0:-4] + '-aha.mha'
if dilate_wall:
    dilate_mm = 1.0     # dilate LV wall to help the probe filter in obtaining the correct labels in the mesh
    lvwall_aha_dil_filename = args.path + args.mask_lvwall_sax[0:-4] + '-dil-aha.mha'

lvendo_mask = sitk.ReadImage(lvendo_sax_filename)
lvepi_mask = sitk.ReadImage(lvepi_sax_filename)
rvepi_mask = sitk.ReadImage(rvepi_sax_filename)
lvwall_mask = sitk.ReadImage(lvwall_sax_filename)

patient_name = get_patientname(lvendo_mask)

if dilate_wall:
    # Dilate a bit the LV wall to help the projection of aha labels using the probe filter
    dilate = sitk.BinaryDilateImageFilter()
    spacing = lvepi_mask.GetSpacing()
    dilate_voxels = int(np.round(np.divide(dilate_mm, np.array(spacing[0]))))
    print('Dilating wall', dilate_voxels, 'voxels, (', dilate_mm, 'mm)')
    dilate.SetKernelRadius(1)
    dilate.SetKernelRadius(dilate_voxels)
    wall_dilated = dilate.Execute(lvwall_mask)
    np_lvwall_dil = sitk.GetArrayFromImage(wall_dilated)

np_lvepi = sitk.GetArrayFromImage(lvepi_mask)
np_lvwall = sitk.GetArrayFromImage(lvwall_mask)
np_lvendo = sitk.GetArrayFromImage(lvendo_mask)

# Find LV extension in the z axis
z_extension_wall = np.unique(np.where(np_lvwall == 1)[0])  # z is [0]
z_extension_endo = np.unique(np.where(np_lvendo == 1)[0])
# long_axis_span = np.max(z_extension_wall) - np.min(z_extension_wall)
# find z-slice index corresponding to the middle of LV wall mask
mid_lv_long_axis = int(np.round(np.divide(np.max(z_extension_wall) + np.min(z_extension_wall), 2)))
# print('Middle point', mid_lv_long_axis)

# Apex
np_apex = np.zeros(np_lvwall.shape)
for z_slice in range(mid_lv_long_axis, np_lvwall.shape[0]):    # check only from middle point to avoid region with wall but not endo in the base
    # print('Slice', z_slice)
    wall_slice = np_lvwall[z_slice, :, :]
    endo_slice = np_lvendo[z_slice, :, :]
    if np.where(wall_slice == 1)[0].shape[0] > 0:   # there is wall
        # if np.where(endo_slice == 1)[0].shape[0] == 0:   # there is NO endo, i.e. cavity
        if np.where(endo_slice == 1)[0].shape[0] < 10:   # allow small number of voxels, 0 may be too strict
            # print('Apex slice', z_slice)
            np_apex[z_slice, :, :] = 1

if len(np.where(np_apex == 1)[0]) == 0:
    print('No apex found according to endo and epi meshes')
    first_apex_slice = np.max(z_extension_wall) - 5      # manually mark last 5 slices as apex
    np_apex[first_apex_slice: np.max(z_extension_wall), :, :] = 1
else:
    first_apex_slice = np.min(np.where(np_apex == 1)[0])

np_apex[np.where(np_lvwall == 0)] = 0  # bg still bg


# ONLY SLICES CONTAINING MYOCARDIUM IN ALL 360' ARE INCLUDED
np_usable_wall = np.zeros(np_lvwall.shape)

for z_slice in range(np.min(z_extension_wall), first_apex_slice):
    wall_slice = np_lvwall[z_slice, :, :]
    all_y, all_x = np.where(wall_slice == 1)
    # use slice dependent center, just to check the 360
    slice_center_x = np.round(np.divide(np.max(all_x) + np.min(all_x), 2))
    slice_center_y = np.round(np.divide(np.max(all_y) + np.min(all_y), 2))
    # r, theta = cartesian_to_polar(all_x - center_x, all_y - center_y)
    r, theta = cartesian_to_polar(all_x - slice_center_x, all_y - slice_center_y)

    if check_360(thetas=theta, tolerance=0.10):
        np_usable_wall[z_slice, :, :] = 1

# Create longitudinal divisions
min_z_usable_wall = np.min(np.where(np_usable_wall == 1)[0])
max_z_usable_wall = np.max(np.where(np_usable_wall == 1)[0])
bin_width = int(np.round(np.divide(max_z_usable_wall - min_z_usable_wall, 3)))
long_bins = np.arange(min_z_usable_wall, max_z_usable_wall, bin_width)

np_longitudinal = np.zeros(np_lvwall.shape)
np_longitudinal[long_bins[0]:long_bins[1], :, :] = 1
np_longitudinal[long_bins[1]:long_bins[2], :, :] = 2
np_longitudinal[long_bins[2]:max_z_usable_wall, :, :] = 3
np_longitudinal[max_z_usable_wall:np_lvwall.shape[0], :, :] = 4  # full z, I will then correct with the LV wall mask

# Create circunferential division
np_circunf6 = np.zeros(np_lvwall.shape)
np_circunf4 = np.zeros(np_lvwall.shape)
bin_theta_width6 = np.divide(2 * np.pi, 6)    # 60º
bin_theta_width4 = np.divide(2 * np.pi, 4)    # 90º
bins_theta6 = np.arange(-np.pi, np.pi + bin_theta_width6, bin_theta_width6)  # [-pi, pi]
bins_theta4 = np.arange(-np.pi, np.pi + bin_theta_width4, bin_theta_width4) + np.pi/4    # phase difference

# only 2 different centroids to do not see jumps when changing longitudinal region
# 1. Section where longitudinal = 1 and 2
np_long12 = np.zeros_like(np_longitudinal)
np_long12[np.where(np_longitudinal == 1)] = 1
np_long12[np.where(np_longitudinal == 2)] = 1
x_extension = np.unique(np.where((np_long12 == 1) & (np_lvepi == 1))[2])    # x real...
y_extension = np.unique(np.where((np_long12 == 1) & (np_lvepi == 1))[1])    # y
center_x = np.round(np.divide(np.max(x_extension) + np.min(x_extension), 2))
center_y = np.round(np.divide(np.max(y_extension) + np.min(y_extension), 2))
x_mat = np.array([np.arange(512) - center_x, ] * 512)  # center matrix subtracting center_x
y_mat = np.array([np.arange(512) - center_y, ] * 512).transpose()
r, theta = cartesian_to_polar(x_mat, y_mat)

np_circunf6[np.where((np_long12 == 1) & (theta >= bins_theta6[0]) & (theta <= bins_theta6[1]))] = 1    # section labels start at 1
np_circunf6[np.where((np_long12 == 1) & (theta >= bins_theta6[1]) & (theta <= bins_theta6[2]))] = 2
np_circunf6[np.where((np_long12 == 1) & (theta >= bins_theta6[2]) & (theta <= bins_theta6[3]))] = 3
np_circunf6[np.where((np_long12 == 1) & (theta >= bins_theta6[3]) & (theta <= bins_theta6[4]))] = 4
np_circunf6[np.where((np_long12 == 1) & (theta >= bins_theta6[4]) & (theta <= bins_theta6[5]))] = 5
np_circunf6[np.where((np_long12 == 1) & (theta >= bins_theta6[5]) & (theta <= 4))] = 6

# # 2. Section where longitudinal = 3
# x_extension = np.unique(np.where((np_longitudinal == 3) & (np_lvepi == 1))[2])    # x real...
# y_extension = np.unique(np.where((np_longitudinal == 3) & (np_lvepi == 1))[1])    # y
# center_x = np.round(np.divide(np.max(x_extension) + np.min(x_extension), 2))
# center_y = np.round(np.divide(np.max(y_extension) + np.min(y_extension), 2))
# x_mat = np.array([np.arange(512) - center_x, ] * 512)  # center matrix subtracting center_x
# y_mat = np.array([np.arange(512) - center_y, ] * 512).transpose()
# r, theta = cartesian_to_polar(x_mat, y_mat)
#
# np_circunf4[np.where((np_longitudinal == 3) & (theta >= bins_theta4[0]) & (theta <= bins_theta4[1]))] = 1
# np_circunf4[np.where((np_longitudinal == 3) & (theta >= bins_theta4[1]) & (theta <= bins_theta4[2]))] = 2
# np_circunf4[np.where((np_longitudinal == 3) & (theta >= bins_theta4[2]) & (theta <= bins_theta4[3]))] = 3
# np_circunf4[np.where((np_longitudinal == 3) & (theta >= bins_theta4[3]) & (theta <= 4))] = 4   # pi discontinuity, continue in the next line
# np_circunf4[np.where((np_longitudinal == 3) & (theta >= -np.pi) & (theta <= -np.pi + np.pi/4))] = 4


# Use Apex as axis center for theta
x_extension = np.unique(np.where((np_lvepi == 1) & (np_apex == 1))[2])
y_extension = np.unique(np.where((np_lvepi == 1) & (np_apex == 1))[1])
center_x = np.round(np.divide(np.max(x_extension) + np.min(x_extension), 2))
center_y = np.round(np.divide(np.max(y_extension) + np.min(y_extension), 2))
x_mat = np.array([np.arange(512) - center_x, ] * 512)  # center matrix subtracting center_x
y_mat = np.array([np.arange(512) - center_y, ] * 512).transpose()
r, theta = cartesian_to_polar(x_mat, y_mat)

np_circunf4[np.where((np_longitudinal == 3) & (theta >= bins_theta4[0]) & (theta <= bins_theta4[1]))] = 1    # section labels start at 1
np_circunf4[np.where((np_longitudinal == 3) & (theta > bins_theta4[1]) & (theta <= bins_theta4[2]))] = 2
np_circunf4[np.where((np_longitudinal == 3) & (theta > bins_theta4[2]) & (theta <= bins_theta4[3]))] = 3
np_circunf4[np.where((np_longitudinal == 3) & (theta > bins_theta4[3]) & (theta <= bins_theta4[4]))] = 4
np_circunf4[np.where((np_longitudinal == 3) & (theta > -np.pi) & (theta < bins_theta4[0]))] = 4


# create 17-aha
np_aha = np.zeros(np_lvwall.shape)
np_aha[np.where((np_longitudinal == 1) & (np_circunf6 == 1))] = 3
np_aha[np.where((np_longitudinal == 1) & (np_circunf6 == 2))] = 4
np_aha[np.where((np_longitudinal == 1) & (np_circunf6 == 3))] = 5
np_aha[np.where((np_longitudinal == 1) & (np_circunf6 == 4))] = 6
np_aha[np.where((np_longitudinal == 1) & (np_circunf6 == 5))] = 1
np_aha[np.where((np_longitudinal == 1) & (np_circunf6 == 6))] = 2
np_aha[np.where((np_longitudinal == 2) & (np_circunf6 == 1))] = 9
np_aha[np.where((np_longitudinal == 2) & (np_circunf6 == 2))] = 10
np_aha[np.where((np_longitudinal == 2) & (np_circunf6 == 3))] = 11
np_aha[np.where((np_longitudinal == 2) & (np_circunf6 == 4))] = 12
np_aha[np.where((np_longitudinal == 2) & (np_circunf6 == 5))] = 7
np_aha[np.where((np_longitudinal == 2) & (np_circunf6 == 6))] = 8
np_aha[np.where((np_longitudinal == 3) & (np_circunf4 == 1))] = 15
np_aha[np.where((np_longitudinal == 3) & (np_circunf4 == 2))] = 16
np_aha[np.where((np_longitudinal == 3) & (np_circunf4 == 3))] = 13
np_aha[np.where((np_longitudinal == 3) & (np_circunf4 == 4))] = 14
np_aha[np.where(np_longitudinal == 4)] = 17
np_aha[np.where(np_lvwall == 0)] = 0  # bg still bg

aha = np_to_image(np_aha, lvwall_mask.GetOrigin(), lvwall_mask.GetSpacing(), lvwall_mask.GetDirection(),
                  sitk.sitkUInt8, name=patient_name, study_description='LV wall', series_description='aha')
sitk.WriteImage(aha, lvwall_aha_filename)


# Save dilated version
if dilate_wall:
    # compute aha segments also in the dilated LV wall and get mesh with labels
    np_aha_wall_dil = np.zeros(np_lvwall_dil.shape)
    np_aha_wall_dil[np.where((np_longitudinal == 1) & (np_circunf6 == 1))] = 3
    np_aha_wall_dil[np.where((np_longitudinal == 1) & (np_circunf6 == 2))] = 4
    np_aha_wall_dil[np.where((np_longitudinal == 1) & (np_circunf6 == 3))] = 5
    np_aha_wall_dil[np.where((np_longitudinal == 1) & (np_circunf6 == 4))] = 6
    np_aha_wall_dil[np.where((np_longitudinal == 1) & (np_circunf6 == 5))] = 1
    np_aha_wall_dil[np.where((np_longitudinal == 1) & (np_circunf6 == 6))] = 2
    np_aha_wall_dil[np.where((np_longitudinal == 2) & (np_circunf6 == 1))] = 9
    np_aha_wall_dil[np.where((np_longitudinal == 2) & (np_circunf6 == 2))] = 10
    np_aha_wall_dil[np.where((np_longitudinal == 2) & (np_circunf6 == 3))] = 11
    np_aha_wall_dil[np.where((np_longitudinal == 2) & (np_circunf6 == 4))] = 12
    np_aha_wall_dil[np.where((np_longitudinal == 2) & (np_circunf6 == 5))] = 7
    np_aha_wall_dil[np.where((np_longitudinal == 2) & (np_circunf6 == 6))] = 8
    np_aha_wall_dil[np.where((np_longitudinal == 3) & (np_circunf4 == 1))] = 15
    np_aha_wall_dil[np.where((np_longitudinal == 3) & (np_circunf4 == 2))] = 16
    np_aha_wall_dil[np.where((np_longitudinal == 3) & (np_circunf4 == 3))] = 13
    np_aha_wall_dil[np.where((np_longitudinal == 3) & (np_circunf4 == 4))] = 14
    np_aha_wall_dil[np.where(np_longitudinal == 4)] = 17
    np_aha_wall_dil[np.where(np_lvwall_dil == 0)] = 0  # bg still bg

    aha_wall_dil = np_to_image(np_aha_wall_dil, lvwall_mask.GetOrigin(), lvwall_mask.GetSpacing(),
                               lvwall_mask.GetDirection(),
                               sitk.sitkUInt8, name=patient_name, study_description='LV wall dil', series_description='aha')
    sitk.WriteImage(aha_wall_dil, lvwall_aha_dil_filename)

print('Elapsed time: ', time.time() - t)
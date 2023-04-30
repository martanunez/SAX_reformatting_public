"""
Implementation of "Automatic multiplanar CT reformatting from trans-axial into left ventricle short-axis view"
Marta Nun~ez-Garcia et al. STACOM 2020

Given a raw trans-axial CT image & corresponding LV endo, LV wall and RV epi segmentations (.mha or .vtk), compute
LV short axis view. Afterwards, use 'reformat_masks_to_SAX.py' to additionally reformat the corresponding segmentations
 if needed.

Get meshes from masks and use them to find the transformation that aligns:
# 1. MV plane
# 2. Septum (RV position with regard to LV position)
# 3. LV long axis

to the corresponding theoretical planes in standard short-axis view

This version includes a 4th rotation that improves LV septum alignment: after a preliminary reformat to sax,
use LV endo and LV epi masks (a slice midway along the long axis) to compute LV and RV centers and get the rotation
matrix that will place the RV to the left of the LV

Manually set image parameters in the beginning: image size (nb of voxels), spacing, keep_physical_location = True/False

Usage example:
python main.py --path example_pat0/ --ct_im ct.mha --mask_lvendo ct-lvendo.mha --mask_lvwall ct-lvwall.mha --mask_rvepi ct-rvepi.mha
python main.py --path example_pat0/ --ct_im ct.mha --mask_lvendo ct-lvendo.mha --mask_lvwall ct-lvwall.mha --mask_rvepi ct-rvepi.mha --isotropic 0

"""

from aux_functions import *
import os, time, argparse, sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, metavar='PATH', help='Data path')
parser.add_argument('--ct_im', type=str, help='Input CT image name')
parser.add_argument('--mask_lvendo', type=str, help='Input LV endo mask name')
parser.add_argument('--mask_lvwall', type=str, help='Input LV wall mask name')
parser.add_argument('--mask_rvepi', type=str, help='Input RV epi mask name')
parser.add_argument('--save', type=bool, default=False, help='Set to 1 to save intermediate results (rotated meshes)')
parser.add_argument('--isotropic', type=int, default=1, help='Set to 0 if output must not be isotropic')
args = parser.parse_args()

# Define output sax image basic parameters:
sax_origin = np.zeros(3)                        # (0,0,0) Origin
sax_size = 512                                  # isotropic size of reformatted sax image (number of voxels)
isotropic_resolution = args.isotropic           # if 0, output spacing depends on initial data size (covers entire image)
#sax_spacing = np.array([0.5, 0.5, 0.5])        # SAX isotropic spacing, it will be increased if LV cropped after reformatting
sax_spacing = np.array([0.5, 0.5, 1.0])         # Subsampling z-axis, reduces probability of image cropping when keep_world_coordinates ==True
check_lv_cropped = True                         # check if not full LV in the original TA image and exit if cropped
keep_world_coordinates = True

t = time.time()

# Input filenames
ifilename_ct = args.path + args.ct_im
ifilename_lvendo = args.path + args.mask_lvendo
ifilename_lvwall = args.path + args.mask_lvwall
ifilename_rvepi = args.path + args.mask_rvepi

if not os.path.isfile(ifilename_ct):
    sys.exit('CT input file does not exist, check the path')
if not os.path.isfile(ifilename_lvendo):
    sys.exit('LV endo mask input file does not exist, check the path')
if not os.path.isfile(ifilename_lvwall):
    sys.exit('LV wall mask input file does not exist, check the path')
if not os.path.isfile(ifilename_rvepi):
    sys.exit('RV mask input file does not exist, check the path')

if check_lv_cropped:        # check if full LV in TA view
    check_cropped_lv_TA(sitk.ReadImage(ifilename_lvwall))


# Output
ofilename_ct = args.path + args.ct_im[0:-4] + '-sax.mha'
R_sax_filename = args.path + args.ct_im[0:-4] + '-R-matrix-sax.txt'


# Read input image and define reference image parameters according to input image and desired output image size,
# origin and spacing. Direction will be changed later
im_TA = sitk.ReadImage(ifilename_ct)
dimension = im_TA.GetDimension()
min_intensity_value = np.min(sitk.GetArrayFromImage(im_TA))

reference_physical_size = np.zeros(dimension)
reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in zip(im_TA.GetSize(),
                                                                                             im_TA.GetSpacing(),
                                                                                             reference_physical_size)]

if isotropic_resolution:
    reference_spacing = sax_spacing
else:
    # high resolution but it crops borders (focus on the center of the image)
    reference_spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip([512] * dimension, reference_physical_size)]

reference_image, reference_center = compute_reference_image(im_TA, size=sax_size, spacing=reference_spacing, reference_origin=sax_origin)

###  Find rotations
R = compute_rotation_to_sax(args.path, args.path + args.mask_lvendo,
                            args.path + args.mask_lvwall, args.path + args.mask_rvepi,
                            reference_image, reference_center, sax_origin, delete_intermediate=not(args.save))
np.savetxt(R_sax_filename, R)

###  Check if LV apex will be cropped with current spacing: first resample LV wall with z spacing = 1mm to get full LV,
# compute min spacing required to avoid apex cropping and modify spacing accordingly if necessary
aux_mask = sitk.ReadImage(ifilename_lvwall)
aux_reference_image, aux_reference_center = compute_reference_image(aux_mask, size=sax_size,
                                        spacing=[reference_spacing[0], reference_spacing[1], 1],
                                        reference_origin=sax_origin)
aux_mask_sax = get_sax_view(im=aux_mask, reference_image=aux_reference_image, reference_origin=sax_origin,
                             reference_center=aux_reference_center, R=R, default_pixel_value=0.0)
# center to apex span:
np_aux = sitk.GetArrayFromImage(aux_mask_sax)
max_z1 = np.max(np.where(np_aux == 1)[0])
center_to_apex_z = max_z1 + 3 - aux_reference_center[2]      # x 1mm
# +3 == sort of safety border, leave a bit of space after the last slice with wall

min_spacing_required = np.divide(center_to_apex_z, sax_size/2)       # sax_size = nb of voxels output image
# divided by 2 because it refers to the space required for haf of the image (from center to 1 extreme)
ceil_spacing_required = round_decimals_up(min_spacing_required, decimals=1)

# change only if LV cropped, otherwise leave the default isotropic resolution specified at the beginning even if a
# smaller spacing is possible
if reference_spacing[2] < ceil_spacing_required:
    if isotropic_resolution:
        reference_spacing = np.ones(3) * ceil_spacing_required
    else:
        reference_spacing[2] = ceil_spacing_required

    # update
    reference_image, reference_center = compute_reference_image(im_TA, size=sax_size, spacing=reference_spacing,
                                                                reference_origin=sax_origin)

# Resample input image
if keep_world_coordinates:
    im_sax = get_sax_view_wcoordinates(im=im_TA, R=R, sax_size=[sax_size] * 3, reference_spacing=reference_spacing,
                                       default_pixel_value=min_intensity_value)
else:
    im_sax = get_sax_view(im=im_TA, reference_image=reference_image, reference_origin=sax_origin,
                          reference_center=reference_center, R=R, default_pixel_value=min_intensity_value)

im_sax = sitk.Cast(im_sax, sitk.sitkInt16)

patient_name = get_patientname(im_TA)
im_sax.SetMetaData('PatientName', patient_name)
im_sax.SetMetaData('StudyDescription', 'sax')
im_sax.SetMetaData('SeriesDescription', 'image')
sitk.WriteImage(im_sax, ofilename_ct, True)

print('Elapsed time: ', time.time() - t)

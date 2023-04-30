""" Given R transformation matrix, masks in TA view, and the image already reformatted to SAX, get the corresponding
masks in SAX view.
Careful: outputs from our segmentation method are:
  - masks with [0, 1] value. The resampling done during the reformatting is not accurate (especially in extremely
  thin regions (typically apical region)) due to numerical interpolation -> change the values to [0, 255] in advance,
  resample, and then threshold the result back to [0, 1].
  - LV endo and LV wall. Resampling the wall is again tricky in thin regions. Also, I need epi and endo (and not wall)
  for computing wall thickness with our method -> create epi (endo + wall) & resample endo and epi (compact shape
  -> less errors due to resampling). I will then get LV wall as epi - endo.

Reformat also RV epi

NOTE version 2 -> first I used conversion to numpy etc, now I changed to shorter/faster/more elegant approach
using only image masks. Results are identical.

Usage example:
python reformat_masks_to_SAX.py --path example_pat0/ --ct_im_sax ct-sax.mha --R_filename ct-R-matrix-sax.txt --mask_lvendo ct-lvendo.mha --mask_lvwall ct-lvwall.mha --mask_rvepi ct-rvepi.mha --mask_lvepi_sax ct-lvepi-sax.mha

"""

from aux_functions import *
import time, argparse


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, metavar='PATH', help='Data path')
parser.add_argument('--ct_im_sax', type=str, help='Already reformatted CT image name')
parser.add_argument('--R_filename', type=str, help='Computed R matrix')
parser.add_argument('--mask_lvendo', type=str, help='Input LV endo mask name')
parser.add_argument('--mask_lvwall', type=str, help='Input LV wall mask name')
parser.add_argument('--mask_rvepi', type=str, help='Input RV epi mask name')
parser.add_argument('--mask_lvepi_sax', type=str, help='OUTPUT LV epi mask name')
args = parser.parse_args()

keep_world_coordinates = True

# inputs
ifilename_im_sax = args.path + args.ct_im_sax
r_filename = args.path + args.R_filename
ifilename_lvendo = args.path + args.mask_lvendo
ifilename_lvwall = args.path + args.mask_lvwall
ifilename_rvepi = args.path + args.mask_rvepi


# outputs
lvendo_sax_filename = args.path + args.mask_lvendo[0:-4] + '-sax.mha'
lvwall_sax_filename = args.path + args.mask_lvwall[0:-4] + '-sax.mha'
rvepi_sax_filename = args.path + args.mask_rvepi[0:-4] + '-sax.mha'

lvepi_sax_filename = args.path + args.mask_lvepi_sax

if not os.path.isfile(ifilename_lvendo) or not os.path.isfile(ifilename_lvwall) or not os.path.isfile(ifilename_rvepi):
    sys.exit('One or several segmentations are missing, please check filenames.')
if not os.path.isfile(r_filename):
    sys.exit('txt file with rotation matrix is missing, please check the filename.')
if not os.path.isfile(ifilename_im_sax):
    sys.exit('CT image in SAX view is missing, please check the filename.')

# compute LV epi mask.
lvendo_TA = sitk.ReadImage(ifilename_lvendo)
lvwall_TA = sitk.ReadImage(ifilename_lvwall)
add = sitk.AddImageFilter()
lvepi_TA = add.Execute(lvendo_TA, lvwall_TA)

rvepi_TA = sitk.ReadImage(ifilename_rvepi)

ref_sax = sitk.ReadImage(ifilename_im_sax)
patient_name = get_patientname(ref_sax)

t = time.time()
lvendo_sax = reformat_mask_to_sax(lvendo_TA, ref_sax, r_filename, keep_world_coordinates)
lvendo_sax = add_basic_metadata(lvendo_sax, patient_name, 'sax', 'lvendo')
sitk.WriteImage(lvendo_sax, lvendo_sax_filename, True)

lvepi_sax = reformat_mask_to_sax(lvepi_TA, ref_sax, r_filename, keep_world_coordinates)
lvepi_sax = add_basic_metadata(lvepi_sax, patient_name, 'sax', 'lvepi')
sitk.WriteImage(lvepi_sax, lvepi_sax_filename, True)

rvepi_sax = reformat_mask_to_sax(rvepi_TA, ref_sax, r_filename, keep_world_coordinates)
rvepi_sax = add_basic_metadata(rvepi_sax, patient_name, 'sax', 'rvepi')
sitk.WriteImage(rvepi_sax, rvepi_sax_filename, True)

lvwall_sax = sitk.SubtractImageFilter().Execute(lvepi_sax, lvendo_sax)
lvwall_sax = sitk.BinaryThreshold(lvwall_sax, 1, 1, 1, 0)   # correct potential -1
lvwall_sax = add_basic_metadata(lvwall_sax, patient_name, 'sax', 'lvwall')
sitk.WriteImage(lvwall_sax, lvwall_sax_filename, True)

print('Elapsed time 1 = ', time.time()-t)     # ~ 3 s


# #####   keep v1, using numpy etc, just in case
# t = time.time()
#
# # Read already computed transformation
# R = np.loadtxt(r_filename)
#
# sax_size = ref_sax.GetSize()[0]   # only one, then compute_reference_images does: reference_size = [size] * dimension
# reference_origin = ref_sax.GetOrigin()
# reference_spacing = ref_sax.GetSpacing()
#
# reference_image, reference_center = compute_reference_image(ref_sax, size=sax_size, spacing=reference_spacing, reference_origin=reference_origin)
# patient_name = get_patientname(ref_sax)
#
# lvendo255_sax = get_sax_view(change_masks(lvendo_TA), reference_image, reference_origin, reference_center, R, default_pixel_value=0)
# np_lvendo255_sax = sitk.GetArrayFromImage(lvendo255_sax)
# np_lvendo255_sax[np.where(np_lvendo255_sax < 128)] = 0
# np_lvendo255_sax[np.where(np_lvendo255_sax >= 128)] = 1
# # lvendo_sax = np_to_im(np_lvendo255_sax, ref_im=lvendo255_sax, pixel_type=sitk.sitkUInt8)
# lvendo_sax = np_to_image(img_arr=np_lvendo255_sax, origin=lvendo255_sax.GetOrigin(), spacing=lvendo255_sax.GetSpacing(),
#                          direction=lvendo255_sax.GetDirection(), pixel_type=sitk.sitkUInt8,
#                          name=patient_name, study_description='sax', series_description='lvendo')
# sitk.WriteImage(lvendo_sax, lvendo_sax_filename, True)
#
# lvepi255_sax = get_sax_view(change_masks(lvepi_TA), reference_image, reference_origin, reference_center, R, default_pixel_value=0)
# np_lvepi255_sax = sitk.GetArrayFromImage(lvepi255_sax)
# np_lvepi255_sax[np.where(np_lvepi255_sax < 128)] = 0
# np_lvepi255_sax[np.where(np_lvepi255_sax >= 128)] = 1
# # lvepi_sax = np_to_im(np_lvepi255_sax, ref_im=lvepi255_sax, pixel_type=sitk.sitkUInt8)
# lvepi_sax = np_to_image(img_arr=np_lvepi255_sax, origin=lvepi255_sax.GetOrigin(), spacing=lvepi255_sax.GetSpacing(),
#                          direction=lvepi255_sax.GetDirection(), pixel_type=sitk.sitkUInt8,
#                          name=patient_name, study_description='sax', series_description='lvepi')
# sitk.WriteImage(lvepi_sax, lvepi_sax_filename, True)
#
# rvepi255_sax = get_sax_view(change_masks(rvepi_TA), reference_image, reference_origin, reference_center, R, default_pixel_value=0)
# np_rvepi255_sax = sitk.GetArrayFromImage(rvepi255_sax)
# np_rvepi255_sax[np.where(np_rvepi255_sax < 128)] = 0
# np_rvepi255_sax[np.where(np_rvepi255_sax >= 128)] = 1
# # rvepi_sax = np_to_im(np_rvepi255_sax, ref_im=rvepi255_sax, pixel_type=sitk.sitkUInt8)
# rvepi_sax = np_to_image(img_arr=np_rvepi255_sax, origin=rvepi255_sax.GetOrigin(), spacing=rvepi255_sax.GetSpacing(),
#                          direction=rvepi255_sax.GetDirection(), pixel_type=sitk.sitkUInt8,
#                          name=patient_name, study_description='sax', series_description='rvepi')
# sitk.WriteImage(rvepi_sax, rvepi_sax_filename, True)
#
# np_lvwall_sax = np_lvepi255_sax - np_lvendo255_sax   # only 0 and 1 hopefully... shoudn't have -1...
# if len(np.unique(np_lvwall_sax)) > 2:
#     print('Values in LV wall mask: ', np.unique(np_lvwall_sax))
# # lvwall_sax = np_to_im(np_lvwall_sax, ref_im=rvepi255_sax, pixel_type=sitk.sitkUInt8)
# lvwall_sax = np_to_image(img_arr=np_lvwall_sax, origin=lvendo255_sax.GetOrigin(), spacing=lvendo255_sax.GetSpacing(),
#                          direction=lvendo255_sax.GetDirection(), pixel_type=sitk.sitkUInt8,
#                          name=patient_name, study_description='sax', series_description='lvwall')
# sitk.WriteImage(lvwall_sax, lvwall_sax_filename, True)
#
# print('Elapsed time 2 = ', time.time()-t)    # 20.15 s
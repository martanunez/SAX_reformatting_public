""" Given a LV wall segmentation and one corresponding mesh (endo, epi, midwall etc) compute 2 surface parcellations:
      1. 17-AHA segmentation projecting the division on the LV wall image to the mesh (probe filter)
      2. Another parcellation directly computed on the mesh (17 regions but not 17-AHA since this one fully includes
      the basal region)

Similar algorithms than the ones used in compute_17_aha_segments_LVwall.py
Usage example:
python compute_17_segments_mesh.py --path example_pat0/ --ct_mask ct-lvepi-sax.mha --ct_lvwall_labels_mask ct-lvwall-sax-dil-aha.mha
"""

from aux_functions import *
import argparse, time

def read_mhaimage(filename):
    """Read .mha file"""
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, metavar='PATH', help='Data path')
parser.add_argument('--ct_mask', type=str, help='Input mask. I will compute a mesh from this mask')
parser.add_argument('--ct_lvwall_labels_mask', type=str, help='Input LV wall mask with labels')
args = parser.parse_args()

t = time.time()

# Input filenames
mask_input_filename = args.path + args.ct_mask     # I will create and parcellate the epi mesh as example case
lvwall_filename = args.path + args.ct_lvwall_labels_mask    # mask with labels computed with compute_17_aha_segments_LVwall.py'

# output
mesh_output_filename = mask_input_filename[0:-4] + '-parcellated-mesh.vtk'

if not os.path.isfile(lvwall_filename):
    sys.exit('Input file does not exist, check the path or run compute_17_aha_segments_LVwall.py')

patient_name = get_patientname(sitk.ReadImage(lvwall_filename))

# compute mesh
mesh = get_mesh(mask_input_filename, args.path)

# probe the mesh with LV wall dilated with AHA labels
source = read_mhaimage(lvwall_filename)
probe = vtk.vtkProbeFilter()
probe.SetSourceData(source)
probe.SetInputData(mesh)
probe.Update()
mesh = probe.GetOutput()

aux_array = mesh.GetPointData().GetArray('MetaImage')
aux_array.SetName('17-aha')
mesh.GetPointData().AddArray(aux_array)
mesh.GetPointData().RemoveArray('MetaImage')

# get mesh point coordinates
m_coords = np.zeros([mesh.GetNumberOfPoints(), 3])
for i in range(mesh.GetNumberOfPoints()):
    m_coords[i, :] = np.array(mesh.GetPoint(i))

# CAREFUL aha array obtained after probing the image has known errors in the regions limits, better use mesh threshold
# + connectivity
apex_region = extractlargestregion(pointthreshold(mesh, '17-aha', 17, 17))
m_coords_apex = np.zeros([apex_region.GetNumberOfPoints(), 3])
for i in range(apex_region.GetNumberOfPoints()):
    m_coords_apex[i, :] = np.array(apex_region.GetPoint(i))
limit_z = np.min(m_coords_apex[:, 2])

z_wall_values = np.unique(m_coords[:, 2])
extension_z = limit_z - np.min(z_wall_values)

# Create longitudinal divisions
nbins = 3
bin_width = np.divide(extension_z, nbins)
long_bins = np.arange(np.min(z_wall_values), limit_z, bin_width)
np_longitudinal = np.zeros(mesh.GetNumberOfPoints())
np_longitudinal[np.where((m_coords[:, 2] >= long_bins[0]) & (m_coords[:, 2] < long_bins[1]))] = 1
np_longitudinal[np.where((m_coords[:, 2] >= long_bins[1]) & (m_coords[:, 2] < long_bins[2]))] = 2
np_longitudinal[np.where((m_coords[:, 2] >= long_bins[2]) & (m_coords[:, 2] < limit_z))] = 3
np_longitudinal[np.where(m_coords[:, 2] >= limit_z)] = 4

# I have to include this array to do a pointthreshold later, I will remove it afterwards
array_longitudinal = numpy_to_vtk(np_longitudinal)
array_longitudinal.SetName('longitudinal')
mesh.GetPointData().AddArray(array_longitudinal)

# Create circunferential division
np_circunf6 = np.zeros(np_longitudinal.shape)
np_circunf4 = np.zeros(np_longitudinal.shape)
bin_theta_width6 = np.divide(2 * np.pi, 6)    # 60'
bin_theta_width4 = np.divide(2 * np.pi, 4)    # 90'
bins_theta6 = np.arange(-np.pi, np.pi + bin_theta_width6, bin_theta_width6)     # [-pi, pi]
bins_theta4 = np.arange(-np.pi, np.pi + bin_theta_width4, bin_theta_width4) + np.pi/4    # phase difference

# only 2 different centroids to do not see jumps when changing longitudinal region
centers = np.zeros([2, 3])      # use same shape, storing, just repeat values
lv_centroid = get_center_of_mass(pointthreshold(mesh, 'longitudinal', 1, 2))
centers[0, 0] = lv_centroid[0]   # x
centers[1, 0] = lv_centroid[1]   # y
centers[0, 1] = lv_centroid[0]
centers[1, 1] = lv_centroid[1]
# for this last one, better use ONLY apex, otherwise the junction between apex and the 4 sections looks weird...
# lv_centroid = get_center_of_mass(pointthreshold(mesh, 'longitudinal', 3, 4))
# centers[0, 2] = lv_centroid[0]
# centers[1, 2] = lv_centroid[1]
lv_centroid = get_center_of_mass(pointthreshold(mesh, '17-aha', 17, 17))
centers[0, 2] = lv_centroid[0]
centers[1, 2] = lv_centroid[1]

for section in range(3):
    r, theta = cartesian_to_polar(m_coords[:, 0] - centers[0, section], m_coords[:, 1] - centers[1, section])  # use centroid of region with longitudinal = section = {1,2,3,4}
    np_circunf6[np.where((np_longitudinal == section + 1) & (theta >= bins_theta6[0]) & (theta <= bins_theta6[1]))] = 1    # section labels start at 1
    np_circunf6[np.where((np_longitudinal == section + 1) & (theta >= bins_theta6[1]) & (theta <= bins_theta6[2]))] = 2
    np_circunf6[np.where((np_longitudinal == section + 1) & (theta >= bins_theta6[2]) & (theta <= bins_theta6[3]))] = 3
    np_circunf6[np.where((np_longitudinal == section + 1) & (theta >= bins_theta6[3]) & (theta <= bins_theta6[4]))] = 4
    np_circunf6[np.where((np_longitudinal == section + 1) & (theta >= bins_theta6[4]) & (theta <= bins_theta6[5]))] = 5
    np_circunf6[np.where((np_longitudinal == section + 1) & (theta >= bins_theta6[5]) & (theta <= 4))] = 6

    np_circunf4[np.where((np_longitudinal == section + 1) & (theta >= bins_theta4[0]) & (theta <= bins_theta4[1]))] = 1
    np_circunf4[np.where((np_longitudinal == section + 1) & (theta >= bins_theta4[1]) & (theta <= bins_theta4[2]))] = 2
    np_circunf4[np.where((np_longitudinal == section + 1) & (theta >= bins_theta4[2]) & (theta <= bins_theta4[3]))] = 3
    np_circunf4[np.where((np_longitudinal == section + 1) & (theta >= bins_theta4[3]) & (theta <= 4))] = 4   # pi discontinuity, continue in the next line
    np_circunf4[np.where((np_longitudinal == section + 1) & (theta >= -np.pi) & (theta <= -np.pi + np.pi/4))] = 4

# using only 1 reference centering point, the apex
centroid = get_center_of_mass(pointthreshold(mesh, '17-aha', 17, 17))
r, theta = cartesian_to_polar(m_coords[:, 0] - centroid[0], m_coords[:, 1] - centroid[1])  # use centroid of region with longitudinal = section = {1,2,3,4}
np_circunf6[np.where((np_longitudinal < 3) & (theta >= bins_theta6[0]) & (theta <= bins_theta6[1]))] = 1    # section labels start at 1
np_circunf6[np.where((np_longitudinal < 3) & (theta >= bins_theta6[1]) & (theta <= bins_theta6[2]))] = 2
np_circunf6[np.where((np_longitudinal < 3) & (theta >= bins_theta6[2]) & (theta <= bins_theta6[3]))] = 3
np_circunf6[np.where((np_longitudinal < 3) & (theta >= bins_theta6[3]) & (theta <= bins_theta6[4]))] = 4
np_circunf6[np.where((np_longitudinal < 3) & (theta >= bins_theta6[4]) & (theta <= bins_theta6[5]))] = 5
np_circunf6[np.where((np_longitudinal < 3) & (theta >= bins_theta6[5]) & (theta <= 4))] = 6
np_circunf4[np.where((np_longitudinal == 3) & (theta >= bins_theta4[0]) & (theta <= bins_theta4[1]))] = 1
np_circunf4[np.where((np_longitudinal == 3) & (theta >= bins_theta4[1]) & (theta <= bins_theta4[2]))] = 2
np_circunf4[np.where((np_longitudinal == 3) & (theta >= bins_theta4[2]) & (theta <= bins_theta4[3]))] = 3
np_circunf4[np.where((np_longitudinal == 3) & (theta >= bins_theta4[3]) & (theta <= 4))] = 4   # pi discontinuity, continue in the next line
np_circunf4[np.where((np_longitudinal == 3) & (theta >= -np.pi) & (theta <= -np.pi + np.pi/4))] = 4


# define the 17 regions
np_regions = np.zeros(np_circunf6.shape)
np_regions[np.where((np_longitudinal == 1) & (np_circunf6 == 1))] = 3
np_regions[np.where((np_longitudinal == 1) & (np_circunf6 == 2))] = 4
np_regions[np.where((np_longitudinal == 1) & (np_circunf6 == 3))] = 5
np_regions[np.where((np_longitudinal == 1) & (np_circunf6 == 4))] = 6
np_regions[np.where((np_longitudinal == 1) & (np_circunf6 == 5))] = 1
np_regions[np.where((np_longitudinal == 1) & (np_circunf6 == 6))] = 2
np_regions[np.where((np_longitudinal == 2) & (np_circunf6 == 1))] = 9
np_regions[np.where((np_longitudinal == 2) & (np_circunf6 == 2))] = 10
np_regions[np.where((np_longitudinal == 2) & (np_circunf6 == 3))] = 11
np_regions[np.where((np_longitudinal == 2) & (np_circunf6 == 4))] = 12
np_regions[np.where((np_longitudinal == 2) & (np_circunf6 == 5))] = 7
np_regions[np.where((np_longitudinal == 2) & (np_circunf6 == 6))] = 8
np_regions[np.where((np_longitudinal == 3) & (np_circunf4 == 1))] = 15
np_regions[np.where((np_longitudinal == 3) & (np_circunf4 == 2))] = 16
np_regions[np.where((np_longitudinal == 3) & (np_circunf4 == 3))] = 13
np_regions[np.where((np_longitudinal == 3) & (np_circunf4 == 4))] = 14
np_regions[np.where(np_longitudinal == 4)] = 17

array = numpy_to_vtk(np_regions)
array.SetName('regions')
mesh.GetPointData().AddArray(array)
try:
    mesh.GetPointData().RemoveArray('vtkValidPointMask')     # Added when probing, useless now
except:
    pass

mesh.GetPointData().RemoveArray('longitudinal')

writevtk(mesh, mesh_output_filename)

print('Elapsed time: ', time.time() - t)
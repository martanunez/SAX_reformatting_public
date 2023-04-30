import SimpleITK as sitk
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import math, os, sys
import pyvista as pv
import pyacvd


def readvtk(filename):
    """Read VTK file"""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def writevtk(surface, filename, type='binary'):
    """Write binary or ascii VTK file"""
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(surface)
    writer.SetFileName(filename)
    if type == 'ascii':
        writer.SetFileTypeToASCII()
    elif type == 'binary':
        writer.SetFileTypeToBinary()
    writer.Write()


def surfacearea(polydata):
    properties = vtk.vtkMassProperties()
    properties.SetInputData(polydata)
    properties.Update()
    return properties.GetSurfaceArea()


def volume(polydata):
    """Compute volume in polydata."""
    properties = vtk.vtkMassProperties()
    properties.SetInputData(polydata)
    properties.Update()
    return properties.GetVolume()


def check_cropped_lv_TA(lvwall_im):
    """ Check if the LV is not complete in the original raw TA image """
    np_mask = sitk.GetArrayFromImage(lvwall_im)
    overlap = np.sum(np_mask[0, :, :])      # check the z axis (ITK-snap world), add up the last slice.
    if overlap > 0.0:                       # If not overlap, sum = 0
        print('Nb of overlapping (with border) voxels', overlap)
        sys.exit('Not complete LV in TA view. If you still want to compute the corresponding SAX, set variable '
                 'check_lv_cropped to false and run again.')


def change_masks(mask):
    """Change mask labels from [0,1] to [0, 255]"""
    np_mask = sitk.GetArrayFromImage(mask)
    np_mask[np.where(np_mask == 1)] = 255
    mask_out = sitk.GetImageFromArray(np_mask)
    mask_out = sitk.Cast(mask_out, sitk.sitkUInt8)
    mask_out.SetSpacing(mask.GetSpacing())
    mask_out.SetOrigin(mask.GetOrigin())
    mask_out.SetDirection(mask.GetDirection())
    return mask_out


def np_to_image(img_arr, origin, spacing, direction, pixel_type, name='',  study_description='', series_description=''):
    """Save numpy array as itk image (volume) specifiying origin, spacing and direction desired
    Add also few metadata."""
    itk_img = sitk.GetImageFromArray(img_arr, isVector=False)
    itk_img = sitk.Cast(itk_img, pixel_type)    # reduce size by specifying minimum required pixel type, i.e. sitk.sitkUInt8 for masks, sitk.sitkInt16 for CTs, etc
    itk_img.SetSpacing(spacing)
    itk_img.SetOrigin(origin)
    itk_img.SetDirection(direction)
    itk_img.SetMetaData('PatientName', name)
    itk_img.SetMetaData('StudyDescription', study_description)
    itk_img.SetMetaData('SeriesDescription', series_description)
    return itk_img


def get_patientname(im):
    """ Try to get the patient name from the metadata. If empty return '' """
    patient_name = ''
    try:
        patient_name = im.GetMetaData('PatientName')
    except:
        pass
    try:
        patient_name = im.GetMetaData('0010|0020')
    except:
        pass
    return patient_name


def get_mesh(inputFilename, path):
    """Apply marching cubes to get mesh from corresponding mask
    If masks are .mha read and write as .vtk before"""
    remove = False
    if inputFilename[-3:] == 'mha':
        sitk.WriteImage(sitk.ReadImage(inputFilename),  path + '/aux.vtk')
        inputFilename = path + '/aux.vtk'
        remove = True
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(inputFilename)
    reader.Update()
    # Marching Cubes
    mc = vtk.vtkImageMarchingCubes()
    mc.SetInputData(reader.GetOutput())
    mc.SetNumberOfContours(1)
    mc.SetValue(0, 0.5)                             # for [0, 1] masks. Change accordingly otherwise
    mc.Update()
    if remove:
        os.remove(path + '/aux.vtk')
    return mc.GetOutput()


def vtk_to_pyvista_mesh(mesh):
    """Read vtk Polydata mesh and convert it to pyvista polydata type
    It's the same as pv.wrap() I didn't know that one"""
    vertices = np.zeros((mesh.GetNumberOfPoints(), 3))
    faces = np.zeros([mesh.GetNumberOfCells(), 4]).astype(int)
    for i in range(mesh.GetNumberOfPoints()):
        vertices[i, :] = mesh.GetPoint(i)
    for i in range(mesh.GetNumberOfCells()):
        faces[i, :] = [3, mesh.GetCell(i).GetPointIds().GetId(0), mesh.GetCell(i).GetPointIds().GetId(1), mesh.GetCell(i).GetPointIds().GetId(2)]
    return pv.PolyData(vertices, faces)


def uniform_remesh(ifilename, ofilename, nbpoints, subdivide=False):
    vtkmesh = readvtk(ifilename)
    # mesh = vtk_to_pyvista_mesh(vtkmesh)
    mesh = pv.wrap(vtkmesh)    # faster
    clus = pyacvd.Clustering(mesh)
    if subdivide:
        clus.subdivide(3)        # not necessary most of the cases (very slow), but mandatory sometimes :s
    clus.cluster(nbpoints)
    remeshed = clus.create_mesh()
    remeshed.save(ofilename, binary=True, texture=None)


def get_center_of_mass(m):
    """ Get center of mass of mesh m as numpy array"""
    centerOfMassFilter = vtk.vtkCenterOfMass()
    centerOfMassFilter.SetInputData(m)
    centerOfMassFilter.SetUseScalarsAsWeights(False)
    centerOfMassFilter.Update()
    center = np.array(centerOfMassFilter.GetCenter())
    return center


def euclideandistance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 +
                     (point1[1] - point2[1])**2 +
                     (point1[2] - point2[2])**2)


def numpy_to_vtk_M(nparray, name):
    vtkarray = vtk.vtkDoubleArray()
    vtkarray.SetName(name)
    vtkarray.SetNumberOfTuples(len(nparray))
    for j in range(len(nparray)):
        vtkarray.SetTuple1(j, nparray[j])
    return vtkarray


def pointthreshold(polydata, arrayname, start=0, end=1, alloff=0):
    """ Clip polydata according to given thresholds in scalar array"""
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, arrayname)
    threshold.ThresholdBetween(start, end)
    if (alloff):
        threshold.AllScalarsOff()
    threshold.Update()
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputData(threshold.GetOutput())
    surfer.Update()
    return surfer.GetOutput()


def extractlargestregion(polydata):
    """Keep only biggest region"""
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputData(polydata)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surfer.GetOutput())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputData(cleaner.GetOutput())
    connect.SetExtractionModeToLargestRegion()
    connect.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(connect.GetOutput())
    cleaner.Update()
    return cleaner.GetOutput()


def extract_connected_components(polydata):
    """ Extract connected components, return polydata with RegionId array and number of connected components"""
    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputData(polydata)
    connect.ScalarConnectivityOn()
    connect.SetExtractionModeToAllRegions()
    connect.ColorRegionsOn()
    connect.Update()
    n_cc = connect.GetNumberOfExtractedRegions()
    return connect.GetOutput(), n_cc


def detect_mv(lv_endo, wall, rv, max_dist_wall=5.0, factor_for_maxdist_rv=2):
    """ Detect points in the MV plane as points in LV endo far from LV wall (distance to wall > max_dist) AND far
    from the RV (to avoid getting orientations corresponding to the Aorta). Return MV polydata.
    CAREFUL: cases with holes in the LV wall segmentation (segmentation errors due to extremely thin wall,
    calcifications etc.) may wrongly detect the MV close to those holes too. Added condition related to the position of the
    connected components & added condition to keep only biggest region """

    # max_dist_wall = 5.0  # this can be left generic (independent of LV size), distance from wall
    endo_npoints = lv_endo.GetNumberOfPoints()
    mv_array = np.zeros(endo_npoints)
    np_distances_wall = np.zeros(endo_npoints)
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(wall)
    locator.BuildLocator()
    for i in range(lv_endo.GetNumberOfPoints()):
        point = lv_endo.GetPoint(i)
        closestpoint_id = locator.FindClosestPoint(point)
        np_distances_wall[i] = euclideandistance(point, wall.GetPoint(closestpoint_id))
    lv_endo.GetPointData().AddArray(numpy_to_vtk_M(np_distances_wall, 'dist_to_wall'))

    np_distances_rv = np.zeros(endo_npoints)
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(rv)
    locator.BuildLocator()
    for i in range(lv_endo.GetNumberOfPoints()):
        point = lv_endo.GetPoint(i)
        closestpoint_id = locator.FindClosestPoint(point)
        np_distances_rv[i] = euclideandistance(point, rv.GetPoint(closestpoint_id))
    lv_endo.GetPointData().AddArray(
        numpy_to_vtk_M(np_distances_rv, 'dist_to_rv'))  # I will use this later for alignment wrt RV
    max_abs_dist_rv = np.max(np_distances_rv)
    max_dist_rv = np.divide(max_abs_dist_rv, factor_for_maxdist_rv)  # factor_for_maxdist_rv=2 -> more than half way from RV
    # consider a bigger region, allow to be closer to RV if there is any problem... it doesn't seem related to the size
    # but sometimes the remeshing fails and produces empty polydata
    # max_dist_rv = np.divide(max_abs_dist_rv, 2.5)

    mv_array[np.where((np_distances_wall >= max_dist_wall) & (np_distances_rv >= max_dist_rv))] = 1  # I can't do the verification of empty MV points here because I may have few points with mv_array = 1 that will create an empty surface (not connected, no cells)
    lv_endo.GetPointData().AddArray(numpy_to_vtk_M(mv_array, 'mv'))

    # Careful, trabeculations may be also far from LV wall and RV and will be marked as MV
    far_regions = pointthreshold(lv_endo, 'mv', 1, 1)
    if far_regions.GetNumberOfPoints() == 0:
        max_dist_rv = np.divide(max_abs_dist_rv, 3)  # Allow smaller distance. This may be needed with spherical LV where the highest distances are far from the base
        mv_array[np.where((np_distances_wall >= max_dist_wall) & (np_distances_rv >= max_dist_rv))] = 1

    lv_endo.GetPointData().AddArray(numpy_to_vtk_M(mv_array, 'mv'))
    far_regions = pointthreshold(lv_endo, 'mv', 1, 1)

    # get number of cc, ideally only 1 but there can be more if there are holes in wall
    far_regions_ccs, nb = extract_connected_components(far_regions)

    if nb > 1:
        # the MV is more likely to have more positive Y. Filter using y position + biggest region later
        centroids = np.zeros([nb, 3])
        for i in range(nb):
            centroids[i, :] = get_center_of_mass(pointthreshold(far_regions_ccs, 'RegionId', i, i))
        y_span = np.max(centroids[:, 1]) - np.min(centroids[:, 1])
        y_threshold = np.max(centroids[:, 1]) - np.divide(y_span, 2)

        append = vtk.vtkAppendPolyData()
        for i in range(nb):
            if centroids[i, 1] > y_threshold:
                # create new polydata only with pieces that pass the threshold
                append.AddInputData(pointthreshold(far_regions_ccs, 'RegionId', i, i))
                append.Update()

        # still get the biggest one among the ones that pass the threshold
        mv_mesh = extractlargestregion(append.GetOutput())

        # # if the previous filtering fails, check the mesh and see if directly getting the biggest cc does the work:
        # mv_mesh = extractlargestregion(far_regions)

    else:  # there is only 1, no need to extract biggest one
        mv_mesh = extractlargestregion(far_regions)

    # Update 'mv' array, keep only biggest region, I'll need it later
    mv_array = np.zeros(endo_npoints)
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(lv_endo)
    locator.BuildLocator()
    for i in range(mv_mesh.GetNumberOfPoints()):
        if euclideandistance(mv_mesh.GetPoint(i), lv_endo.GetPoint(locator.FindClosestPoint(mv_mesh.GetPoint(i)))) < 0.1:
            mv_array[int(locator.FindClosestPoint(mv_mesh.GetPoint(i)))] = 1
    lv_endo.GetPointData().AddArray(numpy_to_vtk_M(mv_array, 'mv'))

    return lv_endo, mv_mesh


def cellnormals(polydata):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOff()
    normals.ComputeCellNormalsOn()
    normals.Update()
    return normals.GetOutput()


def get_vertices(polydata):
    npoints = polydata.GetNumberOfPoints()
    v = np.zeros([npoints, 3])
    for i in range(npoints):
        v[i, :] = np.array(polydata.GetPoint(i))
    return v


def set_vertices(polydata, v):
    points = vtk.vtkPoints()
    npoints = v.shape[0]
    points.SetNumberOfPoints(npoints)
    for i in range(npoints):
        points.SetPoint(i, v[i, :])
    polydata.SetPoints(points)
    return polydata


def compute_reference_image(im_ref, size=512, spacing=np.array([0.5, 0.5, 0.5]),
                            reference_origin=np.array([0, 0, 0])):
    """ Given input image, desired physical center and spacing, compute reference parameters needed for resampling"""
    dimension = im_ref.GetDimension()
    reference_direction = np.identity(dimension).flatten()
    reference_size = [size] * dimension

    reference_spacing = spacing
    reference_image = sitk.Image(reference_size, im_ref.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)

    reference_image.SetDirection(reference_direction)
    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(
        np.array(reference_image.GetSize()) / 2.0))  # geometrical center (coordinates)
    return reference_image, reference_center


def compute_reference_image_keep_location(im_ref, size=512, spacing=np.array([0.5, 0.5, 0.5])):
    """ Given input image compute reference parameters needed for resampling maintaining (approx) the same physical
    location """

    dimension = im_ref.GetDimension()
    reference_size = [size] * dimension
    reference_spacing = spacing
    reference_image = sitk.Image(reference_size, im_ref.GetPixelIDValue())
    reference_image.SetOrigin(im_ref.GetOrigin())
    reference_image.SetDirection(im_ref.GetDirection())
    reference_image.SetSpacing(reference_spacing)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(
        np.array(reference_image.GetSize()) / 2.0))  # geometrical center (coordinates)
    return reference_image, reference_center


def compute_rotation_to_sax(path, lvendo_filename, lvwall_filename, rvepi_filename,
                            reference_image, reference_center, reference_origin, delete_intermediate=True):
    lvendo_m_filename = path + 'lvendo-mesh.vtk'
    lvwall_m_filename = path + 'lvwall-mesh.vtk'
    rvepi_m_filename = path + 'rvepi-mesh.vtk'
    npoints_remesh = 2000  # points for uniform remesh.
    remeshed_lv_filename = 'lvendo-remeshed-' + str(npoints_remesh) + '.vtk'
    remeshed_wall_filename = 'lvwall-remeshed-' + str(npoints_remesh) + '.vtk'
    remeshed_rv_filename = 'rvepi-remeshed-' + str(npoints_remesh) + '.vtk'
    aux_mv_filename1 = path + 'mv-rotated.vtk'
    aux_mv_filename2 = 'mv-rotated-remeshed.vtk'
    lv_rotated1_mesh_filename = path + 'lvendo-mesh-rotated1.vtk'
    rv_rotated1_mesh_filename = path + 'rvepi-mesh-rotated1.vtk'
    lv_rotated2_mesh_filename = path + 'lvendo-mesh-rotated2.vtk'
    # rv_rotated2_mesh_filename = path + 'rvepi-mesh-rotated2.vtk'

    # Create meshes from image (marching cubes) and uniformly remesh them. Needed only once, check if file already exists.
    if not os.path.isfile(lvendo_m_filename):
        lv_source_m = get_mesh(lvendo_filename, path)
        writevtk(lv_source_m, lvendo_m_filename)
    if not os.path.isfile(lvwall_m_filename):
        wall_source_m = get_mesh(lvwall_filename, path)
        writevtk(wall_source_m, lvwall_m_filename)
    if not os.path.isfile(rvepi_m_filename):
        rv_source_m = get_mesh(rvepi_filename, path)
        writevtk(rv_source_m, rvepi_m_filename)

    # # Uniform remesh
    # if not os.path.isfile(path + remeshed_lv_filename):
    #     # print('Uniform remesh LV endo mesh...')
    #     os.system('/home/marta/code/Uniform_remesh/ACVD/bin/ACVD ' + lvendo_m_filename + ' ' + str(
    #         npoints_remesh) + ' 0 -o ' + path + ' -of ' + remeshed_lv_filename + ' -d 0')
    # if not os.path.isfile(path + remeshed_wall_filename):
    #     # print('Uniform remesh LV wall mesh...')
    #     os.system('/home/marta/code/Uniform_remesh/ACVD/bin/ACVD ' + lvwall_m_filename + ' ' + str(
    #         npoints_remesh) + ' 0 -o ' + path + ' -of ' + remeshed_wall_filename + ' -d 0')
    # if not os.path.isfile(path + remeshed_rv_filename):
    #     # print('Uniform remesh RV epi mesh...')
    #     os.system('/home/marta/code/Uniform_remesh/ACVD/bin/ACVD ' + rvepi_m_filename + ' ' + str(
    #         npoints_remesh) + ' 0 -o ' + path + ' -of ' + remeshed_rv_filename + ' -d 0')
    #
    if not os.path.isfile(path + remeshed_lv_filename):
        uniform_remesh(lvendo_m_filename, path + remeshed_lv_filename, npoints_remesh)
    if not os.path.isfile(path + remeshed_wall_filename):
        uniform_remesh(lvwall_m_filename, path + remeshed_wall_filename, npoints_remesh)
    if not os.path.isfile(path + remeshed_rv_filename):
        uniform_remesh(rvepi_m_filename, path + remeshed_rv_filename, npoints_remesh)

    # print('Reading LV endo, LV wall, and RV epi uniform meshes...')
    lv_endo_m = readvtk(path + remeshed_lv_filename)
    lv_wall_m = readvtk(path + remeshed_wall_filename)
    rv_epi_m = readvtk(path + remeshed_rv_filename)

    ##### Align MV to theoretical MV plane.
    # Detect MV and MV centroid. Find points in the endo that are far from points in the LV wall.
    lv_endo_m, mv_m = detect_mv(lv_endo_m, lv_wall_m, rv_epi_m, max_dist_wall=5.0, factor_for_maxdist_rv=2.0)
    # print('Number of points of detected MV', mv_m.GetNumberOfPoints())

    mv_normals_m = cellnormals(mv_m)
    mv_normals = vtk_to_numpy(mv_normals_m.GetCellData().GetArray('Normals'))
    mv_normal = np.mean(mv_normals, axis=0)
    mv_centroid = get_center_of_mass(mv_m)
    writevtk(lv_endo_m, path + remeshed_lv_filename)

    # Find apex id as furthest point to mv_centroid (I will use it in the latest alignment)
    np_distances_mv = np.zeros(lv_endo_m.GetNumberOfPoints())
    for i in range(lv_endo_m.GetNumberOfPoints()):
        np_distances_mv[i] = euclideandistance(mv_centroid, lv_endo_m.GetPoint(i))
    lv_endo_m.GetPointData().AddArray(numpy_to_vtk_M(np_distances_mv, 'dist_to_MV_centroid'))
    writevtk(lv_endo_m, path + remeshed_lv_filename)
    apex_id = np.argmax(np_distances_mv)
    # Get initial long axis (just to draw the plane for the paper)
    p0_aux = mv_centroid
    p1_aux = np.array(lv_endo_m.GetPoint(apex_id))
    v_lax = np.divide(p1_aux - p0_aux, np.linalg.norm(p1_aux - p0_aux))  # get unit vector, normalize

    # Find rotation matrix that will align the MV plane to theoretical MV plane
    v1 = - np.divide(mv_normal, np.linalg.norm(mv_normal))  # Use opposite of the normal (normals point towards the outside) and positive normal of the XY plane. SAME.
    v2 = np.array([0, 0, 1])  # Theoretical MV normal, short axis plane, XY plane

    # Given v1 and v2 vectors, find rotation matrix that aligns v1 to v2.
    # Adapted from https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v = np.cross(v1, v2)
    # s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R1 = np.eye(3) + Vx + np.dot(Vx, Vx) * np.divide(1, 1 + c)

    # Apply transformation to LV endo mesh
    lv_vertices_ori = get_vertices(lv_endo_m).T
    lv_vertices_rotated = np.dot(R1, lv_vertices_ori)
    m_lv_vertices_rotated = set_vertices(lv_endo_m, lv_vertices_rotated.T)
    writevtk(m_lv_vertices_rotated, lv_rotated1_mesh_filename)

    # Rotate also RV. I will need it later for subsequent alignment
    rv_vertices_ori = get_vertices(rv_epi_m).T
    rv_vertices_rotated = np.dot(R1, rv_vertices_ori)
    m_rv_vertices_rotated = set_vertices(rv_epi_m, rv_vertices_rotated.T)
    writevtk(m_rv_vertices_rotated, rv_rotated1_mesh_filename)

    # Align LV septum. Find vector within the MV surface that points from LV to RV
    mv_rotated_m = pointthreshold(m_lv_vertices_rotated, 'mv', 1, 1)
    writevtk(mv_rotated_m, aux_mv_filename1)
    # os.system('/home/marta/code/Uniform_remesh/ACVD/bin/ACVD ' + aux_mv_filename1 + ' ' + str(
    #     2000) + ' 0 -o ' + path + ' -of ' + aux_mv_filename2 + ' -d 0')
    uniform_remesh(aux_mv_filename1, path + aux_mv_filename2, 2000, subdivide=True)    # need subdivide in this case..

    m_mv_rotated_remeshed = readvtk(path + aux_mv_filename2)

    p_center_mv = get_center_of_mass(m_mv_rotated_remeshed)
    # Compute distances to center of mass to keep only closest ones
    np_distances_center = np.zeros(m_mv_rotated_remeshed.GetNumberOfPoints())
    for i in range(m_mv_rotated_remeshed.GetNumberOfPoints()):
        np_distances_center[i] = euclideandistance(m_mv_rotated_remeshed.GetPoint(i), p_center_mv)

    m_mv_rotated_remeshed.GetPointData().AddArray(numpy_to_vtk_M(np_distances_center, 'dist_to_center'))
    center_poly = pointthreshold(m_mv_rotated_remeshed, 'dist_to_center', 0, 5)

    # Re-compute distances to RV, now only for the center_poly
    np_distances_rv = np.zeros(center_poly.GetNumberOfPoints())
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(m_rv_vertices_rotated)
    locator.BuildLocator()
    for i in range(center_poly.GetNumberOfPoints()):
        point = center_poly.GetPoint(i)
        closestpoint_id = locator.FindClosestPoint(point)
        np_distances_rv[i] = euclideandistance(point, m_rv_vertices_rotated.GetPoint(closestpoint_id))

    center_poly.GetPointData().AddArray(numpy_to_vtk_M(np_distances_rv, 'dist_to_rv'))
    writevtk(center_poly, path + aux_mv_filename2)
    p0_id = np.argmin(np_distances_rv)
    p1_id = np.argmax(np_distances_rv)
    p0 = np.array(center_poly.GetPoint(p0_id))
    p1 = np.array(center_poly.GetPoint(p1_id))

    v11 = np.divide(p1 - p0, np.linalg.norm(p1 - p0))  # get unit vector, normalize
    v21 = np.array([1, 0, 0])  # unit vector within MV plane and pointing from RV to LV

    v = np.cross(v11, v21)
    # s = np.linalg.norm(v)
    c = np.dot(v11, v21)
    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R2 = np.eye(3) + Vx + np.dot(Vx, Vx) * np.divide(1, 1 + c)

    # Apply the 2 rotations at the time -> R2 * (R * Vertices)
    # print('Aligning LV with regard to RV...')
    R2_R1 = np.dot(R2, R1)
    vertices_lv_rotated_twice = np.dot(R2_R1, lv_vertices_ori)
    m_vertices_lv_rotated_twice = set_vertices(lv_endo_m, vertices_lv_rotated_twice.T)
    writevtk(m_vertices_lv_rotated_twice, lv_rotated2_mesh_filename)

    # Finally, align LV long axis (i.e. line from center of MV to LV apex)
    mv_centroid = get_center_of_mass(pointthreshold(m_vertices_lv_rotated_twice, 'mv', 1, 1))
    # print('Aligning LV long axis...')
    p0 = mv_centroid
    p1 = np.array(m_vertices_lv_rotated_twice.GetPoint(apex_id))

    v1 = np.divide(p1 - p0, np.linalg.norm(p1 - p0))  # get unit vector, normalize
    v2 = np.array([0, 0, 1])  # Theoretical LV long axis
    v = np.cross(v1, v2)
    # s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R3 = np.eye(3) + Vx + np.dot(Vx, Vx) * np.divide(1, 1 + c)

    # Now, the 3 correct rotations would be R3 * (R2 * (R * vertices) )    Right!!
    R_aux = np.dot(R2, R1)
    R_final = np.dot(R3, R_aux)

    # Resample initial LV endo and RV epi masks at this stage to compute the 4th rotation (Nico)
    # (improvement of LV septum alignment)
    im_aux = sitk.ReadImage(lvendo_filename)
    dimension = im_aux.GetDimension()
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(im_aux.GetDirection())
    transform.SetTranslation(np.array(im_aux.GetOrigin()) - reference_origin)
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(im_aux.TransformContinuousIndexToPhysicalPoint(np.array(im_aux.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    # centered_transform = sitk.Transform(transform)
    centered_transform = sitk.CompositeTransform(transform)
    centered_transform.AddTransform(centering_transform)

    rotate_transform = sitk.AffineTransform(dimension)
    rotate_transform.SetCenter(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))
    direction_matrix = [R_final[0, 0], R_final[1, 0], R_final[2, 0],
                        R_final[0, 1], R_final[1, 1], R_final[2, 1],
                        R_final[0, 2], R_final[1, 2], R_final[2, 2]]

    rotate_transform.SetMatrix(direction_matrix)
    centered_transform.AddTransform(rotate_transform)

    mask_lv_rot = sitk.Resample(im_aux, reference_image, centered_transform, sitk.sitkLinear, 0.0)
    rv_im = sitk.ReadImage(rvepi_filename)
    mask_rv_rot = sitk.Resample(rv_im, reference_image, centered_transform, sitk.sitkLinear, 0.0)
    r_lvendo = sitk.GetArrayFromImage(mask_lv_rot)
    r_rvepi = sitk.GetArrayFromImage(mask_rv_rot)

    start = None
    end = None
    for i, mask2d in enumerate(r_lvendo):
        n = mask2d.sum()
        if start is None and n != 0:
            start = i
        if start is not None and n == 0:
            end = i
            break

    # if long LV, endo mask can reach the limit, in that case, take as 'end' the last slice
    if end == None:
        end = r_lvendo.shape[0]    # z in ITK
    index = start + (end - start) // 2  # midway along the long axis

    lvcenter = np.array(np.nonzero(r_lvendo[index])).mean(axis=1)
    rvcenter = np.array(np.nonzero(r_rvepi[index])).mean(axis=1)

    direction = rvcenter - lvcenter
    angle = np.arctan2(direction[1], direction[0]) + np.pi / 2

    last_rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                              [np.sin(angle), np.cos(angle), 0],
                              [0, 0, 1],
                              ])
    R_final2 = np.dot(last_rotation, R_final)               # add the 4th rotation to previous rotation matrix

    # # Not necessary I think, small effect
    # # Align again the LV long axis (most important aspect) since it may have been altered...
    # vertices_lv_rotated_aux = np.dot(R_final2, lv_vertices_ori)
    # m_vertices_lv_rotated_aux = set_vertices(lv_endo_m, vertices_lv_rotated_aux.T)
    # # writevtk(m_vertices_lv_rotated_twice, lv_rotated2_mesh_filename)
    # mv_centroid_aux = get_center_of_mass(pointthreshold(m_vertices_lv_rotated_aux, 'mv', 1, 1))
    # p0 = mv_centroid_aux
    # p1 = np.array(m_vertices_lv_rotated_aux.GetPoint(apex_id))
    # v1 = np.divide(p1 - p0, np.linalg.norm(p1 - p0))  # get unit vector, normalize
    # v2 = np.array([0, 0, 1])  # Theoretical LV long axis
    # v = np.cross(v1, v2)
    # c = np.dot(v1, v2)
    # Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    # R_aux = np.eye(3) + Vx + np.dot(Vx, Vx) * np.divide(1, 1 + c)
    #
    # R_final3 = np.dot(R_aux, R_final2)                  # final-final :)

    if delete_intermediate:  # Remove meshes and intermediate results.
        os.remove(lvendo_m_filename)
        os.remove(lvwall_m_filename)
        os.remove(rvepi_m_filename)
        os.remove(path + remeshed_lv_filename)
        os.remove(path + remeshed_wall_filename)
        os.remove(path + remeshed_rv_filename)
        os.remove(aux_mv_filename1)
        os.remove(path + aux_mv_filename2)
        os.remove(lv_rotated1_mesh_filename)
        os.remove(rv_rotated1_mesh_filename)
        os.remove(lv_rotated2_mesh_filename)
        if os.path.isfile(path + 'smooth_' + remeshed_lv_filename):  # smooth version of remeshed ims
            os.remove(path + 'smooth_' + remeshed_lv_filename)
        if os.path.isfile(path + 'smooth_' + remeshed_wall_filename):
            os.remove(path + 'smooth_' + remeshed_wall_filename)
        if os.path.isfile(path + 'smooth_' + remeshed_rv_filename):
            os.remove(path + 'smooth_' + remeshed_rv_filename)
        if os.path.isfile(path + 'smooth_' + aux_mv_filename2):
            os.remove(path + 'smooth_' + aux_mv_filename2)

    return R_final2
    # return R_final3
    # return R_final


def get_sax_view(im, reference_image, reference_origin, reference_center, R, default_pixel_value=0):
    dimension = im.GetDimension()
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(im.GetDirection())
    transform.SetTranslation(np.array(im.GetOrigin()) - reference_origin)
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(im.TransformContinuousIndexToPhysicalPoint(np.array(im.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    # centered_transform = sitk.Transform(transform)
    centered_transform = sitk.CompositeTransform(transform)
    centered_transform.AddTransform(centering_transform)
    rotate_transform = sitk.AffineTransform(dimension)
    rotate_transform.SetCenter(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))
    direction_matrix = [R[0, 0], R[1, 0], R[2, 0],
                        R[0, 1], R[1, 1], R[2, 1],
                        R[0, 2], R[1, 2], R[2, 2]]
    rotate_transform.SetMatrix(direction_matrix)
    centered_transform.AddTransform(rotate_transform)

    im_sax = sitk.Resample(im, reference_image, centered_transform, sitk.sitkLinear, int(default_pixel_value))
    im_sax.SetDirection(np.array(direction_matrix).flatten())
    return im_sax


def get_sax_view_wcoordinates(im, R, sax_size, reference_spacing, default_pixel_value):
    """ Compute SAX view maintaining image world coordinates (by Nicolas Cedilnik) """

    direction_matrix = [R[0, 0], R[1, 0], R[2, 0],
                        R[0, 1], R[1, 1], R[2, 1],
                        R[0, 2], R[1, 2], R[2, 2], ]

    transform = sitk.AffineTransform(3)
    transform.SetMatrix(direction_matrix)

    inverse = transform.GetInverse()

    extreme_points = [
        im.TransformIndexToPhysicalPoint((0, 0, 0)),
        im.TransformIndexToPhysicalPoint((im.GetWidth(), 0, 0)),
        im.TransformIndexToPhysicalPoint((im.GetWidth(), im.GetHeight(), 0)),
        im.TransformIndexToPhysicalPoint((0, im.GetHeight(), 0)),
        im.TransformIndexToPhysicalPoint((0, 0, im.GetDepth())),
        im.TransformIndexToPhysicalPoint((im.GetWidth(), 0, im.GetDepth())),
        im.TransformIndexToPhysicalPoint((im.GetWidth(), im.GetHeight(), im.GetDepth())),
        im.TransformIndexToPhysicalPoint((0, im.GetHeight(), im.GetDepth())),
    ]

    extreme_points_transformed = [inverse.TransformPoint(pnt) for pnt in extreme_points]

    min_x = min(extreme_points_transformed)[0]
    min_y = min(extreme_points_transformed, key=lambda p: p[1])[1]
    min_z = min(extreme_points_transformed, key=lambda p: p[2])[2]
    # max_x = max(extreme_points_transformed)[0]
    # max_y = max(extreme_points_transformed, key=lambda p: p[1])[1]
    # max_z = max(extreme_points_transformed, key=lambda p: p[2])[2]

    nw_direction = np.eye(3).flatten()
    nw_ori = [min_x, min_y, min_z]

    im_sax = sitk.Resample(im, sax_size, transform, sitk.sitkLinear, nw_ori, reference_spacing, nw_direction,
                           int(default_pixel_value))
    im_sax.SetOrigin(transform.TransformPoint(nw_ori))

    old_direction = np.array(im_sax.GetDirection()).reshape((3, 3))
    new_direction = []
    for cosine in old_direction:
        new_cosine = inverse.TransformVector(cosine, [0, 0, 0])
        new_direction.extend(new_cosine)
    im_sax.SetDirection(new_direction)

    return im_sax



def reformat_mask_to_sax(input_mask, ref_sax, r_sax_filename, world_coordinates):     # ref_sax = im_sax
    """ Reformat given BINARY (0,1) mask to sax. First change to 0-255 mask to avoid interpolation problems, and go
    back to 0-1 after reformatting. Get image parameters from corresponding ref_sax = ct1-sax-iso.mha for example """

    mask_255 = sitk.BinaryThreshold(input_mask, lowerThreshold=1, upperThreshold=1, insideValue=255, outsideValue=0)

    R = np.loadtxt(r_sax_filename)

    # sax_size = ref_sax.GetSize()[0]  # only one, then compute_reference_images does: reference_size = [size] * dimension
    reference_origin = ref_sax.GetOrigin()
    reference_spacing = ref_sax.GetSpacing()
    dimension = ref_sax.GetDimension()
    reference_direction = np.identity(dimension).flatten()
    reference_size = ref_sax.GetSize()

    reference_image = sitk.Image(reference_size, ref_sax.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)

    reference_image.SetDirection(reference_direction)
    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(
        np.array(reference_image.GetSize()) / 2.0))  # geometrical center (coordinates)


    if world_coordinates:
        mask_255_sax = get_sax_view_wcoordinates(mask_255, R, ref_sax.GetSize(), ref_sax.GetSpacing(),
                                                 default_pixel_value=0)

    else:
        mask_255_sax = get_sax_view(mask_255, reference_image, reference_origin, reference_center, R,
                                 default_pixel_value=0)
    # go back to [0, 1] mask
    mask_sax = sitk.BinaryThreshold(mask_255_sax, lowerThreshold=128, upperThreshold=255, insideValue=1, outsideValue=0)

    return mask_sax


def add_basic_metadata(im, patient_name, study_description, series_description):
    im.SetMetaData('PatientName', patient_name)
    im.SetMetaData('StudyDescription', study_description)
    im.SetMetaData('SeriesDescription', series_description)
    return im


def round_decimals_up(number: float, decimals: int = 2):
    """ Returns a value rounded up to a specific number of decimal places.
    From: https://kodify.net/python/math/round-decimals/#example-round-up-to-2-decimal-places"""
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)
    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def do_lv_lax_qc(prefix_path, lvendo_filename, lvwall_filename, rvepi_filename, tolerance=0.05, npoints_remesh=1500):
    """ Perform LV long-axis quality control check. Output is a warning message if LV long axis differs more than tolerance
    from the theoretical one = [0, 0, 1] """
    lvendo_mesh_filename = prefix_path + 'lvendo-sax-mesh.vtk'
    lvwall_mesh_filename = prefix_path + 'lvwall-sax-mesh.vtk'
    rvepi_mesh_filename = prefix_path + 'rvepi-sax-mesh.vtk'
    lvendo_mesh_filename2 = prefix_path + 'lvendo-sax-remeshed.vtk'
    lvwall_mesh_filename2 = prefix_path + 'lvwall-sax-remeshed.vtk'
    rvepi_mesh_filename2 = prefix_path + 'rvepi-sax-remeshed.vtk'

    lv_endo = get_mesh(lvendo_filename, prefix_path)
    wall = get_mesh(lvwall_filename, prefix_path)
    rv = get_mesh(rvepi_filename, prefix_path)
    writevtk(lv_endo, lvendo_mesh_filename)
    writevtk(wall, lvwall_mesh_filename)
    writevtk(rv, rvepi_mesh_filename)

    uniform_remesh(lvendo_mesh_filename, lvendo_mesh_filename2, npoints_remesh)
    uniform_remesh(lvwall_mesh_filename, lvwall_mesh_filename2, npoints_remesh)
    uniform_remesh(rvepi_mesh_filename, rvepi_mesh_filename2, npoints_remesh)
    lv_endo = readvtk(lvendo_mesh_filename2)
    wall = readvtk(lvwall_mesh_filename2)
    rv = readvtk(rvepi_mesh_filename2)

    lv_endo, mv_m = detect_mv(lv_endo, wall, rv, max_dist_wall=5.0, factor_for_maxdist_rv=2)

    mv_centroid = get_center_of_mass(mv_m)

    # Find apex id as furthest point to mv_centroid
    np_distances_mv = np.zeros(lv_endo.GetNumberOfPoints())
    for i in range(lv_endo.GetNumberOfPoints()):
        np_distances_mv[i] = euclideandistance(mv_centroid, lv_endo.GetPoint(i))
    lv_endo.GetPointData().AddArray(numpy_to_vtk_M(np_distances_mv, 'dist_to_MV_centroid'))
    writevtk(lv_endo, lvendo_mesh_filename2)
    apex_id = np.argmax(np_distances_mv)
    # Get LV long axis
    p0_aux = mv_centroid
    p1_aux = np.array(lv_endo.GetPoint(apex_id))
    lv_lax = np.divide(p1_aux - p0_aux, np.linalg.norm(p1_aux - p0_aux))  # get unit vector, normalize

    if lv_lax[0] > tolerance or lv_lax[0] < -tolerance or lv_lax[1] > tolerance or lv_lax[1] < -tolerance or \
            lv_lax[2] < (1 - tolerance):
        print('LV long axis seems wrong, please check. LV long axis = ', lv_lax)
    else:
        print('LV long axis seems fine. LV lax = ', lv_lax)

    os.remove(lvendo_mesh_filename)
    os.remove(lvendo_mesh_filename2)
    os.remove(lvwall_mesh_filename)
    os.remove(lvwall_mesh_filename2)
    os.remove(rvepi_mesh_filename)
    os.remove(rvepi_mesh_filename2)


def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def check_360(thetas, tolerance=0.10):
    output = True
    ref_array = np.arange(-np.pi, np.pi, 2*np.pi/360)
    for i in range(len(ref_array)):
        # find closest value in thetas
        idx = (np.abs(thetas - ref_array[i])).argmin()
        val = thetas[idx]
        if np.abs(val-ref_array[i]) > tolerance:
            output = False
    return output

import os
import sys
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, get_vtk_array_type
import pdb

def image_largest_connected(vtkimage):
    f = vtk.vtkImageConnectivityFilter()
    f.SetInputData(vtkimage)
    f.SetExtractionModeToLargestRegion()
    f.Update()
    return f.GetOutput()

def find_connected_points(cells, mesh, constraint_set):
    add_cells = []
    for i in cells:
        pt_arr = vtk.vtkIdList()
        mesh.GetCellPoints(i, pt_arr)
        for k in range(pt_arr.GetNumberOfIds()):
            p_id = pt_arr.GetId(k)
            c_arr = vtk.vtkIdList()
            mesh.GetPointCells(p_id, c_arr)
            for c in range(c_arr.GetNumberOfIds()):
                c_id = c_arr.GetId(c)
                if (c_id not in constraint_set) and (c_id not in add_cells):
                    add_cells.append(c_id)
                    constraint_set.append(c_id)
    return add_cells, constraint_set

def find_connected_points_points(points, mesh, constraint_set):
    add_points = []
    for p_id in points:
        c_arr = vtk.vtkIdList()
        mesh.GetPointCells(p_id, c_arr)
        for c in range(c_arr.GetNumberOfIds()):
            p_arr = vtk.vtkIdList()
            mesh.GetCellPoints(c_arr.GetId(c), p_arr)
            for p in range(p_arr.GetNumberOfIds()):
                p_id_i = p_arr.GetId(p)
                if (p_id_i not in constraint_set) and (p_id_i not in add_points):
                    add_points.append(p_id_i)
                    constraint_set.append(p_id_i)
    return add_points, constraint_set

def fill_hole(poly, size=10000000.):
    """
    Fill holes in VTK PolyData

    Args:
        poly: VTK PolyData to fill
        size: max size of the hole to fill
    Returns:
        poly: filled VTK PolyData
    """
    filler = vtk.vtkFillHolesFilter()
    filler.SetInputData(poly)
    filler.SetHoleSize(size)
    filler.Update()

    return filler.GetOutput()

def build_transform_matrix(image):
    import SimpleITK as sitk
    matrix = np.eye(4)
    matrix[:-1,:-1] = np.matmul(np.reshape(image.GetDirection(), (3,3)), np.diag(image.GetSpacing()))
    matrix[:-1,-1] = np.array(image.GetOrigin())
    return matrix

def map_polydata_coords(poly, displacement, transform, size):
    coords = vtk_to_numpy(poly.GetPoints().GetData())
    coords += displacement
    coords = np.concatenate((coords,np.ones((coords.shape[0],1))), axis=-1) 
    coords = np.matmul(np.linalg.inv(transform), coords.transpose()).transpose()[:,:3]
    coords /= np.array(size)
    return coords

def transform_polydata(poly, displacement, transform, size):
    coords = map_polydata_coords(poly, displacement, transform, size)
    poly.GetPoints().SetData(numpy_to_vtk(coords))
    return poly

def bound_polydata_by_image(image, poly, threshold):
    bound = vtk.vtkBox()
    image.ComputeBounds()
    b_bound = image.GetBounds()
    b_bound = [b+threshold if (i % 2) ==0 else b-threshold for i, b in enumerate(b_bound)]
    #print("Bounding box: ", b_bound)
    bound.SetBounds(b_bound)
    clipper = vtk.vtkClipPolyData()
    clipper.SetClipFunction(bound)
    clipper.SetInputData(poly)
    clipper.InsideOutOn()
    clipper.Update()
    return clipper.GetOutput()

def smooth_polydata(poly, iteration=25, boundary=False, feature=False, smoothingFactor=0.):
    """
    This function smooths a vtk polydata
    Args:
        poly: vtk polydata to smooth
        boundary: boundary smooth bool
    Returns:
        smoothed: smoothed vtk polydata
    """
    import vtk
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetPassBand(pow(10., -4. * smoothingFactor))
    smoother.SetBoundarySmoothing(boundary)
    smoother.SetFeatureEdgeSmoothing(feature)
    smoother.SetNumberOfIterations(iteration)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    smoothed = smoother.GetOutput()

    return smoothed

def geodesic_distance_map(poly, start, threshold, max_depth=3000):
    '''
    compute the geodesic distance map on a polydata from a starting point
    '''
    count = 0
    dist_arr = np.ones(poly.GetNumberOfPoints())*np.inf
    vtk_math = vtk.vtkMath()
    points = poly.GetPoints()

    pt_queue = [start]
    dist_queue = [0.]
    visited_list = [start]
    while pt_queue and count < max_depth:
        count+=1
        print("debug: ", pt_queue)
        node = pt_queue.pop(0)
        dist = dist_queue.pop(0)
        dist_arr[node] = dist

        tmplt_list = vtk.vtkIdList()
        poly.GetPointCells(node, tmplt_list)
        for i in range(tmplt_list.GetNumberOfIds()):
            tmplt_pt_list = vtk.vtkIdList()
            poly.GetCellPoints(tmplt_list.GetId(i), tmplt_pt_list)
            for j in range(tmplt_pt_list.GetNumberOfIds()):
                p_id = tmplt_pt_list.GetId(j)
                if not p_id in visited_list:
                    visited_list.append(p_id)
                    cum_dist = np.linalg.norm(np.array(points.GetPoint(node))-np.array(points.GetPoint(p_id))) + dist
                    print("dist: ", cum_dist)
                    if cum_dist < threshold:
                        pt_queue.append(p_id)
                        dist_queue.append(cum_dist)
    print("count: ", count)
    return dist_arr

def find_point_correspondence(mesh,points):
    """
    Find the point IDs of the points on a VTK PolyData
    Args:
        mesh: the PolyData to find IDs on
        points: vtk Points
    
    Returns
        IdList: list containing the IDs
    """
    IdList = [None]*points.GetNumberOfPoints()
    locator = vtk.vtkKdTreePointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    for i in range(len(IdList)):
        newPt = points.GetPoint(i)
        IdList[i] = locator.FindClosestPoint(newPt)
    return IdList
def geodesic_distance(poly, start, end):
    '''
    The mesh must be a manifold
    '''
    d_graph = vtk.vtkDijkstraGraphGeodesicPath()
    d_graph.SetInputData(poly)
    d_graph.SetStartVertex(start)
    d_graph.SetEndVertex(end)
    d_graph.Update()

    id_list = d_graph.GetIdList()
    if id_list.GetNumberOfIds() == 0:
        return float("inf")
    distance = 0.
    points = poly.GetPoints()
    for i in range(id_list.GetNumberOfIds()-1):
        i_curr = id_list.GetId(i)
        i_next = id_list.GetId(i+1)
        dist = vtk.vtkMath()
        distance += dist.Distance2BetweenPoints(points.GetPoint(i_curr), points.GetPoint(i_next))
    return distance
        
def decimation(poly, rate):
    """
    Simplifies a VTK PolyData
    Args: 
        poly: vtk PolyData
        rate: target rate reduction
    """
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(poly)
    decimate.AttributeErrorMetricOn()
    decimate.ScalarsAttributeOn()
    decimate.SetTargetReduction(rate)
    decimate.VolumePreservationOff()
    decimate.Update()
    output = decimate.GetOutput()
    #output = cleanPolyData(output, 0.)
    return output

def get_all_connected_polydata(poly):
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(poly)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    #connectivity.MarkVisitedPointIdsOn()
    connectivity.Update()
    poly = connectivity.GetOutput()
    return poly

def get_largest_connected_polydata(poly):
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(poly)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()
    poly = connectivity.GetOutput()
    return poly

def get_poly_surface_area(poly):
    mass = vtk.vtkMassProperties()
    mass.SetInputData(poly)
    mass.Update()
    return mass.GetSurfaceArea()

def vtkImageResample(image, spacing, opt):
    """
    Resamples the vtk image to the given dimenstion
    Args:
        image: vtk Image data
        spacing: image new spacing
        opt: interpolation option: linear, NN, cubic
    Returns:
        image: resampled vtk image data
    """
    reslicer = vtk.vtkImageReslice()
    reslicer.SetInputData(image)
    if opt=='linear':
        reslicer.SetInterpolationModeToLinear()
    elif opt=='NN':
        reslicer.SetInterpolationModeToNearestNeighbor()
    elif opt=='cubic':
        reslicer.SetInterpolationModeToCubic()
    else:
        raise ValueError("interpolation option not recognized")

    #size = np.array(image.GetSpacing())*np.array(image.GetDimensions())
    #new_spacing = size/np.array(dims)

    reslicer.SetOutputSpacing(*spacing)
    reslicer.Update()

    return reslicer.GetOutput()

def convertPolyDataToImageData(poly, ref_im):
    """
    Convert the vtk polydata to imagedata 
    Args:
        poly: vtkPolyData
        ref_im: reference vtkImage to match the polydata with
    Returns:
        output: resulted vtkImageData
    """
    
    ref_im.GetPointData().SetScalars(numpy_to_vtk(np.zeros(vtk_to_numpy(ref_im.GetPointData().GetScalars()).shape)))
    ply2im = vtk.vtkPolyDataToImageStencil()
    ply2im.SetTolerance(0.05)
    ply2im.SetInputData(poly)
    ply2im.SetOutputSpacing(ref_im.GetSpacing()) 
    ply2im.SetInformationInput(ref_im)
    ply2im.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(ref_im)
    stencil.ReverseStencilOn()
    stencil.SetStencilData(ply2im.GetOutput())
    stencil.Update()
    output = stencil.GetOutput()
 
    return output

def multiclass_convert_polydata_to_imagedata(poly, ref_im):
    if poly.GetPointData().GetArray('RegionId') is None:
        poly = get_all_connected_polydata(poly)
    out_im_py = np.zeros(vtk_to_numpy(ref_im.GetPointData().GetScalars()).shape)
    c = 0
    poly_i = thresholdPolyData(poly, 'RegionId', (c, c), 'point')
    while poly_i.GetNumberOfPoints() > 0:
        poly_im = convertPolyDataToImageData(poly_i, ref_im)
        poly_im_py = vtk_to_numpy(poly_im.GetPointData().GetScalars())
        mask = (poly_im_py==1) & (out_im_py==0) if c==6 else poly_im_py==1
        out_im_py[mask] = c+1
        c+=1
        poly_i = thresholdPolyData(poly, 'RegionId', (c, c), 'point')
    im = vtk.vtkImageData()
    im.DeepCopy(ref_im)
    im.GetPointData().SetScalars(numpy_to_vtk(out_im_py))
    return im

def exportSitk2VTK(sitkIm,spacing=None):
    """
    This function creates a vtk image from a simple itk image
    Args:
        sitkIm: simple itk image
    Returns:
        imageData: vtk image
import SimpleITK as sitk
    """
    if not spacing:
        spacing = sitkIm.GetSpacing()
    import SimpleITK as sitk
    import vtk
    img = sitk.GetArrayFromImage(sitkIm).transpose(2,1,0)
    vtkArray = exportPython2VTK(img)
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(sitkIm.GetSize())
    imageData.GetPointData().SetScalars(vtkArray)
    imageData.SetOrigin([0.,0.,0.])
    imageData.SetSpacing(spacing)
    matrix = build_transform_matrix(sitkIm)
    space_matrix = np.diag(list(spacing)+[1.])
    matrix = np.matmul(matrix, np.linalg.inv(space_matrix))
    matrix = np.linalg.inv(matrix)
    vtkmatrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtkmatrix.SetElement(i, j, matrix[i,j])
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(imageData)
    reslice.SetResliceAxes(vtkmatrix)
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.Update()
    imageData = reslice.GetOutput()
    #imageData.SetDirectionMatrix(sitkIm.GetDirection())

    return imageData, vtkmatrix


def load_vtk_image(fn, mode='linear'):
    """
    This function imports image file as vtk image.
    Args:
        fn: filename of the image data
    Return:
        label: label map as a vtk image
    """
    name_list = fn.split(os.extsep)
    ext = name_list[-1]

    if ext=='vti':
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(fn)
        reader.Update()
        label = reader.GetOutput()
    elif ext=='nii' or '.'.join([name_list[-2], ext])=='nii.gz':
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(fn)
        reader.Update()
        image = reader.GetOutput()
        matrix = reader.GetQFormMatrix()
        if matrix is None:
            matrix = reader.GetSFormMatrix()
        matrix.Invert()
        Sign = vtk.vtkMatrix4x4()
        Sign.Identity()
        Sign.SetElement(0, 0, -1)
        Sign.SetElement(1, 1, -1)
        M = vtk.vtkMatrix4x4()
        M.Multiply4x4(matrix, Sign, M)
        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(image)
        reslice.SetResliceAxes(M)
        if mode=='linear':
            reslice.SetInterpolationModeToLinear()
        else:
            reslice.SetInterpolationModeToNearestNeighbor()
        reslice.SetOutputSpacing(np.min(image.GetSpacing())*np.ones(3))
        reslice.Update()
        label = reslice.GetOutput()
        py_label = vtk_to_numpy(label.GetPointData().GetScalars())
        py_label = (py_label + reader.GetRescaleIntercept())/reader.GetRescaleSlope()
        label.GetPointData().SetScalars(numpy_to_vtk(py_label))
    else:
        raise IOError("File extension is not recognized: ", ext)
    return label
def vtk_write_mask_as_nifty2(mask, image_fn, mask_fn):
    origin = mask.GetOrigin()
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(image_fn)
    reader.Update()
    writer = vtk.vtkNIFTIImageWriter()
    Sign = vtk.vtkMatrix4x4()
    Sign.Identity()
    Sign.SetElement(0, 0, -1)
    Sign.SetElement(1, 1, -1)
    M = reader.GetQFormMatrix()
    if M is None:
        M = reader.GetSFormMatrix()
    M2 = vtk.vtkMatrix4x4()
    M2.Multiply4x4(Sign, M, M2)
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(mask)
    reslice.SetResliceAxes(M2)
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.Update()
    mask = reslice.GetOutput()
    mask.SetOrigin([0.,0.,0.])

    writer.SetInputData(mask)
    writer.SetFileName(mask_fn)
    writer.SetQFac(reader.GetQFac())
    q_mat = reader.GetQFormMatrix()
    writer.SetQFormMatrix(q_mat)
    s_mat = reader.GetSFormMatrix()
    writer.SetSFormMatrix(s_mat)
    writer.Write()
    return

def vtk_write_mask_as_nifty(mask,M , image_fn, mask_fn):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(image_fn)
    reader.Update()
    writer = vtk.vtkNIFTIImageWriter()
    M.Invert()
    if reader.GetQFac() == -1:
        for i in range(3):
            temp = M.GetElement(i, 2)
            M.SetElement(i, 2, temp*-1)
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(mask)
    reslice.SetResliceAxes(M)
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.Update()
    mask = reslice.GetOutput()
    mask.SetOrigin([0.,0.,0.])

    writer.SetInputData(mask)
    writer.SetFileName(mask_fn)
    writer.SetQFac(reader.GetQFac())
    q_mat = reader.GetQFormMatrix()
    writer.SetQFormMatrix(q_mat)
    s_mat = reader.GetSFormMatrix()
    writer.SetSFormMatrix(s_mat)
    writer.Write()
    return 


def write_vtk_image(vtkIm, fn, M=None):
    """
    This function writes a vtk image to disk
    Args:
        vtkIm: the vtk image to write
        fn: file name
    Returns:
        None
    """
    print("Writing vti with name: ", fn)
    if M is not None:
        M.Invert()
        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(vtkIm)
        reslice.SetResliceAxes(M)
        reslice.SetInterpolationModeToNearestNeighbor()
        reslice.Update()
        vtkIm = reslice.GetOutput()

    _, extension = os.path.splitext(fn)
    if extension == '.vti':
        writer = vtk.vtkXMLImageDataWriter()
    elif extension =='.vtk':
        writer = vtk.vtkStructuredPointsWriter()
    elif extension == '.mhd':
        writer = vtk.vtkMetaImageWriter()
    else:
        raise ValueError("Incorrect extension " + extension)
    writer.SetInputData(vtkIm)
    writer.SetFileName(fn)
    writer.Update()
    writer.Write()
    return

def appendPolyData(poly_list):
    """
    Combine two VTK PolyData objects together
    Args:
        poly_list: list of polydata
    Return:
        poly: combined PolyData
    """
    appendFilter = vtk.vtkAppendPolyData()
    for poly in poly_list:
        appendFilter.AddInputData(poly)
    appendFilter.Update()
    out = appendFilter.GetOutput() 
    return out

def cleanPolyData(poly, tol):
    """
    Cleans a VTK PolyData

    Args:
        poly: VTK PolyData
        tol: tolerance to merge points
    Returns:
        poly: cleaned VTK PolyData
    """

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.SetTolerance(tol)
    clean.PointMergingOn()
    clean.Update()

    #clean_poly = vtk.vtkRemoveDuplicatePolys()
    #clean_poly.SetInputData(clean.GetOutput())
    #clean_poly.Update()
    #poly = clean_poly.GetOutput()
    poly = clean.GetOutput()
    return poly


def load_vtk(fn, clean=True,num_mesh=1):
    poly = load_vtk_mesh(fn)    
    if clean:
        poly = cleanPolyData(poly, 0.)
        write_vtk_polydata(poly, '/Users/fanweikong/Downloads/debug.vtk')
    
    print("Loading vtk, number of cells: ", poly.GetNumberOfCells())
    poly2_l = [poly]
    for i in range(1, num_mesh):
        poly2 = vtk.vtkPolyData()
        poly2.DeepCopy(poly)
        poly2_l.append(poly2)
    poly_f = appendPolyData(poly2_l)
    print("Appending: ", poly_f.GetNumberOfCells())
    coords = vtk_to_numpy(poly_f.GetPoints().GetData())
    cells = vtk_to_numpy(poly_f.GetPolys().GetData())
    cells = cells.reshape(poly_f.GetNumberOfCells(), 4)
    cells = cells[:,1:]
    mesh = dict()
    mesh['faces'] = cells
    mesh['vertices'] = coords

    return mesh

def vtk_marching_cube(vtkLabel, bg_id, seg_id, smooth=None):
    """
    Use the VTK marching cube to create isosrufaces for all classes excluding the background
    Args:
        labels: vtk image contraining the label map
        bg_id: id number of background class
        smooth: smoothing iteration
    Returns:
        mesh: vtk PolyData of the surface mesh
    """
    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(vtkLabel)
    contour.SetValue(0, seg_id)
    contour.Update()
    mesh = contour.GetOutput()

    return mesh

def vtk_marching_cube_continuous(sdf_img, thresh):
    """
    Use the VTK marching cube to create isosrufaces 
    Args:
        sdf_img: vtkImageData of the sign distance volume
        thresh: threshold above which to extract mesh
    Returns:
        mesh: vtkPolyData of the extracted polygon mesh
    """
    contour = vtk.vtkMarchingCubes()
    contour.SetInputData(sdf_img)
    contour.SetValue(0, thresh)
    contour.Update()
    mesh = contour.GetOutput()

    return mesh

def vtk_marching_cube_multi(vtkLabel, bg_id, smooth=None):
    """
    Use the VTK marching cube to create isosrufaces for all classes excluding the background
    Args:
        labels: vtk image contraining the label map
        bg_id: id number of background class
        smooth: smoothing iteration
    Returns:
        mesh: vtk PolyData of the surface mesh
    """
    ids = np.unique(vtk_to_numpy(vtkLabel.GetPointData().GetScalars()))
    ids = np.delete(ids, np.where(ids==bg_id))

    #smooth the label map
    #vtkLabel = utils.gaussianSmoothImage(vtkLabel, 2.)

    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(vtkLabel)
    for index, i in enumerate(ids):
        print("Setting iso-contour value: ", i)
        contour.SetValue(index, i)
    contour.Update()
    mesh = contour.GetOutput()

    return mesh

def load_vtk_mesh(fileName, verbose=False):
    """
    Loads surface/volume mesh to VTK
    """
    if (fileName == ''):
        return 0
    fn_dir, fn_ext = os.path.splitext(fileName)
    if (fn_ext == '.vtk'):
        if verbose:
            print('Reading vtk with name: ', fileName)
        reader = vtk.vtkPolyDataReader()
    elif (fn_ext == '.vtp'):
        if verbose:
            print('Reading vtp with name: ', fileName)
        reader = vtk.vtkXMLPolyDataReader()
    elif (fn_ext == '.stl'):
        if verbose:
            print('Reading stl with name: ', fileName)
        reader = vtk.vtkSTLReader()
    elif (fn_ext == '.obj'):
        if verbose:
            print('Reading obj with name: ', fileName)
        reader = vtk.vtkOBJReader()
    elif (fn_ext == '.vtu'):
        if verbose:
            print('Reading vtu with name: ', fileName)
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif (fn_ext == '.pvtu'):
        if verbose:
            print('Reading pvtu with name: ', fileName)
        reader = vtk.vtkXMLPUnstructuredGridReader()
    else:
        print(fn_ext)
        raise ValueError('File extension not supported')

    reader.SetFileName(fileName)
    reader.Update()
    return reader.GetOutput()

def load_image_to_nifty(fn):
    """
    This function imports image file as vtk image.
    Args:
        fn: filename of the image data
    Return:
        label: label map as a vtk image
    """
    import SimpleITK as sitk
    ext = fn.split(os.extsep)[-1]
    if ext=='vti':
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(fn)
        reader.Update()
        label_vtk = reader.GetOutput()
        size = label_vtk.GetDimensions()
        py_arr = np.reshape(vtk_to_numpy(label_vtk.GetPointData().GetScalars()), tuple(size), order='F')
        label = sitk.GetImageFromArray(py_arr.transpose(2,1,0))
        label.SetOrigin(label_vtk.GetOrigin())
        label.SetSpacing(label_vtk.GetSpacing())
        label.SetDirection(np.eye(3).ravel())
    elif ext=='nii' or ext=='nii.gz':
        label = sitk.ReadImage(fn)
    else:
        raise IOError("File extension is not recognized: ", ext)
    return label

def exportVTK2Sitk(label_vtk):
    import SimpleITK as sitk
    # assume identity orientation
    size = label_vtk.GetDimensions()
    py_arr = np.reshape(vtk_to_numpy(label_vtk.GetPointData().GetScalars()), tuple(size), order='F')
    label = sitk.GetImageFromArray(py_arr.transpose(2,1,0))
    label.SetOrigin(label_vtk.GetOrigin())
    label.SetSpacing(label_vtk.GetSpacing())
    label.SetDirection(np.eye(3).ravel())
    return label


def exportPython2VTK(img):
    """
    This function creates a vtk image from a python array
    Args:
        img: python ndarray of the image
    Returns:
        imageData: vtk image
    """
    vtkArray = numpy_to_vtk(num_array=img.flatten('F'), deep=True, array_type=get_vtk_array_type(img.dtype))
    #vtkArray = numpy_to_vtk(img.flatten())
    return vtkArray



def write_vtk_polydata(poly, fn):
    """
    This function writes a vtk polydata to disk
    Args:
        poly: vtk polydata
        fn: file name
    Returns:
        None
    """
    print('Writing vtp with name:', fn)
    if (fn == ''):
        return 0

    _ , extension = os.path.splitext(fn)

    if extension == '.vtk':
        writer = vtk.vtkPolyDataWriter()
    elif extension == '.stl':
        writer = vtk.vtkSTLWriter()
    elif extension == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    elif extension == '.obj':
        writer = vtk.vtkOBJWriter()
    else:
        raise ValueError("Incorrect extension"+extension)
    writer.SetInputData(poly)
    writer.SetFileName(fn)
    writer.Update()
    writer.Write()
    return

def write_polydata_points(poly, fn):
    verts = vtk.vtkVertexGlyphFilter()
    verts.AddInputData(poly)
    verts.Update()
    write_vtk_polydata(verts.GetOutput(), fn)
    return

def write_numpy_points(pts, fn):
    vtkPts = vtk.vtkPoints()
    vtkPts.SetData(numpy_to_vtk(pts[:,:3]))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtkPts)
    write_polydata_points(poly, fn)
    return 


def build_transform_matrix(image):
    matrix = np.eye(4)
    matrix[:-1,:-1] = np.matmul(np.reshape(image.GetDirection(), (3,3)), np.diag(image.GetSpacing()))
    matrix[:-1,-1] = np.array(image.GetOrigin())
    return matrix 

def surface_to_image(mesh, image):
    """
    Find the corresponding pixel of the mesh vertices,
    create a new image delineate the surface for testing
    
    Args:
        mesh: VTK PolyData
        image: VTK ImageData or Sitk Image
    """
    import SimpleITK as sitk
    mesh_coords = vtk_to_numpy(mesh.GetPoints().GetData())
    if type(image) == vtk.vtkImageData:
        indices = ((mesh_coords - image.GetOrigin())/image.GetSpacing()).astype(int)

        py_im = np.zeros(image.GetDimensions())
        for i in indices:
            py_im[i[0], i[1], i[2]] = 1.

        new_image = vtk.vtkImageData()
        new_image.DeepCopy(image)
        new_image.GetPointData().SetScalars(numpy_to_vtk(py_im.flatten('F')))
    elif type(image) == sitk.Image:
        matrix = build_transform_matrix(image)
        mesh_coords = np.append(mesh_coords, np.ones((len(mesh_coords),1)),axis=1)
        matrix = np.linalg.inv(matrix)
        indices = np.matmul(matrix, mesh_coords.transpose()).transpose().astype(int)
        py_im = sitk.GetArrayFromImage(image).transpose(2,1,0)
        for i in indices:
            py_im[i[0], i[1], i[2]] = 1000.
        new_image = sitk.GetImageFromArray(py_im.transpose(2,1,0))
        new_image.SetOrigin(image.GetOrigin())
        new_image.SetSpacing(image.GetSpacing())
        new_image.SetDirection(image.GetDirection())
    return new_image

def surface_to_maxpool_image(mesh, image, num):
    """
    Find the corresponding pixel of the mesh vertices on 
    a maxpooled image

    Args:
        mesh: VTK PolyData
        image: sitk image
        num: number of max pool applied
    Return:
        new_image: image with matched vertices 
    """
    import SimpleITK as sitk
    mesh_coords = vtk_to_numpy(mesh.GetPoints().GetData())

    matrix = build_transform_matrix(image)
    mesh_coords = np.append(mesh_coords, np.ones((len(mesh_coords),1)),axis=1)
    matrix = np.linalg.inv(matrix)
    indices = np.matmul(matrix, mesh_coords.transpose()).transpose().astype(int)
    #MAX POOL IMAGE
    py_im = sitk.GetArrayFromImage(image).transpose(2,1,0)
    for i in range(num):
        py_im = max_pool_image(py_im)
    for i in indices:
        py_im[i[0], i[1], i[2]] = 1000.
    new_image = sitk.GetImageFromArray(py_im.transpose(2,1,0))
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetSpacing(np.array(image.GetSpacing())*(2**num))
    new_image.SetDirection(image.GetDirection())
    return new_image

def get_point_normals(poly):
    norms = vtk.vtkPolyDataNormals()
    norms.SetInputData(poly)
    norms.ComputePointNormalsOn()
    norms.ComputeCellNormalsOff()
    norms.ConsistencyOn()
    norms.SplittingOff()
    norms.Update()
    poly = norms.GetOutput()
    pt_norm = poly.GetPointData().GetArray("Normals")
    return vtk_to_numpy(pt_norm)

def get_surface_normals(poly):
    norms = vtk.vtkPolyDataNormals()
    norms.SetInputData(poly)
    norms.ComputePointNormalsOn()
    norms.ComputeCellNormalsOff()
    norms.ConsistencyOn()
    norms.SplittingOff()
    #norms.FlipNormalsOn()
    #norms.AutoOrientNormalsOn()
    norms.Update()
    poly = norms.GetOutput()
    return poly

def thresholdPolyData(poly, attr, threshold, mode):
    """
    Get the polydata after thresholding based on the input attribute
    Args:
        poly: vtk PolyData to apply threshold
        atrr: attribute of the cell array
        threshold: (min, max)
    Returns:
        output: resulted vtk PolyData
    """
    surface_thresh = vtk.vtkThreshold()
    surface_thresh.SetInputData(poly)
    print(f'threshold: {threshold}, and reference to threshold: {*threshold,}')
    surface_thresh.Between(*threshold)
    if mode=='cell':
        surface_thresh.SetInputArrayToProcess(0, 0, 0,
            vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, attr)
    else:
        surface_thresh.SetInputArrayToProcess(0, 0, 0,
            vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, attr)
    surface_thresh.Update()
    surf_filter = vtk.vtkDataSetSurfaceFilter()
    surf_filter.SetInputData(surface_thresh.GetOutput())
    surf_filter.Update()
    return surf_filter.GetOutput()

def eraseBoundary(labels, pixels, bg_id):
    """
    Erase anything on the boundary by a specified number of pixels
    Args:
        labels: python nd array
        pixels: number of pixel width to erase
        bg_id: id number of background class
    Returns:
        labels: editted label maps
    """
    x,y,z = labels.shape
    labels[:pixels,:,:] = bg_id
    labels[-pixels:,:,:] = bg_id
    labels[:,:pixels,:] = bg_id
    labels[:,-pixels:,:] = bg_id
    labels[:,:,:pixels] = bg_id
    labels[:,:,-pixels:] = bg_id
    return labels

def convert_to_surfs(seg, new_spacing=[1.,1.,1.], target_node_num=2048, bound=False):
    import SimpleITK as sitk
    py_seg = sitk.GetArrayFromImage(seg)
    py_seg = eraseBoundary(py_seg, 1, 0)
    labels = np.unique(py_seg)
    for i, l in enumerate(labels):
        py_seg[py_seg==l] = i
    seg2 = sitk.GetImageFromArray(py_seg)
    seg2.CopyInformation(seg)

    seg_vtk,_ = exportSitk2VTK(seg2)
    seg_vtk = vtkImageResample(seg_vtk,new_spacing,'NN')
    poly_l = []
    for i, _ in enumerate(labels):
        if i==0:
            continue
        p = vtk_marching_cube(seg_vtk, 0, i)
        p = smooth_polydata(p, iteration=50)
        rate = max(0., 1. - float(target_node_num)/float(p.GetNumberOfPoints()))
        p = decimation(p, rate)
        arr = np.ones(p.GetNumberOfPoints())*i
        arr_vtk = numpy_to_vtk(arr)
        arr_vtk.SetName('RegionId')
        p.GetPointData().AddArray(arr_vtk)
        poly_l.append(p)
    poly = appendPolyData(poly_l)
    if bound:
        poly = bound_polydata_by_image(seg_vtk, poly, 1.5)
    return poly

def write_vtu(mesh, fn):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputData(mesh)
    writer.SetFileName(fn)
    writer.Update()
    writer.Write()
    return 
 
def create_cell(elem):
    tetra = vtk.vtkTetra()
    ids = tetra.GetPointIds()
    ids.SetId(0, elem[0])
    ids.SetId(1, elem[1])
    ids.SetId(2, elem[2])
    ids.SetId(3, elem[3])
    return tetra

def create_vtu_mesh(node, face):
    mesh = vtk.vtkUnstructuredGrid()
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(node))
    mesh.SetPoints(vtk_pts)
    cells = vtk.vtkCellArray()
    print(node.shape, np.max(face), np.min(face))
    for i in range(face.shape[0]):
        tetra = create_cell((face[i, :]).astype(int))
        cells.InsertNextCell(tetra)
    mesh.SetCells(vtk.VTK_TETRA, cells)
    return mesh


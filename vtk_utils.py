import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
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
    decimate.SetTargetReduction(rate)
    decimate.VolumePreservationOn()
    decimate.Update()
    return decimate.GetOutput()

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

def load_vtk_image(fn):
    """
    This function imports image file as vtk image.
    Args:
        fn: filename of the image data
    Return:
        label: label map as a vtk image
    """
    _, ext = fn.split(os.extsep, 1)

    if ext=='vti':
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(fn)
        reader.Update()
        label = reader.GetOutput()
    elif ext=='nii' or ext=='nii.gz':
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
        reslice.SetInterpolationModeToLinear()
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

def load_vtk_mesh(fileName):
    """
    Loads surface/volume mesh to VTK
    """
    if (fileName == ''):
        return 0
    fn_dir, fn_ext = os.path.splitext(fileName)
    if (fn_ext == '.vtk'):
        print('Reading vtk with name: ', fileName)
        reader = vtk.vtkPolyDataReader()
    elif (fn_ext == '.vtp'):
        print('Reading vtp with name: ', fileName)
        reader = vtk.vtkXMLPolyDataReader()
    elif (fn_ext == '.stl'):
        print('Reading stl with name: ', fileName)
        reader = vtk.vtkSTLReader()
    elif (fn_ext == '.vtu'):
        print('Reading vtu with name: ', fileName)
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif (fn_ext == '.pvtu'):
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
    _, ext = fn.split(os.extsep, 1)
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
    elif extension == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
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

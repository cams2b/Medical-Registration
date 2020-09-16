import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt 
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
import plotly
import os
import view
from IPython.display import Image 


def contours(img):
	"""
	Currently using the value 400 for thresholding contours.
	This number is based upon the Hounsfield scale where 400 is
	the average value for bone in CT scan.
	https://pyscience.wordpress.com/2014/09/11/surface-extraction-creating-a-mesh-from-pixel-data-using-python-and-vtk/

	"""
	r =sitk.GetArrayFromImage(img)
	# Find contours at a constant value of 0.8
	contours = measure.find_contours(r, 400)

	print(len(contours))

	# Display the image and plot all contours found
	fig, ax = plt.subplots()
	ax.imshow(r, cmap=plt.cm.gray)

	
	for contour in contours:
		ax.plot(contour[:, 1], contour[:, 0], linewidth = 2)

	ax.axis('image')
	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()



def cloud(PathDicom):
	#r = sitk.GetArrayFromImage(img)
	#VTK_data = numpy_support.numpy_to_vtk(num_array=r.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

	reader = vtk.vtkDICOMImageReader()
	reader.SetDirectoryName(PathDicom)
	reader.Update()


	_extent = reader.GetDataExtent()
	ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

	# Load spacing values
	ConstPixelSpacing = reader.GetPixelSpacing()
	ArrayDicom = vtkImageToNumPy(reader.GetOutput(), ConstPixelDims)
	#view.arr_show(np.flipud(np.rot90(ArrayDicom[:, 225, :])))



	threshold = vtk.vtkImageThreshold()
	threshold.SetInputConnection(reader.GetOutputPort())
	threshold.ThresholdByLower(400)  # remove all soft tissue
	threshold.ReplaceInOn()
	threshold.SetInValue(0)  # set all values below 400 to 0
	threshold.ReplaceOutOn()
	threshold.SetOutValue(1)  # set all values above 400 to 1
	threshold.Update()

	ArrayDicom = vtkImageToNumPy(threshold.GetOutput(), ConstPixelDims)
	#view.arr_show(np.flipud(np.rot90(ArrayDicom[:, 225, :])))

	if True:
		print('=============================')
		return np.flipud(np.rot90(ArrayDicom[:, 225, :]))

	dmc = vtk.vtkDiscreteMarchingCubes()
	dmc.SetInputConnection(threshold.GetOutputPort())
	dmc.GenerateValues(1, 1, 1)
	dmc.Update()


	mapper = vtk.vtkPolyDataMapper()
	mapper.SetInputConnection(dmc.GetOutputPort())

	actor = vtk.vtkActor()
	actor.SetMapper(mapper)

	renderer = vtk.vtkRenderer()
	renderer.AddActor(actor)
	renderer.SetBackground(1.0, 1.0, 1.0)

	camera = renderer.MakeCamera()
	camera.SetPosition(-500.0, 245.5, 122.0)
	camera.SetFocalPoint(301.0, 245.5, 122.0)
	camera.SetViewAngle(30.0)
	camera.SetRoll(-90.0)
	renderer.SetActiveCamera(camera)
	vtk_show(renderer, 600, 600)



def vtkImageToNumPy(image, pixelDims):
    pointData = image.GetPointData()
    arrayData = pointData.GetArray(0)
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    ArrayDicom = ArrayDicom.reshape(pixelDims, order='F')
    
    return ArrayDicom

def vtk_show(renderer, width=400, height=300):
    """
    Takes vtkRenderer instance and returns an IPython Image with the rendering.
    """
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(width, height)
    renderWindow.Render()
     
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()
     
    writer = vtk.vtkPNGWriter()
    writer.SetWriteToMemory(1)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    data = str((writer.GetResult()))
    
    return Image(data)
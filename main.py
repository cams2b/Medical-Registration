import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pydicom
from skimage import morphology
import view
import image
import vision as v
import lung_register as lung
import register as r
def main():
	print('testing')

	ct_path = r'C:\Users\cbeeche\Desktop\Fall_2020\Research\2018-04-20\8_125__mm__volume.nii'
	xray_path = r'C:\Users\cbeeche\Desktop\Fall_2020\Research\2018-04-20\5\1.2.840.113619.2.289.3.296546961.13.1523755933.403.1.dcm'
	dicom_path = r'C:\Users\cbeeche\Desktop\Fall_2020\Research\2018-04-20\8'
	eye_ct_path = r'C:\Users\cbeeche\Desktop\Fall_2020\Research\Test\img-eyeball\HB039126OAV_00230_2014-03-29_2_img.nii'


	head = sitk.ReadImage(eye_ct_path, sitk.sitkFloat32)
	print(head.GetDepth())
	eye_arr = sitk.GetArrayFromImage(head)

	eye_slice1 = sitk.GetImageFromArray(eye_arr[30, :, :])
	eye_slice2 = sitk.GetImageFromArray(eye_arr[35, :, :])
	
	xray = sitk.ReadImage(xray_path, sitk.sitkFloat32)
	ct = sitk.ReadImage(ct_path, sitk.sitkFloat32)
	xray = image.norm_xray(xray)
	

	## Slices for ct chest.
	ct_slice1 = image.slice(ct, 225)
	ct_slice2 = image.slice(ct, 200)




	

	test = image.shift_random(eye_slice2)
	test = image.rotate_random(test)
	# =========================================================
	fixed_image = eye_slice1
	moving_image = test
	view.display_images(fixed_image, moving_image)
	rsample = sitk.Resample(moving_image, fixed_image, sitk.Transform())
	view.display_images(fixed_image, rsample)

	print("Displayed resample")
	# Centered 2D affine transform and show the resampled moving_image using this transform.
	registration_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.AffineTransform(2), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
	rsample2 = sitk.Resample(moving_image, fixed_image, registration_transform)
	view.itk_show(rsample2, 'initial affine')


	# Register using 2D affine initial transform that is overwritten
	# and show the resampled moving_image using this transform.
	r.multires_registration(fixed_image, moving_image, registration_transform)
	view.display_images(fixed_image, sitk.Resample(moving_image, fixed_image, registration_transform))



	## ==== Angular registration ====

	



if __name__=="__main__":
	main()
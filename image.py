import SimpleITK as sitk
import numpy as np
import scipy.ndimage







def norm_intensity(fixed, moving):
    metrics = sitk.MinimumMaximumImageFilter()

    metrics.Execute(fixed)
    fix_min = metrics.GetMinimum()
    fix_max = metrics.GetMaximum()

    metrics.Execute(moving)
    mov_min = metrics.GetMinimum()
    mov_max = metrics.GetMaximum()

    minV = min(fix_min, mov_min)
    maxV = max(fix_max, mov_max)
    ## Normalize fixed image
    fix_arr = sitk.GetArrayFromImage(fixed)
    fix_arr = np.subtract(fix_arr, minV)
    fix_arr = np.divide(fix_arr, maxV)
    fixed_image = sitk.GetImageFromArray(fix_arr)

    ## Normalize moving image
    mov_arr = sitk.GetArrayFromImage(moving)
    mov_arr = np.subtract(mov_arr, minV)
    mov_arr = np.divide(mov_arr, maxV)
    moving_image = sitk.GetImageFromArray(mov_arr)

    #view.display_images(fixed_image, moving_image)

    return (fixed_image, moving_image)

    metrics.Execute(moving)
    mov_max = metrics.GetMaximum()

    maxV = max(fix_max, mov_max)
    fix_arr = sitk.GetArrayFromImage(fixed)
    mov_arr = sitk.GetArrayFromImage(moving)

    fix_arr = np.divide(fix_arr, maxV)
    mov_arr = np.divide(mov_arr, maxV)

    fixed = sitk.GetImageFromArray(fix_arr)
    moving = sitk.GetImageFromArray(mov_arr)
    print("=== Fixed values ===")
    metrics.Execute(fixed)
    print("Min: ", metrics.GetMinimum())
    print("Max: ", metrics.GetMaximum())

    print("=== Moving values ===")
    metrics.Execute(moving)
    print("Min: ", metrics.GetMinimum())
    print("Max: ", metrics.GetMaximum())
    return (fixed, moving)


def norm_xray(image):
    ## Extract one slice allowing image to be 2D
    moving_slice = sitk.GetArrayFromImage(image)
    moving_slice = moving_slice[0, :, :]
    moving_imag = sitk.GetImageFromArray(moving_slice)
    return moving_imag


def slice(image, num):
    fixed_array = sitk.GetArrayFromImage(image)
    sub_array = fixed_array[:, num, :]
    ## Flip array
    nda = sub_array
    sub_array = np.flipud(nda)
    fixed_image = sitk.GetImageFromArray(sub_array)
    return fixed_image

def adjust(fixed, moving, spacing):
    """

    :param image:
    :param spacing: spacing adjustment for X-Ray
    :return: X-Ray image with equal dimensions as CT-scan
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    ## Adjusted output spacing to acheive alignment
    resampler.SetOutputSpacing([spacing, spacing])
    resampler.SetOutputOrigin([10, 10])
    resampled = resampler.Execute(moving)
    # display_images(fixed_image, resampled)
    moving_image = resampled
    return moving_image

def flip(image):
    to_flip = sitk.GetArrayFromImage(image)
    nda = to_flip
    to_flip = np.flipud(nda)
    return sitk.GetImageFromArray(to_flip)


def angle(img, angle, slice):
    print(img.GetSize())
    nda = sitk.GetArrayFromImage(img)

    # axes = (0,2) is front facing slice

    array_rotated = scipy.ndimage.interpolation.rotate(nda, angle = angle)
    print(array_rotated.shape)
    sz, sy, sx = array_rotated.shape

    arr = array_rotated[:, slice, :]

    print(sy// 2)
    return sitk.GetImageFromArray(arr)



## Image preprocessing

def guided_filter(img):

    guide = sitk.CurvatureFlowImageFilter()
    return guide.Execute(img)


def shift_random(img):
    """
    Arguments: 2D image
    Rotate an image by a random angle to allow for registration testing and robustness

    """

    nda = sitk.GetArrayFromImage(img)
    array_shifted = scipy.ndimage.shift(nda, 20)

    return sitk.GetImageFromArray(array_shifted)

def rotate_random(img):

    nda = sitk.GetArrayFromImage(img)

    array_rotated = scipy.ndimage.rotate(nda, 5)
    return sitk.GetImageFromArray(array_rotated)
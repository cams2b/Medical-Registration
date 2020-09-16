import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import image
import view

metric_values = []
iterations = []


result = 0
match =None
fixed = None


def first_2d3d(fixed_images, moving):

    result = 0
    for i in range(180, 220):
        global metric_values
        metric_values.clear()
        fixed_image = image.slice(fixed_images, i)
        fixed_image = sitk.Normalize(fixed_image)

        (res, value) = third_registration(fixed_image, moving)

    if result == 0 or value < result:
        result = value
        # match = res
        fixed = fixed_image
    return fixed_image, res, metric_values






def first_registration(fixed, moving):

    print("Registering utilizing the first registration method")

    initial_transform = sitk.CenteredTransformInitializer(fixed,
                                                          moving,
                                                          sitk.Euler2DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    moving_resampled = sitk.Resample(moving, fixed, initial_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())

    R = sitk.ImageRegistrationMethod()

    ## Similarity metric settings.
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)

    R.SetInterpolator(sitk.sitkLinear)

    ## Optimizer settings.
    R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6,
                                    convergenceWindowSize= 10)
    R.SetOptimizerScalesFromPhysicalShift()

    ## Setup for the multiresolution framework.
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    R.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(R))

    R.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = R.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                                sitk.Cast(moving, sitk.sitkFloat32))

  

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(1)
    resampler.SetTransform(final_transform)

    out = resampler.Execute(moving)

    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)


    display_plot()
    return cimg, R.GetMetricValue()



def sec_registration(fixed, moving):
    R = sitk.ImageRegistrationMethod()

    ## Set performance metric
    R.SetMetricAsJointHistogramMutualInformation()

    R.SetOptimizerAsGradientDescentLineSearch(learningRate = 1.0,
                                              numberOfIterations = 200,
                                              convergenceMinimumValue = 1e-5,
                                              convergenceWindowSize = 5)

    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))

    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(R))

    final_transform = R.Execute(fixed, moving)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(1)
    resampler.SetTransform(final_transform)

    out = resampler.Execute(moving)

    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)

    



    return cimg

def third_registration(fixed, moving):
    transformDomainMeshSize = [8] * moving.GetDimension()

    tx = sitk.BSplineTransformInitializer(fixed,transformDomainMeshSize)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 100)
    R.SetInitialTransform(tx, True)
    R.SetInterpolator(sitk.sitkLinear)
    #R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    R.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(R))


    outTx = R.Execute(fixed, moving)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
    return cimg, R.GetMetricValue()


def fourth_registration(fixed, moving):

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(learningRate = 2.0,
                                               minStep = 1e-4,
                                               numberOfIterations = 500,
                                               gradientMagnitudeTolerance=1e-8)
    R.SetOptimizerScalesFromIndexShift()

    tx = sitk.CenteredTransformInitializer(fixed, moving,
                                           sitk.Similarity2DTransform())
    R.SetInitialTransform(tx)

    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(R))

    outTx = R.Execute(fixed, moving)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(1)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)

    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)

    display_plot()
    return cimg



    

    
    



def plot_values(R):
    global metric_values, iterations
    metric_values.append(R.GetMetricValue())

def display_plot():
    global metric_values
    plt.plot(metric_values)
    plt.ylabel('Performance Metrics')
    plt.show()

def command_iteration(method):
    print("{0:3} = {1:7.5f} : {2}".format(method.GetOptimizerIteration(),
                                          method.GetMetricValue(),
                                          method.GetOptimizerPosition()))
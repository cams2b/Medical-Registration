import cv2
import numpy as np 
import SimpleITK as sitk 
import matplotlib.pyplot as plt 
import math

## https://discourse.itk.org/t/multiresolution-registration-with-2d-affine-transformation-on-pairs-of-2d-images/3096


## The basic components of a typical registration framework are two input images, a transform, a metric, an interpolator and an optimizer.
# The transform component T (X) represents the spatial mapping of points from the fixed image space to points in the moving image space
# The interpolator is used to evaluate moving image intensities at non-grid positions
# The metric component provides a measure of how well the fixed image is matched by the transformed moving image
# This measure forms a quantitative criterion to be optimized by the optimizer over the search space defined by the parameters of the transform
#def novel_register(fixed, moving):

metric_values = []


def multires_registration(fixed_image, moving_image, initial_transform):    
    # Initialize registration method
    registration_method = sitk.ImageRegistrationMethod()


    # Set interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Set metric as Mean squares
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.01)
    #registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0, numberOfIterations=200, convergenceMinimumValue=1e-6, convergenceWindowSize=5)
    registration_method.SetOptimizerScalesFromPhysicalShift() 
    registration_method.SetInitialTransform(initial_transform, inPlace=True)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))
    registration_method.Execute(fixed_image, moving_image)
    
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    display_plot()



def plot_values(registration_method):
    global metric_values, iterations
    cur = registration_method.GetMetricValue()
    cur = math.sqrt(cur)
    metric_values.append(cur)

def display_plot():
    global metric_values
    plt.plot(metric_values)
    plt.ylabel('Root Mean squares')
    plt.show()
'''
LearningKit.Utilities.Math

Collection of Useful Mathematical Functions

Author:  Marissa Dominijanni
Version: v1.1 (2017.12.09)
'''
# Outside Imports
import numpy as np
# LearningKit Imports
from LearningKit.Utilities.Tests import IsNone

'''
Normalize : Normalizes a NumPy Array of Data to a Range
Parameters:
    Data    : Data to Normalize
    NormMin : New Minimum Value
    NormMax : New Maximum Value
    DataMin : Minimum Value of the Data [Default: Minimum Value in Data]
    DataMax : Maximum Value of the Data [Default: Maximum Value in Data]
Returns:
    NumPy Array of the Same Shape as Data with Normalized Values
'''
def Normalize(Data, NormMin, NormMax, DataMin=None, DataMax=None):
        if IsNone(DataMin):
            DataMin = np.amin(Data)
        if IsNone(DataMax):
            DataMax = np.amax(Data)
        return np.add(np.multiply(np.subtract(NormMax, NormMin), np.divide(np.subtract(Data, DataMin), np.subtract(DataMax, DataMin))), NormMin)
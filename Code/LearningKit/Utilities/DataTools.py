'''
LearningKit.Utilities.DataTools

Collection of Functions for Helping Manipulate NumPy Data

Author:  Marissa Dominijanni
Version: v1.1 (2017.12.09)
'''
# Outside Imports
import numpy as np
# LearningKit Imports
from LearningKit.Utilities.Tests import IsNone

'''
CatOrAssign : Concatenates Two NumPy Arrays, or Returns A1 if A0 is NoneType
Parameters:
    A0   : None or NumPy Array (First Concat Element)
    A1   : NumPy Array (Second Concat Element)
    Axis : Axis Along which to Concatenate [Default: 0]
Returns:
    Concatenation of Two NumPy Arrays
'''
def CatOrAssign(A0, A1, Axis=0):
    if IsNone(A0):
        return A1
    else:
        return np.concatenate([A0,A1], axis=Axis)
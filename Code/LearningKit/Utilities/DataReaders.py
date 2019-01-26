'''
LearningKit.Utilities.DataReaders

Collection of Methods for Reading Datasets

Author:  Marissa Dominijanni
Version: v1.1 (2017.12.09)
'''
# Python Imports
import csv
import pickle
from collections import namedtuple
# Outside Imports
import numpy as np
import scipy.io as sio
# LearningKit Imports
import LearningKit.Utilities.Math as LKMath
from LearningKit.Utilities.DataTools import CatOrAssign
from LearningKit.Utilities.Records import MkDataPair, MkDataQuad
from LearningKit.Utilities.Tests import IsSequence
'''
CSV : Reads a CSV File into NumPy Arrays
Parameters:
    Filename    : Path of the CSV File
    LabelPos    : Position Option of the Label
                 'First' : Label is in the First Column
                 'Last'  : Label is in the Last Column
                 (Other) : Dataset is Unlabeled
    HasHeader   : If the Top Row is a Header
    FeatureDT   : NumPy DType for Features [Default: np.float32]
    ClassDT     : NumPy DType for Classes [Default: np.float32]
    Normalize   : If the Dataset Should be Normalized [Default: False]
    NormMin     : New Minimum Value [Default: 0]
    NormMax     : New Maximum Value [Default: 1]
    DataMin     : Minimum Value of the Data [Default: Minimum Observed in Features]
    DataMax     : Maximum Value of the Data [Default: Maximum Observed in Features]
    MemberShape : Shape of Design for Each Member (None -> No Reshaping) [Default: None]
    Onehot      : If the Labels Should be Made Onehot [Default: False]
    LabelDim    : Specify the Dimensionality of the Label [Default: 0]
                  <2 : Label Remains Condensed
                  >1 : Label is Made Onehot with Specified Length
Returns:
    DataPair Record with Design and Labels Tensors
'''
def CSV(Filename, LabelPos, HasHeader, FeatureDT=np.float32, ClassDT=np.float32, Normalize=False, NormMin=0, NormMax=1, DataMin=None, DataMax=None, MemberShape=None, LabelDim=0):
    design = []
    labels = []
    with open(Filename, 'r') as file:
        reader = csv.reader(file)
        # Skip Header if One Exists
        if HasHeader:
            next(reader, None)
        # For Each Row
        for row in reader:
            # Build Data Based on Label Position
            if LabelPos == 'First':
                data = np.asarray(list(map(int, row[1:])), dtype=FeatureDT)
                label = row[0]
            elif LabelPos == 'Last':
                data = np.asarray(list(map(int, row[:-1])), dtype=FeatureDT)
                label = row[-1]
            else:
                data = np.asarray(list(map(int, row)), dtype=FeatureDT)
                label = None
            # Normalize iff Specified
            if Normalize:
                data = LKMath.Normalize(data, NormMin, NormMax, DataMin, DataMax)
            # Reshape iff Specified
            if MemberShape != None:
                data = np.reshape(data, MemberShape)
            # Add to Design
            design.append(data)
            # Add to Labels iff Appropriate and Make Onehot iff Specified
            if label != None:
                if LabelDim < 2:
                    labels.append(np.zeros(LabelDim, dtype=ClassDT))
                    labels[-1][int(label)] = 1
                else:
                    labels.append(int(label))
    design = np.array(design, dtype=FeatureDT)
    labels = np.array(labels, dtype=ClassDT)
    return MkDataPair(design, labels)

'''
Cifar10 : Reads Python Formatted CIFAR-10 Into NumPy Arrays
Parameters:
    TrainingPaths : Paths to the Training Pickle Files (Either a List of Paths or a Single Path)
    TestingPath   : Path to the Testing Pickle File
    FeatureDT     : NumPy DataType to Save As [Default: np.float32]
    ClassDT       : NumPy DataType to Save As [Default: np.float32]
    Normalize     : If the Dataset Should be Normalized [Default: False]
    NormMin       : New Minimum Value [Default: 0]
    NormMax       : New Maximum Value [Default: 1]
    AutoReshape   : Reshape Each Feature Vector to Match True Shape
    LabelDim      : Specify the Dimensionality of the Label [Default: 0]
                    <10 : Label Remains Condensed
                    10  : Label is Made Standard Onehot
                    >10 : Label is Made Onehot with Trailing Zeros
    ColorMode     : Specify the Color Conversion Information [Default: 'RGB']
                    'RGB'       : Leave Colors as Original
                    'HumanGrey' : Converts to Greyscale as a Human Would See It
                    'RobotGrey' : Converts to Greyscale Using Standard Average
Returns:
    DataQuad with Training and Testing Designs and Labels
'''
def Cifar10(TrainingPaths, TestingPath, FeatureDT=np.float32, ClassDT=np.float32, Normalize=False, NormMin=0, NormMax=1, AutoReshape=False, LabelDim=0, ColorMode='RGB'):
    # Helper Function to Read Pickle Into Dictionary
    def CifarFromPath(Path):
        with open(Path, 'rb') as file:
            ddict = pickle.load(file, encoding='bytes')
        fdata = namedtuple('CifarFile', ['Design', 'Labels'])
        fdata.Design = ddict[b'data']
        fdata.Labels = ddict[b'labels']
        return fdata
    # Datastructures to Fill
    trDesign = None
    trLabels = None
    teDesign = None
    teLabels = None
    # Read Training Files
        # Handle Single File Condition
    if not IsSequence(TrainingPaths):
        TrainingPaths = [TrainingPaths]
    # Iterate Over Files
    for path in TrainingPaths:
        trData = CifarFromPath(path)
        trDesign = CatOrAssign(trDesign, np.reshape(trData.Design, [-1, 32, 32, 3]))
        trLabels = CatOrAssign(trLabels, np.asarray(trData.Labels))
    # Read Testing File
    teData = CifarFromPath(TestingPath)
    teDesign = np.reshape(teData.Design, [-1, 32, 32, 3])
    teLabels = np.asarray(teData.Labels)
    # Apply ColorMode
    if ColorMode == 'HumanGrey':
        lR = np.broadcast_to(np.array([0.299]), [32, 32])
        lG = np.broadcast_to(np.array([0.587]), [32, 32])
        lB = np.broadcast_to(np.array([0.114]), [32, 32])
        trDesign = np.sum(trDesign*np.broadcast_to(np.stack([lR, lG, lB], axis=-1), trDesign.shape), axis=-1)
        teDesign = np.sum(teDesign*np.broadcast_to(np.stack([lR, lG, lB], axis=-1), teDesign.shape), axis=-1)
    if ColorMode == 'RobotGrey':
        lR = np.broadcast_to(np.array([1/3]), [32, 32])
        lG = np.broadcast_to(np.array([1/3]), [32, 32])
        lB = np.broadcast_to(np.array([1/3]), [32, 32])
        trDesign = np.sum(trDesign*np.broadcast_to(np.stack([lR, lG, lB], axis=-1), trDesign.shape), axis=-1)
        teDesign = np.sum(teDesign*np.broadcast_to(np.stack([lR, lG, lB], axis=-1), teDesign.shape), axis=-1)
    # Linearize iff Specified
    if not AutoReshape:
        trDesign = np.reshape(trDesign, [trDesign.shape[0], -1])
        teDesign = np.reshape(teDesign, [teDesign.shape[0], -1])
    # Normalize iff Specified
    if Normalize:
        trDesign = LKMath.Normalize(trDesign, NormMin, NormMax, 0, 255)
        teDesign = LKMath.Normalize(teDesign, NormMin, NormMax, 0, 255)
    # Onehot Expansion
    if LabelDim >= 10:
        trOnehot = np.zeros([trLabels.shape[0], LabelDim])
        trOnehot[np.arange(trLabels.shape[0]), trLabels] = 1
        trLabels = trOnehot
        teOnehot = np.zeros([teLabels.shape[0], LabelDim])
        teOnehot[np.arange(teLabels.shape[0]), teLabels] = 1
        teLabels = teOnehot
    # Type Conversion and Build Return Value
    return MkDataQuad([trDesign.astype(FeatureDT), trLabels.astype(ClassDT)], [teDesign.astype(FeatureDT), teLabels.astype(ClassDT)])

'''
SVHN32 : Reads MATLAB Formatted SVHN-32 NumPy Arrays
Parameters:
    TrainingPath : Paths to the Training Pickle Files
    TestingPath  : Path to the Testing Pickle File
    FeatureDT    : NumPy DataType to Save As [Default: np.float32]
    ClassDT      : NumPy DataType to Save As [Default: np.float32]
    LabelDim     : Dimensionality of the Label (<10 = Don't Make Onehot, 10 = Standard Onehot, >10 = Onehot with Trailing Zeros
    Normalize    : If the Dataset Should be Normalized [Default: False]
    NormMin      : New Minimum Value [Default: 0]
    NormMax      : New Maximum Value [Default: 1]
    LabelDim     : Specify the Dimensionality of the Label [Default: 0]
                   <10 : Label Remains Condensed
                   10  : Label is Made Standard Onehot
                   >10 : Label is Made Onehot with Trailing Zeros
'''
def SVHN32(TrainingPath, TestingPath, FeatureDT=np.float32, ClassDT=np.float32, Normalize=False, NormMin=None, NormMax=None, LabelDim=0):
    # Read Files
    trData = sio.loadmat(TrainingPath)
    teData = sio.loadmat(TestingPath)
    # Type Conversions
    trDesign = np.rollaxis(trData['X'].astype(FeatureDT), 3)
    teDesign = np.rollaxis(teData['X'].astype(FeatureDT), 3)
    trLabels = trData['y'].astype(np.int32).flatten()
    teLabels = teData['y'].astype(np.int32).flatten()
    # Zero-Align Labels
    trLabels = trLabels - 1
    teLabels = teLabels - 1
    # Normalize iff Specified
    if Normalize:
        trDesign = LKMath.Normalize(trDesign, NormMin, NormMax, 0, 255)
        teDesign = LKMath.Normalize(teDesign, NormMin, NormMax, 0, 255)
    # Onehot Expansion
    if LabelDim >= 10:
        trOnehot = np.zeros([trLabels.shape[0], LabelDim])
        trOnehot[np.arange(trLabels.shape[0]), trLabels] = 1
        trLabels = trOnehot
        teOnehot = np.zeros([teLabels.shape[0], LabelDim])
        teOnehot[np.arange(teLabels.shape[0]), teLabels] = 1
        teLabels = teOnehot
    # Type Conversion and Build Return Value
    return MkDataQuad([trDesign, trLabels.astype(ClassDT)], [teDesign, teLabels.astype(ClassDT)])
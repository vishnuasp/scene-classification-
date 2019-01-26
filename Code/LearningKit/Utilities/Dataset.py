'''
LearningKit.Utilities.Dataset

Classes and Methods for Working With NumPy-Based Data

Author:  Marissa Dominijanni
Version: v1.0 (2017.12.08)
'''
# Outside Imports
import numpy as np
# LearningKit Imports
import LearningKit.Utilities.Math as LKMath
from LearningKit.Utilities.Records import MkBatchSet
from LearningKit.Utilities.Tests import IsNone, IsNotNone

'''
Dataset : Class Which Contains and Makes Accessible NumPy Data
Methods:
    Batch          : Gets the Next Batch
    Valid          : If the Dataset has a Next Batch
    Reinitialize   : Resets the Dataset
    BroadcastLabel : Broadcast a Single Label to All Members
    Skim           : Produce a Second Dataset by Removing Number of Each Class
    ReBatch        : Changes the Batch Size of the Dataset
Properties:
    BatchCount : Current Batch Sequence Number
    Design     : Design Tensor Being Drawn From
    Labels     : Labels Tensor Being Drawn From
    Random     : If the Draw Order Should be Randomized
    SetSz      : Number of Members in Dataset
    LinDex     : (Internal) Sequence Index
    ROperm     : (Internal) Randomized Ordering
    SOperm     : (Internal) Sequential Ordering
'''
class Dataset():
    '''
    Dataset Constructor
    Parameters:
        Design    : Tensor of Features (First Dimension -> Members)
        Labels    : Tensor of Labels (First Dimension -> Members)
        Broadcast : If All Data Has the Same Label and it Should Be Broadcast
        Random    : If Selection of Members Should be Randomly Selected
        BatchSz   : Size of the Batch (Non-Positive Integer -> Complete Batch) [Default: 0]
    Returns:
        New Dataset Object with Specified Parameters
    '''
    def __init__(self, Design, Labels, Broadcast, Random, BatchSz=0):
        self.BatchCount = 0
        self.Design = Design
        if IsNotNone(Labels) and not Broadcast:
            assert(Design.shape[0] == Labels.shape[0])
        if IsNotNone(Labels) and Broadcast:
            labelEx = [Design.shape[0]]
            labelEx.extend([1]*len(Labels.shape))
            self.Labels = np.tile(Labels, labelEx)
        else:
            self.Labels = Labels
        self.Random = Random
        if BatchSz < 1:
            self.BatchSz = Design.shape[0]
        else:
            self.BatchSz = BatchSz
        self.SetSz = Design.shape[0]
        self.LinDex = 0
        self.ROperm = np.random.permutation(self.SetSz)
        self.SOperm = np.arange(self.SetSz)

    '''
    Dataset.Batch() : Gets the Next Batch
    Parameters:
        None
    Returns:
        BatchSet Record of Design, Labels, Cardnality, and Sequence
    '''
    def Batch(self):
        if self.Random:
            summon = self.ROperm[self.LinDex:self.LinDex+self.BatchSz]
        else:
            summon = self.SOperm[self.LinDex:self.LinDex+self.BatchSz]
        self.LinDex += self.BatchSz
        batchNo = self.BatchCount
        self.BatchCount += 1
        if IsNone(self.Labels):
            return MkBatchSet(self.Design.take(summon, axis=0), None, len(summon), batchNo)
        else:
            return MkBatchSet(self.Design.take(summon, axis=0), self.Labels.take(summon, axis=0), len(summon), batchNo)

    '''
    Dataset.Valid() : If the Dataset has a Next Batch
    Parameters:
        None
    Returns:
        If the Current Dataset Has Exhausted its Batches
    '''
    def Valid(self):
        return self.LinDex < self.SetSz

    '''
    Dataset.Reinitialize() : Restart the Batching Process
    Parameters:
        ReRandom  : If the Random Order Should be Rerandomized [Default: Dataset.Random OR Randomize]
        Randomize : If Batches Should be Drawn in Random Order [Default: Dataset.Random]
        Hard      : If the Set Size Should be Updated [Default: False]
    Returns:
        None
    '''
    def Reinitialize(self, ReRandom=None, Randomize=None, Hard=False):
        if Randomize == None:
            Randomize = self.Random
        if ReRandom == None:
            ReRandom = self.Random
        if Hard:
            self.SetSz = self.Design.shape[0]
            self.SOperm = np.arange(self.SetSz)
        self.LinDex = 0
        self.BatchCount = 0
        self.Random = Randomize
        if ReRandom or Hard:
            self.ROperm = np.random.permutation(self.SetSz)

    '''
    Dataset.BroadcastLabel() : Set Specified Label as Label for All Values
    Parameters:
        NewLabel : New Label to Set
    Returns:
        None
    '''
    def BroadcastLabel(self, Label):
        labelEx = [self.SetSz]
        labelEx.extend([1]*len(Label.shape))
        self.Labels = np.tile(Label, labelEx)

    '''
    Dataset.Skim() : Splits the Dataset By Removing Balanced Subset of Members
    Parameters:
        Quantity : Number Per Class to Skim
        IsHot    : If the Dataset Labels are Onehot Vectors
        MaxK     : Maximum Class Value to Scan For
    Returns:
        New Dataset with Removed Members
    '''
    def Skim(self, Quantity, IsHot, MaxK):
        if IsHot:
            cooled = []
            cooled = np.nonzero(self.Labels)[1]
        else:
            cooled = self.Labels
        kPos = []
        for k in range(0, MaxK+1):
            kWhere = np.where(cooled == k)[0]
            if self.Random:
                np.random.shuffle(kWhere)
            kPos.append(kWhere[0:Quantity])
        kArr = np.asarray(kPos).flatten()
        newDesign = self.Design.take(kArr, axis=0)
        newLabels = self.Labels.take(kArr, axis=0)
        self.Design = np.delete(self.Design, kArr, axis=0)
        self.Labels = np.delete(self.Labels, kArr, axis=0)
        self.Reinitialize(Hard=True)
        return Dataset(newDesign, newLabels, False, self.Random, self.BatchSz)

    '''
    Dataset.ReBatch() : Changes the Size of the Batch and Reinitializes
    Parameters:
        BatchSz : New Batch Size
    Returns:
        None
    '''
    def ReBatch(self, BatchSz):
        self.BatchSz = BatchSz
        self.Reinitialize()

'''
BuildUniformNoiseDataset : Builds a Dataset Object Based on Uniform Random Noise
Parameters:
    Count    : Size of the Dataset to Generate
    Shape    : Shape of Member Design
    Random   : If Selection of Samples Should be Randomly Selected
    Label    : The Label Vector for Each Noise Sample
    BatchSz  : Size of the Batch
    NoiseMin : Minimum Noise Value
    NoiseMax : Maximum Noise Value
Returns:
    Dataset Object Filled with Uniform Random Noise
'''
def BuildUniformNoiseDataset(Count, Shape, Random, Label, BatchSz, NoiseMin, NoiseMax):
    shape = [Count]
    shape.extend(Shape)
    design = np.random.uniform(NoiseMin, NoiseMax, shape)
    return Dataset(design, Label, True, Random, BatchSz)

'''
BuildGaussianNoiseDataset : Builds a Dataset Object Based on Gaussian Random Noise
Parameters:
    Count   : Size of the Dataset to Generate
    Shape   : Shape of Member Design
    Random  : If Selection of Samples Should be Randomly Selected
    Label   : The Label Vector for Each Noise Sample
    BatchSz : Size of the Batch
    Mu      : Mean of Desired Gaussian Distribution
    Sigma   : Standard Deviation of Desired Gaussian Distribution
Returns:
    Dataset Object Filled with Gaussian RandomNoise
'''
def BuildGaussianNoiseDataset(Count, Shape, Random, Label, BatchSz, Mu, Sigma):
    shape = [Count]
    shape.extend(list(Shape))
    design = np.random.normal(Mu, Sigma, shape)
    return Dataset(design, Label, True, Random, BatchSz)
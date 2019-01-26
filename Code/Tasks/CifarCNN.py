import numpy as np
import tensorflow as tf
from LearningKit.TensorFlow.Networks.ConvNets import ClassicCNN
from LearningKit.TensorFlow.Models.StandardClassifier import StandardClassifier
from LearningKit.TensorFlow.Executors.SimpleClassification import SimpleClassification
from LearningKit.Utilities.DataReaders import Cifar10
from LearningKit.Utilities.Dataset import Dataset

def CifarCNN(Locale, Epochs, BatchSz, LogName, EvPath, AdamAlpha, ScopeFix):
    if Locale == 'Local':
        TrainingPaths = ['C:/Users/Marissa/source/repos/CVIP/Data/data_batch_1', 'C:/Users/Marissa/source/repos/CVIP/Data/data_batch_2', 'C:/Users/Marissa/source/repos/CVIP/Data/data_batch_3', 'C:/Users/Marissa/source/repos/CVIP/Data/data_batch_4', 'C:/Users/Marissa/source/repos/CVIP/Data/data_batch_5']
        TestingPath = 'C:/Users/Marissa/source/repos/CVIP/Data/test_batch'
        LogOut = LogName
        EvOut = 'C:/Users/Marissa/source/repos/CVIP/Saved/' + EvPath
        #EvOut = 'C:/Users/Marissa/source/repos/CVIP/Saved/' + EvPath + '.chkp'
    if Locale == 'Floyd':
        TrainingPaths = ['/Data/data_batch_1', '/Data/data_batch_2', '/Data/data_batch_3', '/Data/data_batch_4', '/Data/data_batch_5']
        TestingPath = '/Data/test_batch'
        LogOut = '/output/' + LogName
        EvOut = '/output/' + EvPath + '.chkp'

    data = Cifar10(TrainingPaths, TestingPath, np.float32, np.float32, True, 0, 1, True, 10)
    trainSet = Dataset(data.Train.Design, data.Train.Labels, False, True, BatchSz)
    testSet = Dataset(data.Test.Design, data.Test.Labels, False, False, 0)
    '''
    cnnParams = {
            'ConceptDim': 10,
            'Depths': [16, 32, 64, 128, 256, 512],
            'Kernels': [[4,4], [4,4], [4,4], [4,4], [4,4], [4,4]],
            'Strides': [1, 1, 1, 1, 1, [2,2]],
            'Pools': [None, ([2,2],[2,2]), ([2,2],[2,2]), None, ([2,2],[2,2]), ([2,2],[2,2])],
            'Dense': 1024,
            'Scope': 'CNN' + ScopeFix
        }
    '''
    cnnParams = {
            'ConceptDim': 10,
            'Depths': [32, 32, 64],
            'Kernels': [[5,5], [5,5], [5,5]],
            'Strides': [1, 1, 1],
            'Pools': [([2,2],[2,2]), ([2,2],[2,2]), None],
            'Dense': 64,
            'Scope': 'CNN' + ScopeFix
        }
    adamParams = {
            'learning_rate': AdamAlpha
        }
    model = StandardClassifier(ClassicCNN, cnnParams, tf.train.AdamOptimizer, adamParams, [None, 32, 32, 3], [None, 10])
    LogBase = {
        'Model': 'CNN',
        'DatasetInfo':
        {
            'Name': 'CIFAR-10',
            'TrainSize': 50000,
            'TestSize': 10000,
            'Classes': 10,
            'ElementSize': '32x32x3'
        },
        'TrainingInfo':
        {
            'Epochs': Epochs,
            'BatchSz': BatchSz
        },
        'OptimizerInfo':
        {
            'Optimizer': 'Adam',
            'Alpha'    : AdamAlpha
        }
    }
    classifier = SimpleClassification(model, trainSet, testSet)
    #classifier.Benchmark(Epochs, LogOut, LogBase, EvOut, 'Acc', True, 1)
    classifier.Evaluate(EvOut, LogOut, EvalSet=testSet)
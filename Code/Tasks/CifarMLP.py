import numpy as np
import tensorflow as tf
from LearningKit.TensorFlow.Networks.Perceptrons import MLP
from LearningKit.TensorFlow.Models.StandardClassifier import StandardClassifier
from LearningKit.TensorFlow.Executors.SimpleClassification import SimpleClassification
from LearningKit.Utilities.DataReaders import Cifar10
from LearningKit.Utilities.Dataset import Dataset

def CifarMLP(Locale, Epochs, BatchSz, LogName, EvPath, AdamAlpha, ScopeFix):
    if Locale == 'Local':
        TrainingPaths = ['C:/Users/Marissa/source/repos/CVIP/Data/data_batch_1', 'C:/Users/Marissa/source/repos/CVIP/Data/data_batch_2', 'C:/Users/Marissa/source/repos/CVIP/Data/data_batch_3', 'C:/Users/Marissa/source/repos/CVIP/Data/data_batch_4', 'C:/Users/Marissa/source/repos/CVIP/Data/data_batch_5']
        TestingPath = 'C:/Users/Marissa/source/repos/CVIP/Data/test_batch'
        LogOut = LogName
        EvOut = 'C:/Users/Marissa/source/repos/CVIP/Saved/' + EvPath + '.chkp'
    if Locale == 'Floyd':
        TrainingPaths = ['/Data/data_batch_1', '/Data/data_batch_2', '/Data/data_batch_3', '/Data/data_batch_4', '/Data/data_batch_5']
        TestingPath = '/Data/test_batch'
        LogOut = '/output/' + LogName
        EvOut = '/output/' + EvPath + '.chkp'

    data = Cifar10(TrainingPaths, TestingPath, np.float32, np.float32, True, 0, 1, False, 10, 'HumanGrey')
    trainSet = Dataset(data.Train.Design, data.Train.Labels, False, True, BatchSz)
    testSet = Dataset(data.Test.Design, data.Test.Labels, False, False, 0)
    mlpParams = {
            'HiddenSizes': [512, 256, 128],
            'OutputSize': 10,
            'Activf': tf.nn.relu,
            'Scope': 'MLP' + ScopeFix
        }
    adamParams = {
            'learning_rate': AdamAlpha
        }
    model = StandardClassifier(MLP, mlpParams, tf.train.AdamOptimizer, adamParams, [None, 32*32], [None, 10])
    LogBase = {
        'Model': 'MLP',
        'DatasetInfo':
        {
            'Name': 'Greyscale CIFAR-10',
            'TrainSize': 50000,
            'TestSize': 10000,
            'Classes': 10,
            'ElementSize': '1024x1'
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
    classifier.Evaluate('RawMLP/MLPStateRaw.chkp', LogOut, EvalSet=testSet)
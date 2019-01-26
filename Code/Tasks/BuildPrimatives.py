import cv2
import csv
import numpy as np
from sklearn.cluster import KMeans
from LearningKit.Utilities.DataReaders import Cifar10
from LearningKit.Utilities.DataTools import CatOrAssign

DefaultSiftArgs = {'nfeatures': 50, 'nOctaveLayers': 10, 'contrastThreshold': 0.01, 'edgeThreshold': 20, 'sigma': 0.8}
def BuildPrimatives(Locale, SaveFile, WordCount, SiftArgs=DefaultSiftArgs):
    if Locale == 'Local':
        TrainingPaths = ['C:/Users/Marissa/source/repos/CVIP/Data/data_batch_1', 'C:/Users/Marissa/source/repos/CVIP/Data/data_batch_2', 'C:/Users/Marissa/source/repos/CVIP/Data/data_batch_3', 'C:/Users/Marissa/source/repos/CVIP/Data/data_batch_4', 'C:/Users/Marissa/source/repos/CVIP/Data/data_batch_5']
        TestingPath = 'C:/Users/Marissa/source/repos/CVIP/Data/test_batch'
        SaveTo = SaveFile
    if Locale == 'Floyd':
        TrainingPaths = ['/Data/data_batch_1', '/Data/data_batch_2', '/Data/data_batch_3', '/Data/data_batch_4', '/Data/data_batch_5']
        TestingPath = '/Data/test_batch'
        SaveTo = '/output/' + SaveFile

    # Read Dataset and Make SIFT Object
    print('Reading Dataset...')
    data = Cifar10(TrainingPaths, TestingPath, np.float32, np.float32, False, 0, 1, True, 1, 'HumanGrey')
    sift = cv2.xfeatures2d.SIFT_create(**SiftArgs)

    # Build SIFT Descriptors
    print('Building Training Descriptors...')
    trainDesc = []
    for img in data.Train.Design:
        _, desc = sift.detectAndCompute(img.astype(np.uint8), None)
        trainDesc.append(desc)
    trainDesc = np.concatenate(trainDesc, axis=0)

    # Develop Visual Words
    print('Building Visual Words Dictionary...')
    clusters = KMeans(n_clusters=WordCount).fit(trainDesc)

    # Build Histograms on Training Set
    print('Calculating Training Words...')
    trainDesign = []
    for img in data.Train.Design:
        _, desc = sift.detectAndCompute(img.astype(np.uint8), None)
        words = clusters.predict(desc)
        unique, counts = np.unique(words, return_counts=True)
        fvect = np.zeros([WordCount])
        fvect[unique] = counts
        trainDesign.append(fvect.reshape([1, fvect.shape[0]]))
    trainDesign = np.concatenate(trainDesign, axis=0)

    # Build Histograms on Testing Set
    print('Calculating Testing Words...')
    testDesign = []
    for img in data.Test.Design:
        _, desc = sift.detectAndCompute(img.astype(np.uint8), None)
        words = clusters.predict(desc)
        unique, counts = np.unique(words, return_counts=True)
        fvect = np.zeros([WordCount])
        fvect[unique] = counts
        testDesign.append(fvect.reshape([1, fvect.shape[0]]))
    testDesign = np.concatenate(testDesign, axis=0)

    # Append Labels to Designs
    print('Building NumPy Arrays...')
    trainSet = np.concatenate([data.Train.Labels.reshape([data.Train.Labels.shape[0], 1]), trainDesign], axis=-1)
    testSet = np.concatenate([data.Test.Labels.reshape([data.Test.Labels.shape[0], 1]), testDesign], axis=-1)

    # Save to CSV
    print('Writing Data...')
    np.savetxt(SaveTo + '.Train.csv', trainSet, delimiter=',')
    np.savetxt(SaveTo + '.Test.csv', testSet, delimiter=',')

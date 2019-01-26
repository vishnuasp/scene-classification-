import numpy as np
import tensorflow as tf
import LearningKit.TensorFlow.Records as Records


class StandardClassifier():
    '''
    StandardClassifier Constructor
    Parameters:
        Network       : Class of the Network
        NetworkArgs   : Arguments to Construct the Network
        Optimizer     : Class of the TensorFlow Optimizer
        OptimizerArgs : Arguments to Construct the TensorFlow Optimizer
        FeatureShape  : Shape of the Feature Tensor
        LabelShape    : Shape of the Label Tensor (Must be Onehot)
        FeatureType   : Datatype of the Feature Tensor [Default: tf.float32]
        LabelType     : Datatype of the Label Tensor [Default: tf.float32]
    '''
    def __init__(self, Network, NetworkArgs, Optimizer, OptimizerArgs, FeatureShape, LabelShape, FeatureType=tf.float32, LabelType=tf.float32):
        # Build Network
        self.Network = Network(**NetworkArgs)
        # Build Optimizer
        self.Optimizer = Optimizer(**OptimizerArgs)
        # Set TensorFlow Placeholders
        self.Tf = Records.GetRecord('StandardPlaceholders')
        self.Tf.X = tf.placeholder(FeatureType, FeatureShape, name='Tf.X')
        self.Tf.Y = tf.placeholder(LabelType, LabelShape, name='Tf.Y')
        self.Tf.Mode = tf.placeholder(tf.bool, name='Tf.Mode')
        # Set TensorFlow Operations
        self.TfOps = Records.GetRecord('StandardOps')
        self.TfOps.Feed = self.Network.Feedforward(self.Tf.X, self.Tf.Mode)
        self.TfOps.Loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Tf.Y, logits=self.TfOps.Feed))
        self.TfOps.Acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.TfOps.Feed, axis=1), tf.argmax(self.Tf.Y, axis=1)), tf.float32))
        self.TfOps.Train = self.Optimizer.minimize(self.TfOps.Loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Network.Scope))
        self.TfOps.Predict = tf.argmax(self.TfOps.Feed, axis=1)

    '''
    StandardClassifier.TrainBatch() : Returns Argument Dictionary to Train on a Batch with Session
    Parameters:
        Design : NumPy Design Array (Features)
        Labels : NumPy Labels Array (Labels)
    Returns:
        Arguments Session Record for tf.Session.run()
    '''
    def TrainBatch(self, Design, Labels):
        return Records.MkSession(['Acc', 'Loss', None], [self.TfOps.Acc, self.TfOps.Loss, self.TfOps.Train], {self.Tf.X: Design, self.Tf.Y: Labels, self.Tf.Mode: True})

    '''
    StandardClassifier.TestBatch() : Returns Argument Dictionary to Test Performance on a Batch with Session
    Parameters:
        Design : NumPy Design Array (Features)
        Labels : NumPy Labels Array (Labels)
    Returns:
        Arguments Session Record for tf.Session.run()
    '''
    def TestBatch(self, Design, Labels):
        return Records.MkSession(['Acc', 'Loss'], [self.TfOps.Acc, self.TfOps.Loss], {self.Tf.X: Design, self.Tf.Y: Labels, self.Tf.Mode: False})

    '''
    StandardClassifier.Confusion() : Returns Argument Dictionary to Get List of Predictions on a Batch with Session
    Parameters:
        Design : NumpyDesign Array (Features)
    Returns:
        Arguments Session Record for tf.Session.run()
    '''
    def Confusion(self, Design):
        return Records.MkSession(['Predicted'], [self.TfOps.Predict], {self.Tf.X: Design, self.Tf.Mode: False})
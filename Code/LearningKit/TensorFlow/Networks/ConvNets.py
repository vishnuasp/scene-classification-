# Outside Imports
import tensorflow as tf
# LearningKit Imports
from LearningKit.Utilities.Tests import IsNotNone
'''
ClassicCNN Provides an LK.TF Network for Classification
Follows Relatively Standard CNN Architecture
(Conv -> ReLU -> MaxPool)^n -> Dense
'''
class ClassicCNN():
    '''
    ClassicCNN Constructor
    Parameters:
        ConceptDim : Dimensionality of the Concept Space
        Depths     : List of Depths
        Kernels    : List of Kernels
        Strides    : List of Strides
        Pools      : List of Tuples of (PoolSize, PoolStride)
        Dense      : Dense Layer Output Size (Or None)
        Scope      : Variable Name to Associate
    '''
    def __init__(self, ConceptDim, Depths, Kernels, Strides, Pools, Dense, Scope):
        # Dataset Properties
        self.ConceptDim = ConceptDim
        # CNN Layer Properties
        self.Depths = Depths
        self.Kernels = Kernels
        self.Strides = Strides
        self.Pools = Pools
        # CNN Dense Layer
        self.Dense = Dense
        # TensorFlow Reference Information
        self.Scope = Scope
        # Internal State Information
        self.Initialized = False

    '''
    ClassicCNN.Feedforward : Performs Feedforward Operation on CNN
    Parameters:
        Input     : The Input to the Network (Image)
        TrainMode : If the Network is Currently Being Trained
    '''
    def Feedforward(self, Input, TrainMode):
        with tf.variable_scope(self.Scope, reuse=self.Initialized):
            # Input Layer
            logits = tf.convert_to_tensor(Input)
            # Hidden Layers
            for D, K, S, P in zip(self.Depths, self.Kernels, self.Strides, self.Pools):
                logits = tf.layers.conv2d(logits, D, K, S, 'same')
                logits = tf.nn.relu(logits)
                if IsNotNone(P):
                    logits = tf.layers.max_pooling2d(logits, P[0], P[1])
            # Output Layer
            if IsNotNone(self.Dense):
                logits = tf.layers.dense(logits, self.Dense)
                logits = tf.nn.relu(logits)
            logits = tf.layers.dense(logits, self.ConceptDim)
            # Global Averaging
            logits = tf.reduce_mean(tf.reduce_mean(logits, axis=1), axis=1)
            #logits = tf.squeeze(logits, [1,2])
        self.Initialized = True
        return logits
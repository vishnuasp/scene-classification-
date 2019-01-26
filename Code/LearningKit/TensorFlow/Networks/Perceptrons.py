import tensorflow as tf

'''
MLP Provides an LK.TF Network for Classification
Can Create an MLP Using Specified Activation
'''
class MLP():
    '''
    MLP Constructor
    Parameters:
        HiddenSizes : List of Layer Sizes for Hidden Layers
        OutputSize  : Number of Output Neurons
        Activf      : Activation Function
        Scope       : Variable Name to Associate
    '''
    def __init__(self, HiddenSizes, OutputSize, Activf, Scope):
        # Layer Sizes
        self.HiddenSizes = HiddenSizes
        self.OutputSize = OutputSize
        # Activation Function
        self.Activf = Activf
        # TensorFlow Reference Information
        self.Scope = Scope
        # Internal State Information
        self.Initialized = False

    '''
    MLP.Feedforward : Performs Feedforward Operation on MLP
    Parameters:
        Input     : The Input to the Network
        TrainMode : [Unused]
    '''
    def Feedforward(self, Input, TrainMode):
        with tf.variable_scope(self.Scope, reuse=self.Initialized):
            # Input Layer
            logits = tf.convert_to_tensor(Input)
            # Hidden Layers
            for H in self.HiddenSizes:
                logits = tf.layers.dense(logits, H, activation=self.Activf)
            # Output Layer
            logits = tf.layers.dense(logits, self.OutputSize)
        self.Initialized = True
        return logits
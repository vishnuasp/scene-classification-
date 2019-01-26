'''
LearningKit.TensorFlow.Records

Collection of Methods for Creating Standard LK.TF Records

Author:  Marissa Dominijanni
Version: v0.1 (2017.12.13)
'''
# Python Imports
from collections import namedtuple
# LearningKit Imports
from LearningKit.Utilities.Tests import IsSequence

'''
Specifications : Dictionary of Empty Standard-Formatted RecordClass Objects
Items:
    StandardPlaceholders : TfPlaceholders : Standard Set of TensorFlow Placeholders
    StandardOps          : TfOps          : Standard Set of TensorFlow Operations for Model
    BasicMetrics         : Metrics        : Standard Set of Training Metrics to Return
    Session              : Session        : Set for session.run() in TensorFlow
'''
Specifications = {
    # Standard Set (Model Using TF Layers API with Single Network)
    'StandardPlaceholders' : {'Name': 'TfPlaceholders', 'Fields': ['X', 'Y', 'Mode']},
    'StandardOps'          : {'Name': 'TfOps', 'Fields': ['Feed', 'Loss', 'Acc', 'Train', 'Predict']},
    # Metrics
    'BasicMetrics'         : {'Name': 'Metrics', 'Fields': ['Acc', 'Loss']},
    # Session Run Record
    'Session'              : {'Name': 'Session', 'Fields': ['Fields', 'Fetches', 'FeedDict']}
}

'''
GetRecord : Returns a Copy of a Desired Record
Parameters:
    Spec : Specification of Record
Returns:
    New Object of Desired Record
'''
def GetRecord(Spec):
    return namedtuple(Specifications[Spec]['Name'], Specifications[Spec]['Fields'])

'''
MkStandardPlaceholders : Constructs a StandardPlaceholders Record
Parameters:
    X    : Feature Tensor
    Y    : Label Tensor
    Mode : Training Mode
Returns:
    Loaded StandardPlaceholders Record
'''
def MkStandardPlaceholders(X, Y, Mode):
    rc = GetRecord('StandardPlaceholders')
    rc.X = X
    rc.Y = Y
    rc.Mode = Mode
    return rc

'''
MkStandardOps : Constructs a StandardOps Record
Parameters:
    Feed    : Operation to Perform Feedforward
    Loss    : Operation to Determine Classifier Loss
    Acc     : Operation to Determine Classifier Accuracy
    Train   : Operation to Train Classifier
    Predict : Operation to Get Classifier Predictions
Returns:
    Loaded StandardOps Record
'''
def MkStandardOps(Feed, Loss, Acc, Train, Predict):
    rc = GetRecord('StandardOps')
    rc.Feed = Feed
    rc.Loss = Loss
    rc.Acc = Acc
    rc.Train = Train
    rc.Predict = Predict
    return rc

'''
MkBasicMetrics : Constructs a BasicMetrics Record
Parameters:
    Acc  : Accuracy Value
    Loss : Loss Value
Returns:
    Loaded BasicMetrics Record
'''
def MkBasicMetrics(Acc, Loss):
    rc = GetRecord('BasicMetrics')
    rc.Acc = Acc
    rc.Loss = Loss
    return rc


'''
MkSession : Constructs a Session Record
Parameters:
    Fields   : Fields to Log for Results from Session Run
    Fetches  : Fetches for Session Run
    FeedDict : FeedDict for Session Run
Returns:
    Loaded BasicMetrics Record
'''
def MkSession(Fields, Fetches, FeedDict):
    rc = GetRecord('Session')
    rc.Fields = Fields
    rc.Fetches = Fetches
    rc.FeedDict = FeedDict
    return rc
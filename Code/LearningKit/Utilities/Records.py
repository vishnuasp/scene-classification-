'''
LearningKit.Utilities.Records

Collection of Methods for Creating Standard LK Records

Author:  Marissa Dominijanni
Version: v0.2 (2017.12.14)
'''
# Python Imports
from collections import namedtuple
# LearningKit Imports
from LearningKit.Utilities.Tests import IsSequence

'''
Specifications : Dictionary of Empty Standard-Formatted RecordClass Objects
Items:
    BatchSet : Design Tensor, Labels Tensor, Number of Members, and Sequential Call Number of a Batch
    DataPair : Design Tensor and Labels Tensorsfor a Dataset
    DataQuad : DataPairs of Training Set and Testing Set
    DataHex  : DataPairs of Training Set, Testing Set, and Validation Set
    Tracker  : Little Tracker Return
'''
Specifications = {
    # Dataset Records
    'BatchSet' : {'Name': 'BatchPair', 'Fields': ['Design', 'Labels', 'Sz', 'No']},
    'DataPair' : {'Name': 'DataPair', 'Fields': ['Design', 'Labels']},
    'DataQuad' : {'Name': 'DataQuad', 'Fields': ['Train', 'Test']},
    'DataHex'  : {'Name': 'DataHex', 'Fields': ['Train', 'Test', 'Valid']},
    'Tracker'  : {'Name': 'Tracker', 'Fields': ['Update', 'Value']}
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
MkBatchSet : Constructs a BatchSet Record
Parameters:
    Design : Design Tensor
    Labels : Labels Tensor
    Sz     : Number of Members in Batch
    No     : Sequential Batch Number
Returns:
    Loaded BatchSet Record
'''
def MkBatchSet(Design, Labels, Sz, No):
    rc = GetRecord('BatchSet')
    rc.Design = Design
    rc.Labels = Labels
    rc.Sz = Sz
    rc.No = No
    return rc

'''
MkDataPair : Constructs a DataPair Record
Parameters:
    Design : Design Tensor
    Labels : Labels Tensor
Returns:
    Loaded DataPair Record
'''
def MkDataPair(Design, Labels):
    rc = GetRecord('DataPair')
    rc.Design = Design
    rc.Labels = Labels
    return rc

'''
MkDataQuad : Constructs a DataQuad Record
Parameters:
    Train : Either DataPair Record or Sequence (Design, Labels) of Training Set
    Test  : Either DataPair Record or Sequence (Design, Labels) of Testing Set
Returns:
    Loaded DataQuad Record
'''
def MkDataQuad(Train, Test):
    rc = GetRecord('DataQuad')
    if IsSequence(Train):
        rc.Train = GetRecord('DataPair')
        rc.Train.Design = Train[0]
        rc.Train.Labels = Train[1]
    else:
        rc.Train = Train
    if IsSequence(Test):
        rc.Test = GetRecord('DataPair')
        rc.Test.Design = Test[0]
        rc.Test.Labels = Test[1]
    else:
        rc.Test = Test
    return rc

'''
MkDataHex : Constructs a DataHex Record
Parameters:
    Train : Either DataPair Record or Sequence (Design, Labels) of Training Set
    Test  : Either DataPair Record or Sequence (Design, Labels) of Testing Set
    Valid : Either DataPair Record or Sequence (Design, Labels) of Validation Set
Returns:
    Loaded DataHex Record
'''
def MkDataHex(Train, Test, Valid):
    rc = GetRecord('DataHex')
    if IsSequence(Train):
        rc.Train = GetRecord('DataPair')
        rc.Train.Design = Train[0]
        rc.Train.Labels = Train[1]
    else:
        rc.Train = Train
    if IsSequence(Test):
        rc.Test = GetRecord('DataPair')
        rc.Test.Design = Test[0]
        rc.Test.Labels = Test[1]
    else:
        rc.Test = Test
    if IsSequence(Valid):
        rc.Valid = GetRecord('DataPair')
        rc.Valid.Design = Test[0]
        rc.Valid.Labels = Test[1]
    else:
        rc.Valid = Valid
    return rc

'''
MkTracker : Constructs a Tracker Record
Parameters:
    Update : If Tracker Value Has Updated Since Last
    Value  : Tracker Value
Returns:
    Loaded Tracker Record
'''
def MkTracker(Update, Value):
    rc = GetRecord('Tracker')
    rc.Update = Update
    rc.Value = Value
    return rc
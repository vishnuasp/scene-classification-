'''
LearningKit.Utilities.Tests

Collection of Methods for Testing/Checking Variables

Author:  Marissa Dominijanni
Version: v1.1 (2017.12.09)
'''
# Python Imports
import collections

'''
IsNone : Simple Function to Test if Value is NoneType
Parameters:
    Value : Object to Type Check
Returns:
    If Passed Value is of Type NoneType
'''
def IsNone(Value):
    return type(Value) == type(None)

'''
IsNotNone : Simple Function to Test if Value is not of NoneType
Parameters:
    Value : Object to Type Check
Returns:
    If Passed Value is of a Type Other than NoneType
'''
def IsNotNone(Value):
    return type(Value) != type(None)

'''
ExistsEqual : Checks Value Against Another if Value is not of NoneType
Parameters:
    ValueA : Value to Test
    ValueB : Value to Test
Returns:
    True if Both Values are not None and Equal, Otherwise False
'''
def ExistsEqual(ValueA, ValueB):
    if IsNotNone(ValueA) and IsNotNone(ValueB):
        return ValueA == ValueB
    else:
        return False

'''
IsSequence : Simple Function to Test if Value is a Sequence (List, Tuple, etc)
Parameters:
    Value : Object to Check
Returns:
    If Value is a Sequence
'''
def IsSequence(Value):
    return isinstance(Value, collections.Sequence) and not isinstance(Value, str)
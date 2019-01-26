from LearningKit.Utilities.Records import MkTracker
from LearningKit.Utilities.Tests import IsNone, IsNotNone
class Tracker():
    def __init__(self, SeekMax=None, SeekMin=None):
        assert(SeekMax != SeekMin)
        assert(IsNotNone(SeekMax) or IsNotNone(SeekMin))
        self.Update = None
        if IsNotNone(SeekMax):
            if SeekMax:
                self.SeekMax = True
                self.SeekMin = False
                self.Value = -float('Inf')
            else:
                self.SeekMax = False
                self.SeekMin = True
                self.Value = float('Inf')
            self.Update = False
        if IsNotNone(SeekMin) and IsNone(self.Update):
            if SeekMin:
                self.SeekMax = False
                self.SeekMin = True
                self.Value = float('Inf')
            else:
                self.SeekMax = True
                self.SeekMin = False
                self.Value = -float('Inf')
            self.Update = False
    def Feed(self, NewValue):
        if self.SeekMax and (NewValue > self.Value):
            self.Value = NewValue
            self.Update = True
        if self.SeekMin and (NewValue < self.Value):
            self.Value = NewValue
            self.Update = True
    def Expose(self):
        if self.Update:
            self.Update = False
            return MkTracker(True, self.Value)
        return MkTracker(False, self.Value)
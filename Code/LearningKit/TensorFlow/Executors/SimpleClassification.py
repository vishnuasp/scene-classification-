import json
import numpy as np
import tensorflow as tf
from LearningKit.Utilities.Little import Tracker
from LearningKit.Utilities.Tests import IsNone, IsNotNone, ExistsEqual

class SimpleClassification():
    '''
    SimpleClassification Constructor
    Parameters:
        Model    : LK TensorFlow Model for Classifying
        TrainSet : LK Dataset Object Containing Training Data
        TestSet  : LK Dataset Object Containing Testing Data
    '''
    def __init__(self, Model, TrainSet, TestSet):
        # Load Model
        self.Model = Model
        # Load Datasets
        self.TrainSet = TrainSet
        self.TestSet = TestSet
        # Set TestSet Batch to Whole Dataset
        self.TestSet.ReBatch(self.TestSet.SetSz)
    '''
    SimpleClassification.Benchmark : Train the Model and Evaluate Performance Along the Way
        Epochs    : Number of Epochs to Train On
        LogBase   : Base of the Logfiles Names/Paths [Default: None]
        LogPass   : Header Dictionary to Write to Each Log [Default: None]
        BestPath  : If Path is Provided, Saves the Model with Best Test Field [Default: None]
        BestField : Field Name to Determine Best Model [Default: None]
        SeekMax   : If Higher is Better For BestField [Default: True]
        Reports   : Level of Status to Report (0 = None, 1 = Epoch, 2 = Batch) [Default: 0]
    '''
    def Benchmark(self, Epochs, LogBase=None, LogPass=None, BestPath=None, BestField=None, SeekMax=True, Reports=0):
        # Setup Saver
        saver = tf.train.Saver()
        # Start TensorFlow Session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # Setup Logging
            # Ensure Valid LogPass
        if IsNone(LogBase):
            LogPass = None
            # Setup Log Files
        if IsNotNone(LogBase):
            trainingLog = open(LogBase + '.Train.json', 'w')
            testingLog = open(LogBase + '.Test.json', 'w')
            # Write LogPass
        if IsNotNone(LogPass):
            trainingLog.write(json.dumps(LogPass) + '\n')
            testingLog.write(json.dumps(LogPass) + '\n')
        # Track Test Field
        trackr = Tracker(SeekMax=SeekMax)
        # Epoch Loop
        for epoch in range(Epochs):
            # Report Status
            self._Report(Reports, 'Epoch: '+str(epoch+1), 1)
            # Training Loop Initialization
                # Reinitialize Training Sets
            self.TrainSet.Reinitialize(ReRandom=True, Randomize=True)
                # Report Status
            self._Report(Reports, '       TrainingLoop', 1)
            # Training Loop
            while self.TrainSet.Valid():
                # Collect Batch
                trBatch = self.TrainSet.Batch()
                # Report Status
                self._Report(Reports, 'Batch: '+str(trBatch.No+1), 2)
                # Build Training Operations
                trOps = self.Model.TrainBatch(trBatch.Design, trBatch.Labels)
                # Train Classifier
                trVals = sess.run(trOps.Fetches, trOps.FeedDict)
                # Build Results Dictionary
                trRes = {}
                for v, f in zip(trVals, trOps.Fields):
                    if IsNotNone(f):
                        trRes[f] = v.astype(float).tolist()
                # Log Training Status
                if IsNotNone(LogBase):
                    trainlv = {'Mode': 'Training', 'Locale': {'Epoch': epoch+1, 'Batch': trBatch.No+1}, 'Results': trRes}
                    trainingLog.write(json.dumps(trainlv) + '\n')
            # Testing Loop Initialization
                # Reinitialize Testing Sets
            self.TestSet.Reinitialize()
                # Report Status
            self._Report(Reports, '       TestingLoop', 1)
            # Testing Loop
            while self.TestSet.Valid():
                # Collect Batch
                teBatch = self.TestSet.Batch()
                # Report Status
                self._Report(Reports, 'Batch: '+str(teBatch.No+1), 2)
                # Build Testing Operations
                teOps = self.Model.TestBatch(teBatch.Design, teBatch.Labels)
                # Test Classifier
                teVals = sess.run(teOps.Fetches, teOps.FeedDict)
                # Build Results Dictionary
                teRes = {}
                for v, f in zip(teVals, teOps.Fields):
                    if IsNotNone(f):
                        teRes[f] = v.astype(float).tolist()
                        if ExistsEqual(BestField, f):
                            trackr.Feed(v.astype(float).tolist())
                # Log Testing Status
                if IsNotNone(LogBase):
                    testlv = {'Mode': 'Testing', 'Locale': {'Epoch': epoch+1, 'Batch': teBatch.No+1}, 'Results': teRes}
                    testingLog.write(json.dumps(testlv) + '\n')
            # End of Epoch Logging/Saving
            if IsNotNone(LogBase):
                trainingLog.flush()
                testingLog.flush()
            if IsNotNone(BestPath):
                trackstat = trackr.Expose()
                if trackstat.Update:
                    saver.save(sess, BestPath)
        # Close Logs
        if IsNotNone(LogBase):
            trainingLog.close()
            testingLog.close()
        # End TensorFlow Session
        sess.close()

    '''
    SimpleClassification.Evaluate : Evaluates Model With Saved State on Data
    Parameters:
        StatePath : Path to Saved State
        EvalBase  : Base Path to Save Evalaution to
        EvalSet   : Dataset to Evaluate
    '''
    def Evaluate(self, StatePath, EvalBase, EvalSet):
        # Setup Saver
        saver = tf.train.Saver()
        # Reinitialize EvalSet
        EvalSet.Reinitialize()
        # Open LogFile
        evalLog = open(EvalBase + '.Eval.json', 'w')
        # Start TensorFlow Session
        sess = tf.Session()
        # Restore State
        saver.restore(sess, StatePath)
        # Evaluate
        while EvalSet.Valid():
            evBatch = EvalSet.Batch()
            evOps = self.Model.Confusion(evBatch.Design)
            evVals = sess.run(evOps.Fetches, evOps.FeedDict)
            evRes = {}
            for v, f in zip(evVals, evOps.Fields):
                if IsNotNone(f):
                    evRes[f] = v.astype(float).tolist()
            evallv = {'Mode': 'Evaluation', 'BatchNo': evBatch.No+1, 'Labels': evBatch.Labels.astype(float).tolist(), 'Results': evRes}
            evalLog.write(json.dumps(evallv) + '\n')
        evalLog.close()
        sess.close()

    '''
    SimpleClassification._Report : Helper Function for Printing Reports
    Parameters:
        ReportLevel   : Maximum Level to Report
        ReportMessage : Message to Report
        MessageLevel  : Level of Message
    '''
    def _Report(self, ReportLevel, ReportMessage, MessageLevel):
        if MessageLevel <= ReportLevel:
            print(ReportMessage)
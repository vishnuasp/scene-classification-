from Tasks.CifarCNN import CifarCNN
from Tasks.CifarMLP import CifarMLP

from Tasks.BuildPrimatives import BuildPrimatives

CifarCNN('Local', 100, 200, 'CNN.Alpha', 'CNNState', 0.002, '1')
#CifarMLP('Floyd', 100, 200, 'MLP.Alpha', 'MLPStateRaw', 0.002, '1')
#CifarMLP('Local', 100, 200, 'MLP.Alpha', 'MLPState', 0.002, '1')
#BuildPrimatives('Floyd', 'SIFT.50-50', 50)
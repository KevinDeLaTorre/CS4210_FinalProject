from random import randint
from numpy.lib.function_base import average
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Data filename
readFile = "data.csv"

# Random forest variables
testSplit = .2 # Percentage of dataset that will become test set
testCases = 1000 # Number of full random forest test cases to run
testNum = 20 # Number of sub test cases to run for estimators/splits
testEstimatorIncremental = 20 # What to increase the estimators by in the estimator test
maxFeatures = .5 # Default max number of splits at every node
showClassificationReport = False # Will either show a classification report when doing the tests or not

# Output files
writeFile = "output{}.csv".format( testCases )
logFile = "log{}.txt".format( testCases )
file = open( logFile, 'w' )

# Read in data file
train_data = pd.read_csv( readFile )

# Get data frame of dataset without target feature
x = train_data.drop( columns='DEATH_EVENT' )

# Set target feature
y = train_data[ 'DEATH_EVENT' ]

# Get testCases amount of random integers to set as seeds
seeds = []
for i in range( 0, testCases ):
  seeds.append( randint(0, 10000000) )

globalAccuracies = {}
countSeeds = 0
# Do full test on random seeds
for seed in seeds:
  file.write( "\nSeed {}: {}".format( countSeeds, seed ) )
  print( "Seed {}: {}".format( countSeeds, seed ) )
  # Split the data into test/train datasets
  x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=testSplit, random_state=seed )
  accuracies = []
  f1Scores = []
  count = 0

  file.write( "\tNUMBER OF TREES/ESTIMATORS TEST\n")
  print( "\tNUMBER OF TREES/ESTIMATORS TEST")
  # Testing whether number of trees increases prediction
  for i in range( testEstimatorIncremental, (testEstimatorIncremental*testNum ), testEstimatorIncremental ):
    # Apply data to model
    rnd_clf = RandomForestClassifier( n_estimators=i, max_features=maxFeatures, random_state=seed, n_jobs=-1 )
    rnd_clf.fit( x_train, y_train )

    prediction = rnd_clf.predict( x_test )
    accuracies.append( accuracy_score( y_test, prediction) )
    f1Scores.append( f1_score( y_test, prediction ) )

    if ( showClassificationReport == True ):
      print( classification_report( y_test, prediction ) )
    file.write( "\tTest {} | Number of estimators: {} | Accuracy: {:.3f} | F1-Score: {:.3f}\n".format( count, i, accuracies[ count ], f1Scores[ count ] ) )
    print( "\tTest {} | Number of estimators: {} | Accuracy: {:.3f} | F1-Score: {:.3f}".format( count, i, accuracies[ count ], f1Scores[ count ] ) )
    count += 1

  file.write( "\n\tAverage accuracy: {:.3f}\n".format( average( accuracies ) ) )
  print( "\n\tAverage accuracy: {:.3f}".format( average( accuracies ) ) )
  file.write( "\tAverage F1-Scores: {:.3f}\n".format( average( f1Scores ) ) )
  print( "\tAverage F1-Scores: {:.3f}".format( average( f1Scores ) ) )
  lowestIndexEst = f1Scores.index( max( f1Scores ) )
  estimators = ( lowestIndexEst + 1 )*testEstimatorIncremental # Set estimators to lowest-best num of estimators for performance/results
  file.write( "\tBest estimators: {} with {:.3f} F1-Score\n".format( estimators, f1Scores[ lowestIndexEst ] ) )
  print( "\tBest estimators: {} with {:.3f} F1-Score\n".format( estimators, f1Scores[ lowestIndexEst ] ) )

  file.write( "\n\tMAX FEATURES TEST\n" )
  print( "\tMAX FEATURES TEST" )
  splitAccuracies = []
  splitF1 = []
  # Testing if max number of splits at each node improves prediction
  for i in range( 1, testNum, 1 ):
    # Apply data to model
    tmpSplit = i/testNum
    rnd_clf = RandomForestClassifier( n_estimators=estimators, max_features=tmpSplit, random_state=seed, n_jobs=-1 )
    rnd_clf.fit( x_train, y_train )

    prediction = rnd_clf.predict( x_test )
    splitAccuracies.append( accuracy_score( y_test, prediction) )
    splitF1.append( f1_score( y_test, prediction ) )

    if ( showClassificationReport == True ):
      file.write( classification_report( y_test, prediction ) )
      print( classification_report( y_test, prediction ) )
    file.write( "\tTest {} | Max Feature split: {:.2f} | Accuracy: {:.3f} | F1-Score: {:.3f}\n".format( i-1, tmpSplit, splitAccuracies[ i-1 ], splitF1[ i-1 ] ) )
    print( "\tTest {} | Max Feature splits: {:.2f} | Accuracy: {:.3f} | F1-Score: {:.3f}".format( i-1, tmpSplit, splitAccuracies[ i-1 ], splitF1[ i-1 ] ) )

  file.write( "\n\tAverage accuracy: {:.3f}\n".format( average( splitAccuracies ) ) )
  print( "\n\tAverage accuracy: {:.3f}".format( average( splitAccuracies ) ) )
  file.write( "\tAverage F1-Scores: {:.3f}\n".format( average( f1Scores ) ) )
  print( "\tAverage F1-Scores: {:.3f}".format( average( f1Scores ) ) )
  lowestIndexSplit = splitF1.index( max( splitF1 ) )
  split = ( lowestIndexSplit + 1 )*(1/testNum) # Set splits to lowest-best num of splits for performance/results
  file.write( "\tBest split: {:.2f} with {:.3f} F1-Score\n".format( split, splitF1[ lowestIndexSplit ] ) )
  print( "\tBest split: {:.2f} with {:.3f} F1-Score".format( split, splitF1[ lowestIndexSplit ] ) )

  # Print best classifier found
  file.write( "\tTherefore best classifier found is: {} estimators | {:.2f} split\n".format( estimators, split ) )
  print( "\tTherefore best classifier found is: {} estimators | {:.2f} split".format( estimators, split ) )
  rnd_clf = RandomForestClassifier( n_estimators=estimators, max_features=split, random_state=seed, n_jobs=-1)
  rnd_clf.fit( x_train, y_train )

  prediction = rnd_clf.predict( x_test )
  accuracy = accuracy_score( y_test, prediction )
  f1Score = f1_score( y_test, prediction )
  file.write( classification_report( y_test, prediction ) )
  print( classification_report( y_test, prediction ) )
  file.write( "\tBest F1-Score: {:.3f}\n".format( f1Score ) )
  print( "\tBest F1-Score: {:.3f}\n".format( f1Score ) )
  globalAccuracies[ seed ] = [ f1Score, accuracy, estimators, split ]
  countSeeds += 1

bestSeed = seeds[ 0 ]
for seed in globalAccuracies:
  if ( globalAccuracies[ seed ][ 0 ] > globalAccuracies[ bestSeed ][ 0 ] ):
    bestSeed = seed

file.write( "\n\nBest overall random forest was with Random Seed [ {} ] with {:.2f} F1-Score and {:.2f} accuracy using {} estimators and a {:.2f} Max Feature split\n"
  .format( bestSeed, globalAccuracies[ bestSeed ][ 0 ], globalAccuracies[ bestSeed ][ 1 ], globalAccuracies[ bestSeed ][ 2 ], globalAccuracies[ bestSeed ][ 3 ] ) )
print( "\n\nBest overall random forest was with Random Seed [ {} ] with {:.2f} F1-Score and {:.2f} accuracy using {} estimators and a {:.2f} Max Feature split\n"
  .format( bestSeed, globalAccuracies[ bestSeed ][ 0 ], globalAccuracies[ bestSeed ][ 1 ], globalAccuracies[ bestSeed ][ 2 ], globalAccuracies[ bestSeed ][ 3 ] ) )

# Write results to file
writeDF = pd.DataFrame.from_dict( globalAccuracies, orient='columns').transpose()
writeDF.columns = [ "F1-Score", "Accuracy", "Estimators", "Split" ]
print( writeDF.to_string() )
file.write( writeDF.to_string() )
writeDF.to_csv( writeFile )

# Close log file
file.close()
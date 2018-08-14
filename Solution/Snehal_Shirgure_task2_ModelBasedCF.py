
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import sys
import numpy
import time
import csv
import collections

inputfile = sys.argv[1]
testfile = sys.argv[2]

t= time.time()

sc = SparkContext('local','task2')
# Load and parse the data
data = sc.textFile(inputfile)
test = sc.textFile(testfile)

testdata = test.map(lambda l: l.split(','))\
        .filter(lambda x: 'userId' not in x )\
        .map(lambda l: ( ( int(l[0]), int(l[1]) ) , 1 ) )

ratings = data.map(lambda l: l.split(','))\
    .filter(lambda x: 'userId' not in x )\
    .map(lambda l:  ( ( int(l[0]), int(l[1]) ), float(l[2]) ) )

training = ratings.subtractByKey(testdata).map(lambda x: Rating(x[0][0],x[0][1],x[1]))

test = testdata.collect()

pratings = ratings.filter(lambda x: (x[0],1) in test).map(lambda x: Rating(x[0][0],x[0][1],x[1]))

rank = 10
numIterations = 7
model = ALS.train(training, rank, numIterations,0.1)

testing = testdata.map(lambda p: (p[0][0], p[0][1]))
predictions = model.predictAll(testing).map(lambda r: ((r[0], r[1]), r[2]))

ratesAndPreds = pratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)

mse = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()

predictions2 = predictions.collect()


predictions2 = dict(predictions2)
# print predictions

orderedpredictions = collections.OrderedDict(sorted(predictions2.items()))

outputfile = open("Snehal_Shirgure_ModelBasedCF-big.txt.txt","w+")

for key,value in orderedpredictions.iteritems():
    outputfile.write(str(key[0])+", "+str(key[1])+", "+str(value))
    outputfile.write("\n")
outputfile.close()

pratings2 = ratings.filter(lambda x: (x[0],1) in test).collect()

pratings2 = dict(pratings2)
orderedpratings = collections.OrderedDict(sorted(pratings2.items()))

count1=0
count2=0
count3=0
count4=0
count5=0


mse1 = 0
count=0

for key,value in orderedpratings.iteritems():
    if(key in orderedpredictions):
        count+=1

        diff = abs(orderedpredictions[key]-value)
        mse1 += diff**2

        if(diff>=0 and diff<1):
            count1+=1
        if(diff>=1 and diff<2):
            count2+=1
        if(diff>=2 and diff<3):
            count3+=1
        if(diff>=3 and diff<4):
            count4+=1
        if(diff>=4 ):
            count5+=1 

mse1 = mse1/count 

print (">=0 and <1: "+str(count1))
print (">=1 and <2: "+str(count2))
print (">=2 and <3: "+str(count3))
print (">=3 and <4: "+str(count4))
print (">=4: "+str(count5))

print("RMSE: " + str(mse**0.5))
# print("RMSE2: " + str(mse1**0.5))
print ("Time: " + str(time.time()-t) +" sec")

# print "prediction length-------->"
# print len(predictions2)
# print "ratings length-------->"
# print len(pratings2)
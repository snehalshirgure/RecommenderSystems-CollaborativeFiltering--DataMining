from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import sys
import numpy
import itertools
import collections
import time
 

inputfile = sys.argv[1]
testfile = sys.argv[2]

t= time.time()

sc = SparkContext('local[10]','task2')

data = sc.textFile(inputfile)
test = sc.textFile(testfile)

ratings = data.map(lambda x: x.split(','))\
    .filter(lambda x: 'userId' not in x )\
    .map(lambda x: (  (int(x[0]),int(x[1]) ), float(x[2]) ) ) 

testdata = test.map(lambda l: l.split(','))\
        .filter(lambda x: 'userId' not in x )\
        .map(lambda l: ( ( int(l[0]), int(l[1]) ) , 1 ) )

testlist = testdata.collect()

pratings = data.map(lambda l: l.split(','))\
    .filter(lambda x: 'userId' not in x )\
    .map(lambda l:  ( ( int(l[0]), int(l[1]) ), float(l[2]) ) )\
    .filter(lambda x: (x[0],1) in testlist)


ratings = ratings.subtractByKey(testdata).map(lambda x: ( x[0][0] , [ ( x[0][1], x[1] ) ])).reduceByKey(lambda x,y: x+y).collect()


averageuserratings = data.map(lambda x: x.split(','))\
        .filter(lambda x: 'userId' not in x )\
        .map( lambda x: (  int(x[0]) , (float(x[2]) , 1.0) ) )\
        .reduceByKey( lambda x, y: (x[0]+y[0],x[1]+y[1])  )\
        .map(lambda x: (x[0],(x[1][0]/x[1][1]) ))\
        .collect()

ratings=dict(ratings)
# movies=dict(movies)
averageuserratings = dict(averageuserratings)

for eachrating in ratings:
    ratings[eachrating] = dict(ratings[eachrating])

for user in ratings:
    for movie in ratings[user]:
        ratings[user][movie]  -= averageuserratings[user]

# print ratings

comb = list(itertools.combinations(ratings.keys(),2))
# print comb

weightlist = {}

for eachcomb in comb:
    movielist1= ratings[eachcomb[0]]
    movielist2= ratings[eachcomb[1]]

    numerator = 0
    denom1 = 0
    denom2 = 0

    for m in movielist1:
        if m in movielist2:
            numerator += float(movielist1[m])*float(movielist2[m])
            denom1 += movielist1[m]**2
            denom2 += movielist2[m]**2
    
    denominator = (denom1**0.5) * (denom2**0.5)
    if(denominator!=0):
        weightlist[eachcomb] = numerator/denominator


lastuserid = len(ratings)

predictions = {}

for each in testlist:
    a = int(each[0][0])
    i = int(each[0][1])
    p = averageuserratings[a]
    numerator = 0
    denominator=0
    for u in range(1,lastuserid+1):
        if i in ratings[u]:
            if (a<u):
                if ((a,u)in weightlist):
                    numerator+= ratings[u][i]*weightlist[(a,u)]
                    denominator += abs(weightlist[(a,u)])
            else:
                if ((u,a)in weightlist):
                    numerator+= ratings[u][i]*weightlist[(u,a)]
                    denominator += abs(weightlist[(u,a)])

    if (denominator!=0):
        p += numerator/denominator

    predictions[(a,i)] = p


ordereddict = collections.OrderedDict(sorted(predictions.items()))

outputfile = open("Snehal_Shirgure_UserBasedCF.txt","w+")

for key,value in ordereddict.iteritems():
    outputfile.write(str(key[0])+", "+str(key[1])+", "+str(value))
    outputfile.write("\n")
outputfile.close()


count1=0
count2=0
count3=0
count4=0
count5=0

pratings = pratings.collect()
pratings = dict(pratings)
orderedpratings = collections.OrderedDict(sorted(pratings.items()))

mse = 0
count=0

for key,value in orderedpratings.iteritems():
    if(key in ordereddict):
        count+=1
        diff = abs(ordereddict[key]-value)
        mse += diff**2
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

mse = mse/count

print (">=0 and <1: "+str(count1))
print (">=1 and <2: "+str(count2))
print (">=2 and <3: "+str(count3))
print (">=3 and <4: "+str(count4))
print (">=4: "+str(count5))

print("RMSE: " + str(mse**0.5))
print ("Time: " + str(time.time()-t) +" sec")


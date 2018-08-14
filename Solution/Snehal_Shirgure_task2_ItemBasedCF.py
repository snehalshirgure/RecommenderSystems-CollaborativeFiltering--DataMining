from pyspark import SparkContext
from random import randint
import itertools
import collections
import sys
import csv
import math
import time

inputfile = sys.argv[1]
testfile = sys.argv[2]


sc = SparkContext('local[10]','task1')

t = time.time()

data = sc.textFile(inputfile)
test = sc.textFile(testfile)

rdd1 = data.map( lambda x: x.split(',') )\
            .filter( lambda x: 'userId' not in x )\
            .map( lambda x: ( int(x[0]) , [int(x[1])] ) )\
            .reduceByKey( lambda x,y: x+y)

rdd2 = data.map( lambda x: x.split(',') )\
            .filter( lambda x: 'userId' not in x )\
            .map( lambda x: ( int(x[1]) , [int(x[0])] ) )\
            .reduceByKey( lambda x,y: x+y)

userlist =  rdd1.sortByKey(ascending=True).collect()
movielist = rdd2.sortByKey(ascending=True).collect()
movielistdict = dict(movielist)
# print movielistdict

users = len(userlist)
movies = len(movielist)

# print ("last userid :---> " + str(userlist[users-1][0]))
# print ("last movieid :---> " + str(movielist[movies-1][0]))

lastuserid = userlist[users-1][0]
lastmovieid = movielist[movies-1][0]

m = lastuserid + 1
#prime number ~
#p= 89

#store all minhash values 
minhash = []
numminhash = 60


#for each min hash function ~
for index in range(0,numminhash):
    
    #formula --> ((ax+b)%p)m
    a= randint(0,1000)
    b= randint(0,1000)
    
    # print("a is: "+str(a) + " b is: "+str(b))

    #initialise dictionary of movieids to infinity(max+1)~
    hashvalues = {}
    for mid in range(movies):
            hashvalues[movielist[mid][0]] = m

    #iterate over all userids ~
    for u in range(users):
        #generate random minhash for current userid ~
        u_mh = (a*u+b)%m   
        #print (str(u) + "-->" + str(u_mh))
        for i in userlist[u][1]:
            if(hashvalues[i] > u_mh):
                hashvalues[i] = u_mh
        #print hashvalues

    minhash.append(hashvalues)

# print minhash

rdd3 = sc.parallelize(minhash)

result = rdd3.map(lambda x: [(k , [x[k]]) for k in x] ).reduce(lambda x,y: x+y)

# print result

rdd4 = sc.parallelize(result)
rdd5 = rdd4.reduceByKey(lambda x,y : x+y)
# for each in rdd5.collect():
#       print each

rdd6 = rdd5.map(lambda x: [ [ ( x[0], x[1][i] ) for i in range(itr,itr+3) ] for itr in range(0,58,3) ] )

bandlist = rdd6.collect()

# for each in bandlist:
#     print each

bucketlist = {}

for index in range(0,20):
    buckets={}
    for bands in bandlist:
        # print bands[index]
        mid = bands[index][0][0]
        # print ("m-id is ------->" + str(mid))
        str1 =''
        for each in bands[index]:
            str1 += ''.join(str(each[1]))
        
        hashbin = hash(str1)

        # print ("hash is : --- "+ str(hashbin))
        if hashbin not in buckets:
            buckets[hashbin] = [mid]
        else:
            buckets[hashbin]+= [mid]
        
    bucketlist[index] = buckets.values()
    del buckets 

# print bucketlist

rdd7 = sc.parallelize(list(bucketlist.values())).reduce(lambda x,y: x+y)
# print rdd7
rdd8 = sc.parallelize(rdd7).filter(lambda x: len(x) > 1 )\
                            .map(lambda x: list(set(x)))\
                            .map(lambda x:  list(itertools.combinations(x,2)))\
                            .reduce(lambda x,y:x+y)

finalpairs = rdd8

similarity = {}

def jaccard(x,y):
    s1 = set(x)
    s2 = set(y)
    return ((len(s1&s2)/float(len(s1|s2))))


for pair in finalpairs:
    a = pair[0]
    b = pair[1]
    
    l1 = movielistdict[pair[0]]
    l2 = movielistdict[pair[1]]
    jaccard_value = jaccard(l1,l2)
    if(a<b):
            if (a,b) not in similarity:
                similarity[(a,b)] = jaccard_value        
    else:
            if (b,a) not in similarity:
                similarity[(b,a)] = jaccard_value 


# print similarity


orderedsimilarity = collections.OrderedDict(sorted(similarity.items()))

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

averageuserratings = dict(averageuserratings)

ratings=dict(ratings)
for eachrating in ratings:
    ratings[eachrating] = dict(ratings[eachrating])


# print ratings

lastuserid = len(ratings)

predictions = {}

for each in testlist:
    a = int(each[0][0])
    i = int(each[0][1])

    p = 0
    numerator = 0
    denominator=0
    for u in range(1,lastuserid+1):
        if i in ratings[u]:
                if ((a,u)in similarity):
                    numerator+= float(ratings[u][i])*float(similarity[(a,u)])
                    denominator += abs(float(similarity[(a,u)]))
                if ((u,a)in similarity):
                    numerator+= float(ratings[u][i])*float(similarity[(u,a)])
                    denominator += abs(float(similarity[(u,a)]))

    if (denominator!=0):
        p += float(numerator)/float(denominator)
    else:
        p = averageuserratings[a]
    
    predictions[(a,i)] = p

print predictions

ordereddict = collections.OrderedDict(sorted(predictions.items()))

outputfile = open("Snehal_Shirgure_ItemBasedCF.txt","w+")

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

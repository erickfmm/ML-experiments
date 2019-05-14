from with_nolib.optimization.population.AFSA.AFSAMH import AFSA
from with_nolib.optimization.population.SillyRandom.Greedy_with_leapMH import GreedyMH

import numpy as np

from numpy.random import RandomState


rnd = RandomState(0)
mydata = rnd.uniform(-100, 100, 200)
#mydata = rnd.randint(-100, 100, 200)
ndim = len(mydata)+2
min_x = 0
max_x = len(mydata)

def is_valid(point):
    are_1s = [0 for _ in range(len(mydata))]
    for i in range(2, len(point)):
        if point[i] < 0 or int(point[i]) >= len(mydata):
            return False
        are_1s[int(point[i])] += 1
        if are_1s[int(point[i])] > 1:
            return False
    for el in are_1s:
        if el == 0 or el > 1:
            return False
    return True
        

def partition_problem_obj(point):
    if not is_valid(point):
        return False
    s1 = point[0]
    s2 = point[1]
    s1 = int(s1/float(s1+s2)*len(mydata))
    s1 = s1 if s1 > 0 and s1 < len(mydata) else 1
    s2 = len(mydata)-s1
    sum1 = 0
    sum2 = 0
    for i in range(2, s1+2, 1):
        sum1 += mydata[int(point[i])]
    for i in range(s1+2, len(point), 1):
        sum2 += mydata[int(point[i])]
    return abs(sum1-sum2)

def subsetsum_problem_obj(point):
    if not is_valid(point):
        return False
    s1 = point[0]
    s2 = point[1]
    s1 = int(s1/float(s1+s2)*len(mydata))
    s1 = s1 if s1 >= 0 and s1 < len(mydata) else 0
    sum1 = 0
    for i in range(2, s1+2, 1):
        sum1 += mydata[int(point[i])]
    return np.abs(sum1)

def repair_partition(point):
    point = np.abs(point)
    are_not = []
    are_1s = [0 for _ in range(len(mydata))]
    #dups = []
    dups = [[] for _ in range(len(mydata))]
    for i in range(2, len(point)):
        if int(point[i]) < 0 or int(point[i]) >= len(mydata):
            point[i] = 0
        are_1s[int(point[i])] += 1
        dups[int(point[i])].append(i)
    for i in range(len(are_1s)):
        if are_1s[i] == 0:
            are_not.append(i)
    sum_dups = []
    for i in range(len(dups)):
        sum_dups.append(len(dups[i]))
        if len(dups[i]) > 0:
            del dups[i][0]
    dups = [e for di in dups for e in di] #flatten
    if len(are_not) != len(dups) or are_1s != sum_dups:
        print("rayooooooooooos son distintos")
        print("arenot: ", len(are_not), ", dups: ", len(dups))
        print("are not: ", are_not)
        print("dups: ", dups)
        print("are1s: ", are_1s)
        point2 = [int(e) for e in point]
        print("point int: ", point2)
        print("point: ", point)
        raise Exception("son distintos :c")
    for i in range(len(dups)):
        point[dups[i]] = are_not[i]
    return point

def preproc_partition(point):
    point2 = np.abs(np.copy(point))
    for i in range(2, len(point2)):
        point2[i] = int(point2[i])
    return point2

print("create afsa")
#mh = AFSA(min_x, max_x, ndim, False, partition_problem_obj, repair_partition, preproc_partition, 113)
#mh = AFSA(min_x, max_x, ndim, False, subsetsum_problem_obj, repair_partition, preproc_partition, None)
print("to run afsa")
#fit, pt = mh.run(visual_distance_percentage=0.5, velocity_percentage=0.5, n_points_to_choose=3, crowded_percentage=0.7, its_stagnation=4, leap_percentage=0.3, stagnation_variation=0.4)
#print(fit)
#print(pt)


print("create Greedy")
mh = GreedyMH(min_x, max_x, ndim, False, partition_problem_obj, repair_partition, preproc_partition, 115)
print("to run greedy")
fit, pt = mh.run(iterations=100, population=30, stagnation_variation=0.4, its_stagnation=5, leap_percentage=0.8)
print(fit)
print(pt)

s1 = pt[0]
s2 = pt[1]
s1 = int(s1/float(s1+s2)*len(mydata))
s1 = s1 if s1 >= 0 and s1 < len(mydata) else 0
s2 = len(mydata)-s1
sum1 = 0
sum2 = 0
sub1 = []
sub2 = []
for i in range(2, s1+2, 1):
    sum1 += mydata[int(pt[i])]
    sub1.append(mydata[int(pt[i])])
for i in range(s1+2, len(pt), 1):
    sum2 += mydata[int(pt[i])]
    sub2.append(mydata[int(pt[i])])

print("sum1:", sum1)
print("sum2:", sum2)
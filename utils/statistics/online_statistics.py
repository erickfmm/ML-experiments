# -*- coding: utf-8 -*-


def movingmean2(n, x, lastmean):
    return 1/float(n) * (x + n*lastmean - lastmean)

#more stable version
def movingmean(n, x, lastmean):
    return lastmean + ((x - lastmean)/float(n))


#s_n is a 
def moving_s(last_s_n, n, x, lastmean):
    now_mean = movingmean(n, x, lastmean)
    #s_n = (variance**2) * n
    #s_n = s_n-1 + (x_n - u_n-1)*(x_n - u_n)
    return last_s_n + (x - lastmean)*(x - now_mean)
    
def movingpvariance(s_n, n):
    return s_n/float(n)


def test():
    import random
    l = [random.random() for _ in range(200)]
    lastmean = 0
    s = 0
    for i in range(len(l)):
        n = i+1
        s = moving_s(s, n, l[i], lastmean)
        lastmean = movingmean(n, l[i], lastmean)
        pvar = movingpvariance(s, n)
    import statistics # to compare
    print("movingmean: ", lastmean)
    print("mean: ", sum(l)/float(len(l)))
    print("stats mean: ", statistics.mean(l))
    print("")
    print("moving variance (population): ", pvar)
    print("variance (population): ", statistics.pvariance(l))
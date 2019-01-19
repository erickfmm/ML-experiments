import math

def gauss_polar(uniform_rand1: float, uniform_rand2: float):
    r_cuad = - math.log(uniform_rand1) / (1/2)
    theta = 2 * math.pi * uniform_rand2
    z1 = math.sqrt(r_cuad) * math.cos(theta)
    z2 = math.sqrt(r_cuad) * math.sin(theta)
    return z1, z2

def gauss_polar_wkp(uniform_rand1: float, uniform_rand2: float): #formula from wikipedia, btwn -1 and 1
    #uniform_rand1 = (uniform_rand1 * 2.0) - 1.0
    #uniform_rand2 = (uniform_rand2 * 2.0) - 1.0
    s = (pow(uniform_rand1, 2) + pow(uniform_rand2, 2)) / 2.0
    z1 = uniform_rand1 * math.sqrt((-2 * math.log(s)) / float(s))
    z2 = uniform_rand2 * math.sqrt((-2 * math.log(s)) / float(s))
    return z1, z2

def gauss_sum_uniform(iterations=1000):
    z = 0
    import random
    for _ in range(iterations):
        z += random.random()
    return z / float(iterations)
#import test.categorical_mh_optimization
from with_nolib.optimization.population.Genetic.GeneticMHCategorical import GeneticMHCategorical

#import numpy as np
from numpy.random import RandomState
from typing import List, Callable, Tuple

print("hello it is categorical mh test")
#general parameters
knapsack_capacity:float = 50.8
total_posible_elements:int = 50
seed = 0
max_value = 5
max_cost = 6
rnd = RandomState(seed)

#list of tuples with value and cost
elements = []
#categories
categories = ["is", "not"]
categories_all_elements = [categories for _ in range(total_posible_elements)]
#fill elements
for _ in range(total_posible_elements):
    elements.append((rnd.uniform(0, max_value), rnd.uniform(0, max_cost)))


def get_values_in(point, verbose):
    sum_elements_in = 0
    elements_value_in = []
    elements_cost_in = []
    for i in range(len(point)):
        if point[i] == "is":
            sum_elements_in +=1
            elements_value_in.append(elements[i][0])
            elements_cost_in.append(elements[i][1])
    if verbose:
        print("sum: ", sum_elements_in)
        print("value total: ", sum(elements_value_in))
        print("cost total: ", sum(elements_cost_in))
        print("max knapsack capacity: ", knapsack_capacity)
        print("values: ", elements_value_in)
        print("costs: ", elements_cost_in)
    return sum_elements_in, elements_value_in, elements_cost_in


def is_valid(point) -> bool:
    total_sum = 0
    for i in range(len(point)):
        if point[i] == "is":
            total_sum += elements[i][1]
    if total_sum > knapsack_capacity:
        return False
    return True

def knapsack_obj(point) -> float:
    if not is_valid(point):
        return False
    total_sum = 0
    for i in range(len(point)):
        if point[i] == "is":
            total_sum += elements[i][0]
    return total_sum

def knapsack_repair(point):
    i = -1
    times_founded = 0
    while not is_valid(point):
        found = False
        while not found:
            i += 1
            if i >= len(point):
                print("que pedo wey, i es mayor que len en repair")
                print("i value: ", i)
                print("times founded: ", times_founded)
                print("punto: ", point)
                sum_elements_in, elements_value_in, elements_cost_in = get_values_in(point, True)
                exit()
            if point[i] == "is":
                point[i] = "not"
                found = True
                times_founded +=1
    return point

mh = GeneticMHCategorical(categorics=categories_all_elements, ndims=total_posible_elements, to_max=True, objective_function=knapsack_obj, repair_function=knapsack_repair)
fit, pt = mh.run(verbose=True)
print("fitness:")
print(fit)
print("point:")
print(pt)
print("movements:")
print(mh.movements)

get_values_in(pt, True)
import random

#there are 3 doors.
#In one door (only one) randomly, there are a car, and you win
#You choose a door (here is chosen randomly because doesn't matter)
#Then you can keep the chosen door or change your option
#Probabilistically ¿Wich option is better?
def game(tochange: bool):
    doors = [0, 1, 2]
    car = random.randint(0, 2)
    chosen = random.randint(0, 2)
    doors.remove(car)
    if chosen in doors: #false when car == chosen
        doors.remove(chosen)
    opened = doors[random.randint(0, len(doors)-1)]
    doors = [0, 1, 2]
    doors.remove(opened)
    if tochange:
        doors.remove(chosen)
        chosen = doors[0] #doors always len 1
    if chosen == car:
        return 1
    else:
        return 0

#make montecarlo simulating tons of games changing and not changing
#then calculate the proportion of wins and returning
def montecarlo(its: int):
    changing = 0
    not_changing = 0
    for it in range(its):
        changing += game(True)
        not_changing += game(False)
    changing = float(changing) / float(its)
    not_changing = float(not_changing) / float(its)
    return (changing, not_changing)

print(montecarlo(100000))
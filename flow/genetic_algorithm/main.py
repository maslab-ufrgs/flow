from copy import deepcopy
from math import ceil
import random
import subprocess
import sys
import pandas as pd
from run import *


DEFAULTVALUE = float('inf')

class Individual:
    def __init__(self, id, time_weights=[], toll_weights=[], host='lab') -> None:
        self.id = id
        self.time_weights = time_weights
        self.toll_weights = toll_weights
        assert len(self.toll_weights) == len(self.time_weights)
        self.num_vehicles = len(self.time_weights)
        self.value = deepcopy(DEFAULTVALUE)
        self.mutationflag = False
        self.weight_path = self.to_csv(host)
        self.subfolder = 0
    
    def __str__(self) -> str:
        return 'id: {};\ntimews: {};\ntollws: {};\nnum_vehicles: {};\nvalue: {};\n'.format(
            self.id, 
            self.time_weights, 
            self.toll_weights, 
            self.num_vehicles, 
            self.value)

    def get_time_weights(self, a, b):
        return self.time_weights[a:b] 

    def get_toll_weights(self, a, b):
        return self.toll_weights[a:b]

    def crossover(self, a, b, new_timews, new_tollws):
        timews = []
        tollws = []
        count = 0
        print(new_timews)
        for i in range(self.num_vehicles):
            if a <= i < b:
                timews.append(new_timews[count])
                tollws.append(new_tollws[count])
                count += 1
            else:
                timews.append(self.time_weights[i])
                tollws.append(self.toll_weights[i])
        return timews, tollws


    def mutation(self):
        self.mutationflag = True
        index = random.randint(0, self.num_vehicles-1)
        timew = random.random()
        tollw = 1 - timew
        self.time_weights[index] = timew
        self.toll_weights[index] = tollw


    def add(self, timew, tollw):
        self.num_vehicles += 1
        assert timew + tollw == 1
        self.time_weights.append(timew)
        self.toll_weights.append(tollw)

    def to_csv(self, host='lab'):
        # individual = [(timew1, tollw1), (timew2, tollw2), ..., (timewM, tollwM)]
        if host == 'lab':
            path = '/home/lab204/Desktop/marco/maslab/flow/flow/inputs/weights/individual_{}.csv'.format(self.id)
        elif host == 'home':
            path = '/home/macsilva/Desktop/maslab/flow/flow/inputs/weights/individual_{}.csv'.format(self.id)
        else:
            quit('error -- to_csv -- invalid host!')
        veh_ids = ['human_{}'.format(i) for i in range(self.num_vehicles)]
        individual_dict = {
            'veh_id':veh_ids,
            'time_weight':self.time_weights,
            'toll_weight':self.toll_weights,
            }
        df = pd.DataFrame(individual_dict)
        df.to_csv(path)
        return path

    def set_value(self, v):
        self.mutationflag = False
        self.value = v

    def get_value(self, exp_tag, num_runs, host):
        if not self.mutationflag and not self.value == DEFAULTVALUE:
            return self.value
        else:
            self.set_value(run(
                self.num_vehicles, 
                self.id, exp_tag, 
                self.weight_path, 
                num_runs=num_runs, 
                individual_subfolder= str(self.subfolder),
                host=host)
            )
            self.subfolder += 1
            return self.value

def ga(num_vehicles, pop_size, num_runs, exp_tag,tournament_size=0, ga_executions=1, host='lab'):
    if  0 < tournament_size <= pop_size:
        tsize = tournament_size
    else:
        tsize = ceil(0.1*pop_size) 
    p = population(num_vehicles, pop_size)
    num_individuals = deepcopy(pop_size)
    for exec in range(ga_executions):
        _p = []  
        while len(_p) < len(p):
            mother = tournament(p, tsize, exp_tag, num_runs, host)
            father = tournament(p, tsize, exp_tag, num_runs, host)
            ind1, ind2, num_individuals = crossover(mother, father, num_vehicles, num_individuals)
            ind1.mutation()
            ind2.mutation()
            _p.append(ind1)
            _p.append(ind2)
        selection(p, _p, pop_size, exp_tag, num_runs, host)
    return p[0] # best

def population(num_vehicles, pop_size):
    p = []
    for i in range(pop_size):
        timews = []
        tollws = []
        for j in range(num_vehicles):
            timew = random.random()
            tollw = 1 - timew
            timews.append(timew)
            tollws.append(tollw)
        p.append(Individual(i, timews, tollws))
    return p


# seleciona k individuos aleatoriamente
def tournament(population, size, exp_tag, num_runs, host):
    best = None
    bestv = float('inf')
    for index in random.sample(range(0, len(population)), size):
        v = population[index].get_value(exp_tag, num_runs, host)
        if v < bestv:
            best = deepcopy(population[index])
    return best

def crossover(mother:Individual, father:Individual, num_vehicles:int, num_individuals:int):
    i = random.randint(0, num_vehicles-2)
    j = random.randint(i+1, num_vehicles-1)
    mother_timews = deepcopy(mother.get_time_weights(i, j))
    mother_tollws = deepcopy(mother.get_toll_weights(i, j))
    father_timews = deepcopy(father.get_time_weights(i, j))
    father_tollws = deepcopy(father.get_toll_weights(i, j))
    i1_timews, i1_tollws = mother.crossover(i, j, father_timews, father_tollws)
    i2_timews, i2_tollws = father.crossover(i, j, mother_timews, mother_tollws)
    num_individuals += 1 
    i1 = Individual(
        id=deepcopy(num_individuals), 
        time_weights=i1_timews,
        toll_weights=i1_tollws)
    num_individuals += 1 
    print(i1)
    i2 = Individual(
        id=deepcopy(num_individuals), 
        time_weights=i2_timews,
        toll_weights=i2_tollws)
    print(i2)
    return i1, i2, num_individuals

def selection(p, _p, pop_size, exp_tag, num_runs, host):
    pop = sorted([ind for ind in p + _p], key=lambda ind: ind.get_value(exp_tag, num_runs, host))
    p = deepcopy(pop[0:pop_size])
    assert len(p) == pop_size
    _p = []

num_vehicles = int(sys.argv[1].split('=')[1])
assert sys.argv[1].split('=')[0] == '--num_vehicles'
pop_size = int(sys.argv[2].split('=')[1])
assert sys.argv[2].split('=')[0] == '--pop_size'
num_runs = int(sys.argv[3].split('=')[1])
assert sys.argv[3].split('=')[0] == '--num_runs'
exp_tag = sys.argv[4].split('=')[1]
assert sys.argv[4].split('=')[0] == '--exp_tag'
tournament_size = int(sys.argv[5].split('=')[1])
assert sys.argv[5].split('=')[0] == '--tournament_size'
ga_executions = int(sys.argv[6].split('=')[1])
assert sys.argv[6].split('=')[0] == '--ga_executions'
host = sys.argv[7].split('=')[1]
assert sys.argv[7].split('=')[0] == '--host'
print(ga(num_vehicles, pop_size, num_runs, exp_tag, tournament_size, ga_executions, host))
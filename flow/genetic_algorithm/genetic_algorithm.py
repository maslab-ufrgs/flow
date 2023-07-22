from time import sleep
from copy import deepcopy
from math import ceil
import random
import csv
import sys
import pandas as pd
from flow.genetic_algorithm.parser import parse_args
from flow.genetic_algorithm.OW.run import *
from multiprocessing import *


DEFAULTVALUE = float('inf')

class Individual:
    def __init__(self, id, time_weights=[], toll_weights=[], host='lab', generation=0, value=DEFAULTVALUE) -> None:
        self.id = id
        self.time_weights = time_weights
        self.toll_weights = toll_weights
        assert len(self.toll_weights) == len(self.time_weights)
        self.generation = generation
        self.num_vehicles = len(self.time_weights)
        self.value = value
        self.mutationflag = False
        self.weight_path = self.to_csv(host)
    
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
        for i in range(self.num_vehicles):
            if a <= i < b:
                timews.append(new_timews[count])
                tollws.append(new_tollws[count])
                count += 1
            else:
                timews.append(self.time_weights[i])
                tollws.append(self.toll_weights[i])
        return timews, tollws


    def mutation(self, generation):
        self.generation = generation
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
            path = '/home/lab204/Desktop/marco/maslab/flow/flow/genetic_algorithm/csv/weights/individual_{}.csv'.format(self.id)
        elif host == 'home':
            path = '/home/macsilva/Desktop/maslab/flow/flow/genetic_algorithm/csv/weights/individual_{}.csv'.format(self.id)
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

    def saveState(self):
        data = [self.id, self.value, self.generation]
        for i in range(len(self.time_weights)):
            data.append(self.time_weights[i])
            data.append(self.toll_weights[i])
        return data

    def set_value(self, v):
        self.mutationflag = False
        self.value = v

    def get_value(self, num_runs, host, valuePath):
        if not self.mutationflag and not self.value == DEFAULTVALUE:
            return self.value
        else:
            self.set_value(run(self.id, host, num_runs))
            self.save_value(valuePath)
            return self.value
        
    def save_value(self, path):
        with open(path, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([self.id, self.value, self.generation])
            file.close()

def callGetValue(ind:Individual, num_runs, host, valuePath):
    print("LOG: running flow over individual #{}".format(ind.id))
    value = ind.get_value(num_runs, host, valuePath)
    print("LOG: individual #{} has a value of {}".format(ind.id, value))
    return ind



def ga(num_vehicles, pop_size, num_runs, valuePath, exp_tag, tournament_size=0, num_generations=1, host='lab', load_state=False):
    state_path = ""
    if host == 'home':
        state_path = '/home/macsilva/Desktop/maslab/flow/flow/genetic_algorithm/csv/state/{}.csv'.format(exp_tag)
    elif host == 'lab':
        state_path = '/home/lab204/Desktop/marco/maslab/flow/flow/genetic_algorithm/csv/state/{}.csv'.format(exp_tag)
    else:
        quit('No host found')  
  
    if  0 < tournament_size <= pop_size:
        tsize = tournament_size
    else:
        tsize = ceil(0.1*pop_size) 

    p = []
    generation = 0
    num_individuals = deepcopy(pop_size)        

    if load_state:
        loaded_state, last_state, num_individuals = loadState(state_path, host)
        if last_state != None:
            p = deepcopy(loaded_state)
            generation = deepcopy(last_state)
            if generation > num_generations:
                exit("LOG: the number of generations of the loaded state exceed the goal.")
        else:
            exit("LOG: could not load state that was required.")
        print("LOG: loaded state at genration #{}.".format(generation))
        for ind in p:
            print("-> individual #{}, with value {} successfully loaded.".format(ind.id, ind.value))    
    else:
        p = population(num_vehicles, pop_size, host, 0)
        p = populationValue(p, num_runs, host, valuePath)
        print("LOG: population generated with sucess.")
    
    while generation <= num_generations:
        generation += 1
        _p = []  
        while len(_p) < len(p):
            mother = tournament(p, tsize, num_runs, host, valuePath)
            father = tournament(p, tsize, num_runs, host, valuePath)
            ind1, ind2, num_individuals = crossover(mother, father, num_vehicles, num_individuals, host, generation)
            ind1.mutation(generation)
            ind2.mutation(generation)
            _p.append(ind1)
            _p.append(ind2)
        _p = populationValue(_p, num_runs, host, valuePath)
        p, _p = selection(p, _p, pop_size, num_runs, host, valuePath)
        print("LOG: selected individuals and its values to be the next population P:")
        for i in p:
            print("id: {}, value: {}".format(i.id, i.value))
        saveState(p, num_vehicles, generation, state_path, num_individuals)
    return p[0] # best

def population(num_vehicles, pop_size, host, generation):
    p = []
    for i in range(pop_size):
        timews = []
        tollws = []
        for j in range(num_vehicles):
            timew = random.random()
            tollw = 1 - timew
            timews.append(timew)
            tollws.append(tollw)
        p.append(Individual(i, timews, tollws, host, generation))
    return p

def populationValue(population, num_runs, host, valuePath):
    p = []
    with Pool() as pool:
        args = []
        for individual in population:
            args.append((individual, num_runs, host, valuePath))
        for result in pool.starmap(callGetValue, args):
            p.append(result)
            print("LOG: individual #{} has its value {} computed!".format(result.id, result.value))
    return p


# seleciona k individuos aleatoriamente
def tournament(population, size, num_runs, host, valuePath):
    best = None
    bestv = float('inf')
    for index in random.sample(range(0, len(population)), size):
        v = population[index].get_value(num_runs, host, valuePath)
        if v < bestv:
            best = deepcopy(population[index])
    return best

def crossover(mother:Individual, father:Individual, num_vehicles:int, num_individuals:int, host:str, generation:int):
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
        toll_weights=i1_tollws,
        host=host,
        generation=generation)
    num_individuals += 1 
    i2 = Individual(
        id=deepcopy(num_individuals), 
        time_weights=i2_timews,
        toll_weights=i2_tollws,
        host=host,
        generation=generation)
    return i1, i2, num_individuals

def selection(p, _p, pop_size,  num_runs, host, valuePath):
    best_individuals = []
    selection_poll = [ind for ind in p + _p]
    for i in range(ceil(pop_size/2)):
        best = max(
            selection_poll, 
            key=lambda ind: ind.get_value(num_runs, host, valuePath)
        )
        selection_poll.remove(best)
        best_individuals.append(best)
    normal_individuals = random.sample(selection_poll, floor(pop_size/2))
    assert len(best_individuals) + len(normal_individuals) == pop_size
    _p = []
    for i in selection_poll:
        if i.id not in [n.id for n in normal_individuals]: 
            remove_routes(i.id, host)
    return best_individuals + normal_individuals, _p


def saveState(population, num_vehicles, last_state, state_path, num_individuals):
    print("LOG: saving state")
    header = ["last_state", "num_individuals", "id", "value", "generation"]
    for i in range(num_vehicles):
        header.append("timew_{}".format(i))
        header.append("tollw_{}".format(i))
    with open(state_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for individual in population:
            data = individual.saveState()
            data = [last_state, num_individuals] + data
            writer.writerow(data)

def loadState(state_path, host):
    population = []
    last_state = 0
    num_individuals = 0
    if not os.path.exists(state_path):
        return None, None, None
    df = pd.read_csv(state_path)
    if not df.empty:
        print("LOG: loading state")
        for row_i, row in df.iterrows():
            id = None
            ind_value = None
            generation = None
            time_weights = []
            toll_weights = []
            counter = -1
            for column_i, value in row.items():
                counter += 1
                if counter == 0:
                    last_state = int(value)
                elif counter == 1:
                    num_individuals = int(value)
                elif counter == 2:
                    id = int(value)
                elif counter == 3:
                    ind_value = float(value)
                elif counter == 4:
                    generation = int(value)
                elif counter > 4 and counter % 2 == 1:
                    time_weights.append(float(value))
                elif counter > 4 and counter % 2 == 0:
                    toll_weights.append(float(value))
            # print(last_state)
            # print(num_individuals)
            print(id)
            # print(ind_value)
            # print(generation)
            # print(time_weights)
            # print(toll_weights)
            ind = Individual(
                id=id, 
                time_weights=time_weights, 
                toll_weights=toll_weights, 
                host=host, 
                generation=generation,
                value=ind_value)
            population.append(ind)
        return population, last_state, num_individuals
    else:
        return None, None, None


args = parse_args(sys.argv[1:])
valuepath = ""
if args.host == 'home':
    bestpath = '/home/macsilva/Desktop/maslab/flow/flow/genetic_algorithm/csv/best/{}.txt'.format(args.exp_tag)
    valuepath = '/home/macsilva/Desktop/maslab/flow/flow/genetic_algorithm/csv/values/{}.csv'.format(args.exp_tag)
elif args.host == 'lab':
    bestpath = '/home/lab204/Desktop/marco/maslab/flow/flow/genetic_algorithm/csv/best/{}.txt'.format(args.exp_tag)
    valuepath = '/home/lab204/Desktop/marco/maslab/flow/flow/genetic_algorithm/csv/values/{}.csv'.format(args.exp_tag)
else:
    quit('No host found')

with open(valuepath, "w") as file:
    writer = csv.writer(file)
    writer.writerow(["individual_id", "value", "generation"])
    file.close()

best = ga(
    args.num_vehicles, 
    args.pop_size, 
    args.num_runs, 
    valuepath,
    args.exp_tag,
    args.tournament_size, 
    args.num_generations, 
    args.host,
    args.load_state
    )

with open(bestpath, 'w') as file:
    file.write(str(best))
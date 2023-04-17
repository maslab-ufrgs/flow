from email import header
import linecache
import csv
import pandas as pd
from copy import deepcopy
from math import floor
from turtle import position
from flow.networks import Network
from enum import Enum, auto
import numpy as np
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams

# case 0: variable belongs to [1, ..., n-1]
CASE0_POSSIBLE_CHOICES = [-1,0,1]
CASE0_PROBABILITY_DISTRIBUTION = [1/4,1/4,1/2]
# case 1: variable is equal to 0
CASE1_POSSIBLE_CHOICES = [0,1]
CASE1_PROBABILITY_DISTRIBUTION = [1/3, 2/3]

# number of_nodes
# TODO: alterar código para passar valores V e H como parâmetros.
NUMBER_OF_VERTICAL_NODES = 5
NUMBER_OF_HORIZONTAL_NODES = 5

# (start, end) points for each vehicle
network_vehicles_data = {}

# edges_costs[edge:str] = edge_cost:float
edges_toll_cost = {}

# input file types
# node_to_node: file format is s,t,p where:
#   s (str) is start node
#   t (str) is termination node
#   p (float) is probability of selecting the (s,t) combination of nodes
# edge_to_node: file format is e,t,p where:
#   e (str) is the first edge that will be taken by the vehicle
#   t (str) is termination node
#   p (float) is probability of selecting the (e,t) combination of edge and node
InputFileType = ["node_to_node", "edge_to_node",'edges_toll_cost']

class InputFileVehicleData:
    def __init__(self, vehicle_id, selected_lane,first_edge, termination_node, starting_node=None, time_weight=None, toll_weight=None):
        self.vehicle_id = vehicle_id
        self.selected_lane = selected_lane
        self.first_edge = first_edge
        self.termination_node = termination_node
        self.time_weight = time_weight
        self.toll_weight = toll_weight
        if not starting_node is None:
            self.starting_node = starting_node
        else:
            node_from, node_to = split_string_node_id(first_edge)
            self.starting_node = node_from

    def __str__(self) -> str:
        return "id: {}; src {} to dst {} (edge: {}); timew: {}; tollw: {};".format(
            self.vehicle_id,
            self.starting_node,
            self.termination_node,
            self.first_edge,
            self.time_weight,
            self.toll_weight
        )
    
    def get_src_and_dst(self):
        return self.starting_node, self.termination_node

    def weight(self, time, toll):
        self.time_weight = time
        self.toll_weight = toll

    def get_destiny(self):
        return self.termination_node

    def get_timew(self):
        return self.time_weight

    def get_tollw(self):
        return self.toll_weight


class EdgesOrientation(Enum):
    RIGHT = auto()
    LEFT = auto()
    UP = auto()
    DOWN = auto()
    def __str__(self):
        return str(format(self.value))



ADDITIONAL_NET_PARAMS = {
    "num_lanes": 1,
    "speed_limit": 10,
    "lane_length": 100,
    "use_input_file_to_get_starting_positions": False,
    "input_file_path": "flow/inputs/networks/5by5test.txt",
}

# define the network class, and inherit properties from the base network class
class MyGrid(Network):
    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize a ring scenario."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)
    rts_weights = {}

    @staticmethod
    def gen_custom_start_pos( cls, net_params, initial_config, num_vehicles):
        if "start_positions" in network_vehicles_data and "start_lanes" in network_vehicles_data:
            return network_vehicles_data["start_positions"], network_vehicles_data["start_lanes"]
        else:
            start_lanes = []        # list of int
            start_positions = []    # list of tuples
            use_input_file = net_params.additional_params["use_input_file_to_get_starting_positions"]
            if use_input_file:
                path = net_params.additional_params["input_file_path"]
                gen_custom_start_pos_using_input_file_data(net_params, num_vehicles, start_lanes, start_positions, path)
            else:
                gen_custom_start_pos_using_network_data(net_params, num_vehicles, start_lanes, start_positions)
            return start_positions, start_lanes
    
    # ok
    def generate_edges(self, orientation, speed_limit, edge_length, num_lanes,):
        if orientation is EdgesOrientation.RIGHT or orientation is EdgesOrientation.LEFT:
            y_range = NUMBER_OF_VERTICAL_NODES
            x_range =   NUMBER_OF_HORIZONTAL_NODES - 1
            edges = []
            for y in range(y_range):
                for x in range(x_range):
                    # RIGHT: 00 -> 01 -> ... -> 0N
                    if orientation is EdgesOrientation.RIGHT:
                        new_edge_from = node_id(x, y)
                        new_edge_to = node_id(x+1, y)
                    # LEFT: 00 <- 01 <- ... <- 0N
                    else: 
                        new_edge_from = node_id(x_range - x, y)
                        new_edge_to = node_id(x_range - (x+1), y)                 
                    new_edge = {
                        "id": "from"+new_edge_from+"to"+new_edge_to,
                        "numLanes": num_lanes, 
                        "speed": speed_limit,
                        "from": new_edge_from, 
                        "to": new_edge_to, 
                        "length": edge_length,                 
                    }
                    edges.append(new_edge)
            return edges
        elif orientation is EdgesOrientation.UP or orientation is EdgesOrientation.DOWN:
            y_range = NUMBER_OF_VERTICAL_NODES - 1
            x_range =   NUMBER_OF_HORIZONTAL_NODES
            edges = []
            for x in range(x_range):
                for y in range(y_range):
                    # UP:
                    # N0
                    # ^
                    # |
                    # ...
                    # ^
                    # |
                    # 10
                    # ^
                    # |
                    # 00
                    if orientation is EdgesOrientation.UP:
                        new_edge_from = node_id(x, y)
                        new_edge_to = node_id(x, y+1)
                    # DOWN:
                    # N0
                    # |
                    # v
                    # ...
                    # |
                    # v
                    # 10
                    # |
                    # v
                    # 00
                    else: 
                        new_edge_from = node_id(x, y_range - y)
                        new_edge_to = node_id(x, y_range - (y+1))         
                    new_edge = {
                        "id": "from"+new_edge_from+"to"+new_edge_to,
                        "numLanes": num_lanes, 
                        "speed": speed_limit,
                        "from": new_edge_from, 
                        "to": new_edge_to, 
                        "length": edge_length,                 
                    }
                    edges.append(new_edge)
            return edges
        return []

    # check
    def specify_nodes(self, net_params):
        y_range = NUMBER_OF_VERTICAL_NODES
        x_range =   NUMBER_OF_HORIZONTAL_NODES
        length = net_params.additional_params["lane_length"]
        nodes = []
        new_nodes = []
        for y in range(y_range):
            for x in range(x_range):
                new_nodes.append(node_id(x, y))
                new_node = {
                    "id":   node_id(x,y),
                    "x":    x*length,
                    "y":    y*length,
                }
                nodes.append(new_node)

        return nodes

    # check
    def specify_edges(self, net_params):
        edge_length = net_params.additional_params["lane_length"]
        num_lanes = net_params.additional_params["num_lanes"]
        speed_limit = net_params.additional_params["speed_limit"]
        edges = []

        for orientation in EdgesOrientation:
            edges.extend(self.generate_edges(orientation, speed_limit, edge_length, num_lanes))

        return edges

    # check
    def specify_routes(self, net_params):
        rts = {}

        # destiny = (NUMBER_OF_HORIZONTAL_NODES-1, NUMBER_OF_VERTICAL_NODES-1)
        # random_routes = generating_random_routes(destiny)
        # rts.update(random_routes)

        lane_length =  net_params.additional_params["lane_length"]
        for orientation in EdgesOrientation: 
            rts.update(self.generate_routes(orientation, lane_length))
        return rts
    
    # ok
    def generate_routes(self, orientation, lane_length):
        y_range = NUMBER_OF_VERTICAL_NODES
        x_range =   NUMBER_OF_HORIZONTAL_NODES
        routes = {}
        if orientation is EdgesOrientation.RIGHT or orientation is EdgesOrientation.LEFT:
            x_range -= 1
            for y in range(y_range):
                for x in range(x_range):
                    c = deepcopy(x)
                    new_route = []
                    while c <= x_range-1:
                        if orientation is EdgesOrientation.RIGHT:
                            edge_from = node_id(c, y)
                            edge_to = node_id(c+1, y)
                        else:
                            edge_from = node_id(x_range-(c), y)
                            edge_to = node_id(x_range-(c+1), y) 
                        edge = "from"+edge_from+"to"+edge_to
                        new_route.append(edge)
                        c += 1
                    if len(new_route)>0:
                        routes[new_route[0]] = new_route

        elif orientation is EdgesOrientation.UP or orientation is EdgesOrientation.DOWN:
            y_range -= 1
            for x in range(x_range):
                for y in range(y_range):
                    c = deepcopy(y)
                    new_route = []
                    while c <= y_range-1 :
                        if orientation is EdgesOrientation.UP:
                            edge_from = node_id(x,c)
                            edge_to = node_id(x,c+1)
                        else:
                            edge_from = node_id(x,y_range - c)
                            edge_to = node_id(x,y_range - (c+1))  
                        edge = "from"+edge_from+"to"+edge_to
                        new_route.append(edge)
                        c += 1
                    if len(new_route)>0:
                        routes[new_route[0]] = new_route
        return routes

    # check
    def specify_edge_starts(self):
        lane_length = self.net_params.additional_params["lane_length"]
        edgestarts = []

        for orientation in EdgesOrientation:
            edgestarts.extend(self.generate_edgestarts(orientation, lane_length))

        return edgestarts

    # ok
    def generate_edgestarts(self, orientation, lane_length):
        y_range = NUMBER_OF_VERTICAL_NODES
        x_range =   NUMBER_OF_HORIZONTAL_NODES
        edgestarts = []
        edge_from = ""
        edge_to = ""
        if orientation is EdgesOrientation.RIGHT or orientation is EdgesOrientation.LEFT:
            x_range -= 1
            for y in range(y_range):
                for x in range(x_range):
                    if orientation is EdgesOrientation.RIGHT:
                        edge_from = node_id(x,y)
                        edge_to = node_id(x+1,y)
                    else:
                        edge_from = node_id(x_range - x,y)
                        edge_to = node_id(x_range - (x+1),y)
                    new_edgestart = ("from"+edge_from+"to"+edge_to, y*lane_length)
                    edgestarts.append(new_edgestart)

        elif orientation is EdgesOrientation.UP or orientation is EdgesOrientation.DOWN:
            y_range -= 1
            pos = 0
            for x in range(x_range):
                for y in range(y_range):
                    if orientation is EdgesOrientation.UP:
                        edge_from = node_id(x,y)
                        edge_to = node_id(x,y+1)
                        pos = y
                    else: 
                        edge_from = node_id(x,y_range - y)
                        edge_to = node_id(x,y_range - (y+1))       
                        pos = y_range - y   
                    new_edgestart = ("from"+edge_from+"to"+edge_to, pos*lane_length)
                    edgestarts.append(new_edgestart)
        return edgestarts

    def read_souce_and_termination_input_text_file(cls, file_path):
        probability_distribution = []
        nodes = []
        count = 0
        with open(file_path) as f:
            for line in f.readlines():
                if count == 0 and not line[0] == '#':
                    count += 1
                    splited_line = line.split(',')
                    horizontal, vertical = int(splited_line[0]), int(splited_line[1])
                    if not horizontal == NUMBER_OF_HORIZONTAL_NODES or not vertical == NUMBER_OF_VERTICAL_NODES:
                        raise
                elif count > 0 and not line[0] == '#':
                    count += 1
                    splited_line = line.split(',')
                    s, t, p = splited_line[0], splited_line[1], float(splited_line[2])
                    probability_distribution.append(p)
                    nodes.append((s,t))
        return nodes, probability_distribution

def generating_random_routes(destiny):
    x_range = NUMBER_OF_HORIZONTAL_NODES
    y_range = NUMBER_OF_VERTICAL_NODES
    routes = {}
    for  y in range(y_range):
        for x in range(x_range -1):
            # +1 in x first; after, +1 in y
            last_pos = (x, y)
            if(not (x == destiny[0] and y == destiny[1])):
                first_edge_from = node_id(last_pos[0], last_pos[1])
                current_pos = (x+1, y)
                first_edge_to = node_id(current_pos[0], current_pos[1])
                first_edge = "from"+first_edge_from+"to"+first_edge_to
                routes.update(generating_random_route(current_pos, destiny, first_edge))
    for x in range(x_range):
        for y in range(y_range -1): 
            last_pos = (x, y)
            if(not (x == destiny[0] and y == destiny[1])):
                first_edge_from = node_id(last_pos[0], last_pos[1])
                current_pos = (x, y+1)
                first_edge_to = node_id(current_pos[0], current_pos[1])
                first_edge = "from"+first_edge_from+"to"+first_edge_to
                routes.update(generating_random_route(current_pos, destiny, first_edge))     
    for  y in range(y_range):
        for x in reversed(range(x_range -1)):
            # +1 in x first; after, +1 in y
            last_pos = (x+1, y)
            if(not (x == destiny[0] and y == destiny[1])):
                first_edge_from = node_id(last_pos[0], last_pos[1])
                current_pos = (x, y)
                first_edge_to = node_id(current_pos[0], current_pos[1])
                first_edge = "from"+first_edge_from+"to"+first_edge_to
                routes.update(generating_random_route(current_pos, destiny, first_edge))
    for x in range(x_range):
        for  y in reversed(range(y_range-1)):
            # +1 in x first; after, +1 in y
            last_pos = (x, y+1)
            if(not (x == destiny[0] and y == destiny[1])):
                first_edge_from = node_id(last_pos[0], last_pos[1])
                current_pos = (x, y)
                first_edge_to = node_id(current_pos[0], current_pos[1])
                first_edge = "from"+first_edge_from+"to"+first_edge_to
                routes.update(generating_random_route(current_pos, destiny, first_edge))
    return routes


def generating_random_route( source, destiny, first_edge):
    """
        num_routes: int
        source: (x:int, y:int)
        destiny: (x:int, y:int)
        
        This generates num_routes random routes from source to destiny
    """
    new_route = {}
    path = [first_edge]
    x = deepcopy(source[0])
    y = deepcopy(source[1])
    while x < destiny[0] or y < destiny[1]:
        last_x = deepcopy(x)
        last_y = deepcopy(y)
        # x = 0
        # y = 1
        choosen_var = np.random.choice([0,1], size=1, p=[1/2,1/2])[0]
        if choosen_var == 0: 
            # se tiver como atualizar o x, atualiza. Senão, atualiza o y.
            x, y = apply_update(x, y, destiny)
        else:
            # se tiver como atualizar o y, atualiza. Senão, atualiza o x.
            y, x = apply_update(y, x, (destiny[1], destiny[0]))
        # obriga que pelo menos um tenha sido atualizado
        if x == last_x and y == last_y:
            if x < destiny[0] and y == destiny[1]: 
                x += 1
            elif x == destiny[0] and y < destiny[1]:
                y += 1
            else:
                x += (1- choosen_var)
                y += choosen_var
        
        edge_from = node_id(last_x,last_y)
        edge_to = node_id(x,y)        
        
        # verificar se não há um loop entre dois arcos consecutivos, pois isso acarreta em erro
        # ex.: fromAtoB, fromBtoA não pode ocorrer para todos A, B pertencentes ao conjunto de vértices do grafo     
        last_edge = deepcopy(path[-1])
        last_nodes = last_edge.split("to")
        last_node_from = last_nodes[0].replace("from","")
        if last_node_from == edge_to:
            undo_self_loop(path, edge_to, edge_from)
        else:
            edge = "from"+edge_from+"to"+edge_to
            path.append(edge)
    new_route[first_edge] = path
    return new_route

def update_coordinate(s, possible_choices, probability_distribution):
    random_choice = np.random.choice(possible_choices, size=1, p=probability_distribution)[0]
    s += random_choice
    if s < 0: 
        s = 0
    return s

def apply_update(x, y, destiny):
    if 0 < x < destiny[0]:
        x = update_coordinate(x,  CASE0_POSSIBLE_CHOICES, CASE0_PROBABILITY_DISTRIBUTION)
    elif x == 0:
        x = update_coordinate(x, CASE1_POSSIBLE_CHOICES, CASE1_PROBABILITY_DISTRIBUTION)
    else:
        # x = desntiny[0]
        if y == 0:
            y = update_coordinate(y, CASE1_POSSIBLE_CHOICES, CASE1_PROBABILITY_DISTRIBUTION)
        elif 0 < y < destiny[1]: 
            y = update_coordinate(y, CASE0_POSSIBLE_CHOICES, CASE0_PROBABILITY_DISTRIBUTION)
    return x, y

def undo_self_loop(list_of_edges, A, B):
    """
        Esta função é chamada no caso de encontrarmos um loop entre dois nodos, i.e, A -> B -> A, mas não temos como realizar essa voltar, pois acarretaria em contramão.
        Logo, precisamos desfazer o loop através de um contorno(
        seja ele vertical, no caso do percurso A -> B for horizontal, 
        seja ele horizontal, no caso do percurso A -> B for vertical)

        list_of_edges: string list: rota criada até o momento. A última rota adicionada deve ser fromAtoB.
        A: string: nodo de partida do último arco adicionado na rota e nodo de chegada do último arco que queremos adicionar na rota.
        B: string: nodo de chegada do último arco adicionado na rota e nodo de partida do último arco que queremos adicionar na rota.

        Pra ilucidar melhor, essa função foi chamada no seguinte caso: 
        list_of_edges = [..., fromAtoB]
        next_edge = fromBtoA
        undo_Self_loop(list_of_edges, A, B) 
    """
    x_string_pos = int(floor(NUMBER_OF_HORIZONTAL_NODES/ 10)+1)
    y_string_pos = int(floor(NUMBER_OF_VERTICAL_NODES/10) + 1)

    Ax = int(A[0:x_string_pos])
    Ay = int(A[x_string_pos: x_string_pos + y_string_pos])
    Bx = int(B[0:x_string_pos])
    By = int(B[x_string_pos: x_string_pos + y_string_pos])

    if Ay == By:
        # sentido: direita (A -> B), esquerda (B -> A)
        if Ax < Bx: 
            alteracao = -1
        # sentido: esquerda (A -> B), direita (B -> A)
        else: 
            alteracao = 1
        
        # verificar se eu posso fazer o contorno vertical. Se eu não puder, não adiciona a aresta que realizaria o loop no percurso.
        if NUMBER_OF_VERTICAL_NODES <= 1: 
            new_sequence_of_edges = []
        # contorno por cima
        elif Ay == 0: 
            new_sequence_of_edges = [(Bx, By+1), (Bx + alteracao, By+1), (Bx + alteracao, By)]
        # contorno por baixo
        else: 
            new_sequence_of_edges = [(Bx, By-1), (Bx + alteracao, By-1), (Bx + alteracao, By)]
    
   
    elif Ax == Bx:
        # sentido: cima (A -> B), baixo (B -> A)
        if Ay < By:
            alteracao = -1
        # sentido: baixo (A -> B), cima (B -> A)
        else: 
            alteracao = 1
        
        # verificar se eu posso fazer o contorno horizontal. Se eu não puder, não adiciona a aresta que realizaria o loop no percurso.
        if NUMBER_OF_HORIZONTAL_NODES <= 1:
            new_sequence_of_edges = []
        # contorno pela direita
        if Bx == 0:
            new_sequence_of_edges = [(Bx+1, By), (Bx +1, By+alteracao), (Bx, By + alteracao)]
        # contorno por baixo
        else:
            new_sequence_of_edges = [(Bx-1, By), (Bx -1, By+alteracao), (Bx, By + alteracao)]
    
    else:
        new_sequence_of_edges = []

    last_node = node_id(By, Bx)
    for tup in new_sequence_of_edges:
        current_node = node_id(tup[1], tup[0])
        edge_from = "from" + last_node
        edge_to = "to" + current_node
        edge = edge_from + edge_to
        list_of_edges.append(edge)
        last_node = deepcopy(current_node)

def node_id(x, y):
    horizontal_zeros = int(NUMBER_OF_HORIZONTAL_NODES/ 10) +1
    vertical_zeros = int(NUMBER_OF_VERTICAL_NODES / 10) +1
    y_str = make_string_for_node_id(vertical_zeros, y)
    x_str = make_string_for_node_id(horizontal_zeros, x)
    node_id = y_str + x_str 
    return node_id



def make_string_for_node_id(number_of_zeros, x):
    x_ = deepcopy(x)
    x_str = ""
    if(x == 0):
        for i in range(number_of_zeros):
            x_str += "0"
    else:
        for i in range(number_of_zeros): 
            if(x_ % pow(10, i) >= x_):
                x_str += "0"
            elif(x_ % pow(10, i) < x_):
                x_ = int(x_ / pow(10, i))
        x_str += str(x)
    return x_str



def read_source_node_to_termination_node_input_file(file_path, data, probability_distribution):
    count = 0
    file_type = ""
    with open(file_path) as f:
        for line in f.readlines():
            if count == 0 and not line[0] == '#':
                count += 1
                file_type = line.replace("\n", "")
                if not file_type in InputFileType:
                    raise
            elif count == 1 and not line[0] == '#':
                count += 1
                splited_line = line.split(',')
                horizontal, vertical = int(splited_line[0]), int(splited_line[1])
                if not horizontal == NUMBER_OF_HORIZONTAL_NODES or not vertical == NUMBER_OF_VERTICAL_NODES:
                    raise
            elif count > 1 and not line[0] == '#':
                count += 1
                store_line_content_from_input_file_data(data, probability_distribution, line, file_type)
    return file_type

def store_line_content_from_input_file_data(data, probability_distribution, line, file_type):
    splited_line = line.split(',')
    if file_type == "node_to_node":
        s, t, p = splited_line[0], splited_line[1], float(splited_line[2])
        probability_distribution.append(p)
        data.append((s,t))
    elif file_type == "edge_to_node":
        e, t, p = splited_line[0], splited_line[1], float(splited_line[2])
        probability_distribution.append(p)
        data.append((e, t))
    else:
        raise

def gen_custom_start_pos_using_input_file_data(net_params, num_vehicles, start_lanes, start_positions, file_path):
    probability_distribution = []
    data = []
    file_type = read_source_node_to_termination_node_input_file(file_path, data, probability_distribution)
    if file_type == "node_to_node":
        gen_custom_start_pos_using_input_file_data_in_node_to_node_format(net_params, num_vehicles, start_lanes, start_positions, data, probability_distribution)
    elif file_type == "edge_to_node":
        gen_custom_start_pos_using_input_file_data_in_edge_to_node_format(net_params, num_vehicles, start_lanes, start_positions, data, probability_distribution)
    else:
        raise

def gen_custom_start_pos_using_network_data( net_params, num_vehicles, start_lanes, start_positions):
    print('##########################################')
    num_lanes = net_params.additional_params["num_lanes"]
    lane_length = net_params.additional_params["lane_length"]
    dst_x = np.random.choice(NUMBER_OF_HORIZONTAL_NODES, size=1)[0]
    dst_y = np.random.choice(NUMBER_OF_VERTICAL_NODES, size=1)[0]        
    dst = str(dst_y) + str(dst_x)
    borders = []
    for y in range(NUMBER_OF_VERTICAL_NODES-1):
        for x in range(NUMBER_OF_HORIZONTAL_NODES-1):
            if y == 0 or x == 0 or y == NUMBER_OF_VERTICAL_NODES-1 or x == NUMBER_OF_HORIZONTAL_NODES-1:
                borders.append((x, y))

    for veh in range(num_vehicles):
        # select over which lane the vehicle should start upon
        lane = np.random.choice(num_lanes, size=1)[0]
        start_lanes.append(lane)
        src_x = deepcopy(dst_x)
        src_y = deepcopy(dst_y)
        # select what should be the first node
        while src_x == dst_x and src_y == dst_y:
            src_x = np.random.choice(NUMBER_OF_HORIZONTAL_NODES, size=1)[0]
            src_y = np.random.choice(NUMBER_OF_VERTICAL_NODES, size=1)[0]     
            if src_x == NUMBER_OF_HORIZONTAL_NODES-1 and src_y == NUMBER_OF_VERTICAL_NODES -1 and not src_x == dst_x or not src_y == dst_y:
                break
            elif (src_x, src_y) in borders and not src_x == dst_x or not src_y == dst_y:
                break
            else: 
                to_be_changed = np.random.choice([0,1], size=1)[0]
                new_value = np.random.choice([0,1], size=1)[0]
                if to_be_changed == 0:
                    src_x = new_value*(NUMBER_OF_HORIZONTAL_NODES-1)
                else:
                    src_y = new_value*(NUMBER_OF_VERTICAL_NODES-1)
        src = str(src_y) + str(src_x)
        select_what_node_should_go_next(net_params, start_positions, src[0], src[1])
        veh_id = 'human_' + str(veh)
        network_vehicles_data[veh_id] = InputFileVehicleData(
            vehicle_id=veh_id, 
            selected_lane=lane, 
            first_edge=start_positions[-1], 
            termination_node=dst, 
            starting_node=src)
    network_vehicles_data["start_positions"] = start_positions
    network_vehicles_data["start_lanes"] = start_lanes
    read_weight_input(net_params)


def gen_custom_start_pos_using_input_file_data_in_edge_to_node_format(net_params, num_vehicles, start_lanes, start_positions, edges, probability_distribution):
    num_lanes = net_params.additional_params["num_lanes"]
    lane_length = net_params.additional_params["lane_length"]
    for veh in range(num_vehicles):
        found = False
        edge_pos = None
        while not found:
            # select over which lane the vehicle should start upon
            lane = np.random.choice(num_lanes, size=1)[0]
            start_lanes.append(lane)
            # select over which edge the vehicle should start upon, using a input file to get probability distribution
            index = np.random.choice(len(edges), size=1, p=probability_distribution)[0]
            edge = edges[index][0]
            termination_node = edges[index][1]
            edge_pos = find_a_position_to_insert_new_vehicle_into_the_selected_edge(edge, start_positions, lane_length)
            if edge_pos is not None:
                found = True
        start_positions.append((edge, edge_pos))
        # TODO: alterar isso! Ver se há uma forma de recuperar os id's dos veículos.
        veh_id = 'human_' + str(veh)
        network_vehicles_data[veh_id] = InputFileVehicleData(veh_id, lane, edge, termination_node)
    network_vehicles_data["start_positions"] = start_positions
    network_vehicles_data["start_lanes"] = start_lanes
    read_weight_input(net_params)

def gen_custom_start_pos_using_input_file_data_in_node_to_node_format(net_params, num_vehicles, start_lanes, start_positions, nodes, probability_distribution):
    num_lanes = net_params.additional_params["num_lanes"]
    for veh in range(num_vehicles):
        # select over which lane the vehicle should start upon       
        lane = np.random.choice(num_lanes, size=1)[0]
        start_lanes.append(lane)
        # select over which edge the vehicle should start upon, using a input file to get probability distribution
        index = np.random.choice(len(nodes), size=1, p=probability_distribution)[0]
        s = nodes[index][0]
        t = nodes[index][1]
        # TODO: alterar isso! Ver se há uma forma de recuperar os id's dos veículos.
        veh_id = 'human_' + str(veh)
        select_what_node_should_go_next(net_params, start_positions, s[0], s[1])
        network_vehicles_data[veh_id] = InputFileVehicleData(veh_id, lane, start_positions[-1], t, starting_node=s)
    network_vehicles_data["start_positions"] = start_positions
    network_vehicles_data["start_lanes"] = start_lanes
    read_weight_input(net_params)



def select_what_node_should_go_next(net_params, start_positions, v, h):
    lane_length = net_params.additional_params["lane_length"]  
    found = False
    initial_node = v + h
    edge_position = None
    while(not found):
        possible_choices_v = [-1,0,1]
        possible_choides_h = [-1,0,1]
        
        # borders:
        if int(v) == 0:
            possible_choices_v = [0, 1]
        elif int(v) == (NUMBER_OF_VERTICAL_NODES -1):
            possible_choices_v = [-1, 0]                
        if int(h) == 0:
            possible_choides_h = [0, 1]
        elif int(h) == (NUMBER_OF_HORIZONTAL_NODES -1):
            possible_choides_h = [-1, 0]

        # if isnt, select a neighbor node to go next
        increment_h = 0
        increment_v = 0
        # at least one of them (v or h) have to be different than 0 to next node be also different than initial node.
        while increment_h == 0 and increment_v == 0:
            select_which_direction_to_be_incremented = np.random.choice(2, size=1)[0]
            # increment horizontal node
            if select_which_direction_to_be_incremented == 0:
                increment_h = np.random.choice(possible_choides_h, size=1)[0]
            # increment vertical node
            else:
                increment_v = np.random.choice(possible_choices_v, size=1)[0]

        next_node = str(int(v) + increment_v) + str(int(h) + increment_h)
        edge = "from" + initial_node + "to" + next_node
        edge_position = find_a_position_to_insert_new_vehicle_into_the_selected_edge(edge, start_positions, lane_length)
        if not edge_position is None:
            found = True
    position = (edge, edge_position)
    start_positions.append(position)


def find_a_position_to_insert_new_vehicle_into_the_selected_edge(edge, start_positions, lane_length):
    edge_position = 0
    if (edge, edge_position) in start_positions:
        while edge_position < lane_length:
            if (edge, edge_position) not in start_positions:
                return edge_position
            else:
                edge_position += 10
        return None
    else:
        return edge_position

def split_string_node_id(edge):
    edge = edge.replace("from", "")
    splited_string = edge.split("to")
    return splited_string[0], splited_string[1]

def read_edges_cost_input_file(file):
    count = 0
    with open(file) as f:
        for line in f.readlines():
            if count == 0 and not line[0] == '#':
                count += 1
                line = line.replace('\n','')
                if not line == 'edges_toll_cost':
                    raise 
            elif count == 1 and not line[0] == '#':
                count += 1
                splited_line = line.split(',')
                horizontal, vertical = int(splited_line[0]), int(splited_line[1])
                if not horizontal == NUMBER_OF_HORIZONTAL_NODES or not vertical == NUMBER_OF_VERTICAL_NODES:
                    raise
            elif count > 1 and not line[0] == '#':
                count += 1
                splited_line = line.split(',')
                e, c = splited_line[0], float(splited_line[1])
                edges_toll_cost[e] = c

def get_cost_from_input_file(edge):
    return edges_toll_cost[edge]

def make_edge_string(src_y:int, src_x:int, dst_y:int, dst_x:int):
    return 'from' + str(src_y) + str(src_x) + 'to' + str(dst_y) + str(dst_x) 

def read_weight_input(net_params):
    weight_path = net_params.additional_params['weight_path']
    df = pd.read_csv(weight_path)
    for index, row in df.iterrows():
        network_vehicles_data[row['veh_id']].weight(row['time_weight'], row['toll_weight'])
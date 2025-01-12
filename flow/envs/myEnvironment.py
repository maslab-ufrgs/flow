# import the base environment class
import csv
import os
import string
import sys
from flow.envs import Env
from flow.networks.MyGrid import *
from gym.spaces.box import Box
from gym.spaces import Tuple
from flow.utils.dijkstra import *
import numpy as np

FIRST_EXECUTION = -1

class Vehicle():
    def __init__(self, id, destiny=None, timew=0.5, tollw=0.5, experience={}) -> None:
        self.id = id
        self.destiny = destiny  # string
        self.timew = timew      # float
        self.tollw = tollw      # float
        # dict of floats
        # experiencedTime[edge:string] = time:float, s.t.,
        # it is the time perceived by the vehicle over the edge
        self.experiencedTime = experience
    
    def __str__(self) -> str:
        return "Vehicle [ {} ] = (dst: {}, timew: {}, tollw: {}, expertience: {})".format(
            self.id,
            self.destiny,
            self.timew,
            self.tollw,
            self.experiencedTime
        )

    def get_destiny(self):
        return self.destiny
    
    def get_timew(self):
        return self.timew
        
    def get_tollw(self):
        return self.tollw

    def getExperiencedTime(self, edge):
        if edge in self.experiencedTime:
            return self.experiencedTime[edge]
        else:
            return None
    
    def set_free_flow_time(self, edge, free_flow):
        self.experiencedTime[edge] = free_flow

    def updateTimeExperience(self, edge, time):
        stepsize = 0.1
        if edge in self.experiencedTime:
            # NewEstimate <- OldEstimate + StepSize*(Target - OldEstimate)
            self.experiencedTime[edge] = self.experiencedTime[edge] + stepsize * (time - self.experiencedTime[edge])
        else:
            quit('ERROR -- UPDATE EXPERIENCE -- NO PREVIOUS EDGE TO UPDATE')


    def from_str(self, individual_id:str):
        if self.id in network_vehicles_data:
            self.destiny = network_vehicles_data[self.id].get_destiny()
            self.timew = network_vehicles_data[self.id].get_timew(individual_id)
            self.tollw = network_vehicles_data[self.id].get_tollw(individual_id)
        else:
            self.random_destiny()
            self.timew = 0.5
            self.tollw = 0.5

    def random_destiny(self):
        h = np.random.choice(NUMBER_OF_HORIZONTAL_NODES, size=1)[0]
        v = np.random.choice(NUMBER_OF_VERTICAL_NODES, size=1)[0]
        self.destiny = str(v) + str(h)

class Edge():
    def __init__(self, id, num_vehicles=[]) -> None:
        self.id = id
        # list<int>
        # num_vehicles[exec:int] = number of vehicles at execution exec
        self.num_vehicles = num_vehicles
    
    def __str__(self) -> str:
        return 'Edge [ {} ] = (num_vehicles: {})'.format(
            self.id,
            self.num_vehicles,
        )

    def updateNumberOfVehicles(self, numVehicles:int):
        self.num_vehicles.append(numVehicles)

class RouteSegment():
    def __init__(self, veh_id, edge_id, time_spent=0, cost_spent=0) -> None:
        self.veh_id = veh_id
        self.edge_id = edge_id
        self.time_spent = time_spent
        self.cost_spent = cost_spent

    def __str__(self) -> str:
        return "'veh':'{}',\n'edge':'{}',\n'time':{},\n'cost':{},\n".format(self.veh_id, self.edge_id, self.time_spent, self.cost_spent)

    def update_time(self, time_spent):
        self.time_spent = self.time_spent + time_spent
    
    def update_cost(self, cost_spent):
        if cost_spent > 0 and self.cost_spent == 0: 
            self.cost_spent = deepcopy(cost_spent)

    def update(self, time_spent, cost_spent):
        self.update_time(time_spent)
        self.update_cost(cost_spent)

    def equal(self, veh_id, edge_id):
        if self.veh_id == veh_id and self.edge_id == edge_id:
            return True
        else:
            return False
        
    def isVehicle(self, veh_id):
        if self.veh_id == veh_id:
            return True
        else:
            return False

class DataBase():
    def __init__(self, execution=FIRST_EXECUTION, vehicles={}, edges={}, route_segments=[]):
        # current execution (int)
        self.execution = execution
        # dict<vehicle_id:string> = Vehicle
        self.vehicles = vehicles
        # dict<edge_id:string> = Edge
        self.edges = edges
        # list<list<RouteSegment>>
        # list[exec:int] = list<RouteSegment>
        self.route_segments = route_segments
        # dict<vehicle_id:string> = reseted:bool
        self.reseted = {}

    def from_str(self, individual_id: str, vehicles:list, edges:list, freeflow:dict):
        for vstr in vehicles:
            v = Vehicle(vstr)
            v.from_str(individual_id)
            self.vehicles[vstr] = deepcopy(v)
            self.reseted[vstr] = True

        for estr in edges:
            e = Edge(estr)
            self.edges[estr] = deepcopy(e)

        for vkey in self.vehicles:
            for ekey in self.edges:
                self.vehicles[vkey].set_free_flow_time(ekey, freeflow[ekey])

    def add_route_segment(self, route_segment):
        self.route_segments[-1].append(route_segment)

    def new_execution(self):
        if self.execution >= 0:
            self.updateVehiclesTimeExperiences()
        for vkey in self.reseted:
            self.reseted[vkey] = True
        self.route_segments.append([])
        self.execution += 1

    def updateEdgesNumberOfVehicles(self):
        for ekey in self.edges:
            numVehicles = 0
            for vkey in self.vehicles:
                numVehicles += len([rs for rs in self.route_segments[-1] if rs.equal(vkey, ekey)])
            self.edges[ekey].updateNumberOfVehicles(numVehicles)

    # to be checked
    def updateVehiclesTimeExperiences(self):
        for vkey in self.vehicles:
            for ekey in self.edges:
                time = sum([rs.time_spent for rs in self.route_segments[-1] if rs.equal(vkey, ekey)])
                if time > 0:
                    self.vehicles[vkey].updateTimeExperience(ekey, time)
    
    def to_csv(self, emission_path='/home/macsilva/Desktop/maslab/flow/data/vehicles.csv'):
        veh_header = ['veh_id','run','edge','time','cost','destiny','timew','tollw']
        with open(emission_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(veh_header)
            for exec in range(self.execution+1):
                for vkey in self.vehicles:
                    route_segments = [rs for rs in self.route_segments[exec] if rs.isVehicle(vkey)]
                    if route_segments:
                        for rs in route_segments:
                            writer.writerow([
                                vkey, 
                                exec, 
                                rs.edge_id, 
                                rs.time_spent, 
                                rs.cost_spent, 
                                self.get_destiny(vkey), 
                                self.vehicles[vkey].get_timew(),
                                self.vehicles[vkey].get_tollw()])            

    def get_destiny(self, veh_id):
        return self.vehicles[veh_id].get_destiny()

    def get_tollw(self, veh_id):
        return self.vehicles[veh_id].get_tollw()

    def get_timew(self, veh_id):
        return self.vehicles[veh_id].get_timew()

    def ableToApplyDijkstra(self, veh_id):
        return self.reseted[veh_id]
    
    def applyDijkstra(self, veh_id):
        self.reseted[veh_id] = False

    def getExperiencedTime(self, veh_id, edge):
        return self.vehicles[veh_id].getExperiencedTime(edge)

    def set_free_flow_time(self, veh_id, edge, free_flow):
        self.vehicles[veh_id].set_free_flow_time(edge, free_flow)

    def update_route(self, veh_id, edge_id, timespent, costspent):
        route_segment = [rs for rs in self.route_segments[-1] if rs.equal(veh_id, edge_id)]
        if route_segment:
            route_segment[0].update(timespent, costspent)
        else:
            self.add_route_segment(deepcopy(RouteSegment(veh_id, edge_id, timespent, costspent)))

    def terminate(self, emission_path=None):
        if not emission_path is None:
            emission_path = emission_path + '/vehicles.csv'
        self.to_csv(emission_path)

    
    def print(self):
        print('#Executions = {}'.format(self.execution))
        print('\nVehicles:')
        for v in self.vehicles:
            print(self.vehicles[v])
        print('\nEdges:')
        for e in self.edges:
            print(self.edges[e])
        print('\nRoute Segments:')
        for rs_list in self.route_segments:
            print('\n--------')
            for rs in rs_list:
                print(rs)

class myEnvironment(Env):
    def __init__(self, env_params, sim_params, network=None, simulator='traci', scenario=None):
        individual_id = env_params.additional_params["individual_id"]
        self.databases = {}
        self.databases[individual_id] = DataBase()
        self.execution = FIRST_EXECUTION    
        super().__init__(env_params, sim_params, network, simulator, scenario)
        self.freeflow = {}
        self.costs = {}
        self.calculate_freeflow()
        self.calculate_costs(env_params)
        self.databases[individual_id].from_str(
            self.env_params.additional_params["individual_id"],
            self.initial_ids, 
            self.k.network.get_edge_list(), 
            self.freeflow)
        self.network_system_optimal(env_params)

    @property
    def action_space(self):
        num_actions = self.initial_vehicles.num_rl_vehicles
        accel_ub = self.env_params.additional_params["max_accel"]
        accel_lb = - abs(self.env_params.additional_params["max_decel"])

        return Box(low=accel_lb,
                   high=accel_ub,
                   shape=(num_actions,))

    @property
    def observation_space(self):
        return Box(
            low=0,
            high=float("inf"),
            shape=(2*self.initial_vehicles.num_vehicles,),
        )

    def _apply_rl_actions(self, rl_actions):
        rl_ids = self.k.vehicle.get_rl_ids()
        self.k.vehicle.apply_acceleration(rl_ids, rl_actions)

    def get_state(self, **kwargs):
        ids = self.k.vehicle.get_ids()
        pos = [self.k.vehicle.get_x_by_id(veh_id) for veh_id in ids]
        vel = [self.k.vehicle.get_speed(veh_id) for veh_id in ids]
        return np.concatenate((pos, vel))

    def compute_reward(self, rl_actions, **kwargs):
        ids = self.k.vehicle.get_ids()
        speeds = self.k.vehicle.get_speed(ids)
        return np.mean(speeds)
    
    def applyDijkstra(self, veh_id):
        self.databases[self.env_params.additional_params['individual_id']].applyDijkstra(veh_id)

    def reset(self):    
        if self.execution == FIRST_EXECUTION:
            edges_to_csv(self.k.network._edge_list, self.env_params.additional_params['edges_path'])
            junctions_to_csv(self.k.network._junction_list, self.env_params.additional_params['junctions_path'])

        individual_id = self.env_params.additional_params['individual_id']
        self.databases[individual_id].new_execution()
        self.execution += 1

        observation = super().reset()
        return observation
            
    def terminate(self):
        # TODO: descobrir pq este metodo eh chamado 2x
        emission_path = self.env_params.additional_params['emission_path']
        individual_id = self.env_params.additional_params['individual_id']
        self.databases[individual_id].terminate(emission_path)
        super().terminate()

    def get_destiny(self, veh_id):
        return self.databases[self.env_params.additional_params['individual_id']].get_destiny(veh_id)

    def get_timew(self, veh_id):
        return self.databases[self.env_params.additional_params['individual_id']].get_timew(veh_id)

    def get_tollw(self, veh_id):
        return self.databases[self.env_params.additional_params['individual_id']].get_tollw(veh_id)
    
    def ableToApplyDijkstra(self, veh_id):
        return self.databases[self.env_params.additional_params['individual_id']].ableToApplyDijkstra(veh_id)

    def getExperiencedTime(self, veh_id, edge):
        time = self.databases[self.env_params.additional_params['individual_id']].getExperiencedTime(veh_id, edge)
        if time == None:
            ff = self.freeflow[edge]
            self.databases[self.env_params.additional_params['individual_id']].set_free_flow_time(veh_id, edge, ff)
            return ff
        else:
            return time

    def get_toll_cost(self, edge):
        return self.costs[edge]

    def calculate_costs(self, env_params, option=1):
        # TODO: alterar esse metodo para que o usuario escolha, via cmd line, qual a preferencia dele por pedagio:
        # opcoes:
        #   1. gerado via input file
        #   2. gerado aleatoriamente
        #   3. pegadio que se atualiza conforme os carros vao passando por cima das ruas
        if option == 1:
            edges = self.k.network.get_edge_list()
            junctions = self.k.network.get_junction_list()
            junctions_cost = 0
            input_file = env_params.additional_params["costs_path"]
            read_edges_cost_input_file(input_file)
            for edge in edges:
                self.costs[edge] = deepcopy(get_cost_from_input_file(edge))
            for junction in junctions:
                self.costs[junction] = junctions_cost
        else:
            quit()

    def update_vehicle_route(self, veh_id, edge_id):
        self.databases[self.env_params.additional_params['individual_id']].update_route(veh_id, edge_id, self.sim_step, self.costs[edge_id])

    def freeflow_time(self, edge):
        distance = self.k.network.edge_length(edge)
        maxspeed = self.k.network.speed_limit(edge)
        return distance/ maxspeed

    def calculate_freeflow(self):
        edges = self.k.network.get_edge_list()
        for edge in edges:
            self.freeflow[edge] = deepcopy(self.freeflow_time(edge))

    def network_system_optimal(self, env_params):
        BOTTLENECK_CAPACITY = 10

        ### obtain network data ###
        filepath = env_params.additional_params["sonet_path"]
        if not os.path.exists(filepath):
            edges = self.k.network.get_edge_list()
            edges_capacities_fftimes = []
            nodes = []
            minGap = float('inf')
            for vehType in self.k.vehicle.minGap:
                if self.k.vehicle.minGap[vehType] < minGap:
                    minGap = self.k.vehicle.minGap[vehType]

            for edge in edges:
                u, v = split_string_node_id(edge)
                if u not in nodes:
                    nodes.append(u)
                if v not in nodes:
                    nodes.append(v)
                capacity = BOTTLENECK_CAPACITY
                fftime = int(self.freeflow_time(edge))
                edges_capacities_fftimes.append((edge, capacity, fftime))

            ### headers ###
            function_header = "#function name (args) formula\n"
            node_header = "#node name\n"
            edge_header = "#dedge name origin destination function constants\n"
            od_header = "#od name origin destination flow\n"

            ### start building network file ###
            with open(filepath, "w") as tn:
                ### function ###
                tn.write(function_header)
                tn.write("function BPR (f) t * (1 + 0.15 * (f / c) ^ 4)\n")

                ### node ###
                tn.write(node_header)
                for n in nodes:
                    if not 'Artificial' in n:
                        tn.write("node {}\n".format(n))

                ### edge ###
                tn.write(edge_header)
                # ex: dedge UtoV U V BPR t0 cUV, where:
                #   t0 = free flow time
                #   cUV = how many vehicles does UV support
                for uv, Cuv, FFTuv in edges_capacities_fftimes:
                    u, v = split_string_node_id(uv)
                    u = u.replace('from', '')
                    if not 'Artificial' in u:
                        tn.write("dedge {} {} {} BPR {} {}\n".format(
                            u + "-" + v,
                            u,
                            v,
                            FFTuv,
                            Cuv,
                        ))

                ### od ###
                tn.write(od_header)
                ods = {}
                for key in network_vehicles_data:
                    if  type(network_vehicles_data[key]) is InputFileVehicleData:
                        uv = network_vehicles_data[key].get_first_edge()
                        u, v = split_string_node_id(uv)
                        dst = network_vehicles_data[key].get_termination_node()
                        od = v + "|" + dst
                        if od not in ods:
                            ods[od] = 1
                        else:
                            ods[od] = ods[od] + 1
                ods_string = ""
                for od in ods:
                    o, d = od.split("|")
                    ods_string += "od {} {} {} {}\n".format(od, o, d, ods[od])
                tn.write(ods_string[0:-1])
        
def edges_to_csv(edges, edgespath):
    header = ["edge"]
    with open(edgespath, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for e in edges:
            writer.writerow([e])

def junctions_to_csv(junctions, junctionspath):
    header = ["junction"]
    with open(junctionspath, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for j in junctions:
            writer.writerow([j])
                

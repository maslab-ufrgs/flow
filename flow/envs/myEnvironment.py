# import the base environment class
from flow.envs import Env
from flow.networks.MyGrid import *
from gym.spaces.box import Box
from gym.spaces import Tuple
import numpy as np

FIRST_EXECUTION = -1

class Vehicle():
    def __init__(self, id, destiny=None, timew=0.5, tollw=0.5, costs=[], routes=[], paths=[], experience={}) -> None:
        self.id = id
        self.destiny = destiny
        self.timew = timew
        self.tollw = tollw  
        self.costs = costs 
        self.routes = routes
        self.paths = paths
        self.experience = experience

    def get_destiny(self):
        return self.destiny
    
    def get_timew(self):
        return self.timew
        
    def get_tollw(self):
        return self.tollw

    def get_cost(self, exe):
        return self.costs[exe]
    
    def get_costs(self):
        return self.costs

    def get_total_cost(self):
        return sum(self.costs)

    def get_paths(self):
        return self.paths

    def get_path(self, exec):
        return self.paths[exec]

    def add_path(self, path):
        self.paths.append(path)

    def get_expecience_time(self, edge):
        if edge in self.experience:
            return self.experience[edge]
        else:
            return None
    
    def set_free_flow_time(self, edge, free_flow):
        self.experience[edge] = free_flow

    def update_experience(self, edge, time):
        print('before -- id:{}, time:{}'.format(self.id, self.experience[edge]))
        if edge in self.experience:
            self.experience[edge] = 0.9*self.experience[edge] + 0.1*time
        else:
            print('ERROR -- UPDATE EXPERIENCE -- NO PREVIOUS EDGE TO UPDATE')
            quit()
        print('after -- id:{}, time:{}'.format(self.id, self.experience[edge]))

    def from_str(self):
        if self.id in network_vehicles_data:
            self.destiny = network_vehicles_data[self.id].get_destiny()
            self.timew = network_vehicles_data[self.id].get_timew()
            self.tollw = network_vehicles_data[self.id].get_tollw()
        else:
            self.random_destiny()
            self.timew = 0.5
            self.tollw = 0.5

    def random_destiny(self):
        h = np.random.choice(NUMBER_OF_HORIZONTAL_NODES, size=1)[0]
        v = np.random.choice(NUMBER_OF_VERTICAL_NODES, size=1)[0]
        self.destiny = str(v) + str(h)

class Edge():
    def __init__(self, id, num_vehicles=[], times=[], costs=[], routes=[]) -> None:
        self.id = id
        self.num_vehicles = num_vehicles
        self.times = times
        self.costs = costs 
        self.routes = routes

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

    def update(self, time_spent, cost_spent=0):
        self.update_time(time_spent)
        self.update_cost(cost_spent)

    def equal(self, veh_id, edge_id):
        if self.veh_id == veh_id and self.edge_id == edge_id:
            return True
        else:
            return False

class DataBase():
    def __init__(self, execution=FIRST_EXECUTION, vehicles={}, edges={}, route_segments=[]):
        # int -> number of current execution
        self.execution = execution
        # dict of Vehicle's
        # [veh0, veh1, ...]
        self.vehicles = vehicles
        # dict of Edge's
        # [edge0, edge1, ...] 
        self.edges = edges
        # list of lists.
        # list[execution] -> [rs0, rs1, rs2, ...]
        self.route_segments = route_segments

    def from_str(self, vehicles:list, edges:list, freeflow:dict):
        for vstr in vehicles:
            v = Vehicle(vstr)
            v.from_str()
            self.vehicles[vstr] = deepcopy(v)

        for estr in edges:
            e = Edge(estr)
            self.edges[estr] = deepcopy(e)

        for vkey in self.vehicles:
            for ekey in self.edges:
                self.vehicles[vkey].set_free_flow_time(ekey, freeflow[ekey])

    def add_vehicle(self, vehicle):
        self.vehicles.append(vehicle)

    def add_edge(self, edge):
        self.edges.append(edge)
    
    def add_route_segment(self, route_segment):
        self.route_segments[-1].append(route_segment)

    def new_execution(self):
        if self.execution >= 0:
            self.update_experience()
        self.route_segments.append([])
        self.execution += 1

    def update_experience(self):
        for vkey in self.vehicles:
            for ekey in self.edges:
                time = sum([rs.time_spent for rs in self.route_segments[-1] if rs.equal(vkey, ekey)])
                if time > 0:
                    self.vehicles[vkey].update_experience(ekey, time)
    
    def to_csv(self, emission_path='/home/macsilva/Desktop/maslab/flow/data/vehicles.csv'):
        veh_header = ['veh_id','run','edge','time','cost','destiny','timew','tollw']
        with open(emission_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(veh_header)
            for exec in range(self.execution+1):
                for vkey in self.vehicles:
                    edges = self.vehicles[vkey].get_path(exec)
                    for edge in edges:
                        route_segments = [rs for rs in self.route_segments[exec] if rs.equal(vkey, edge)]
                        writer.writerow([
                            vkey, 
                            exec, 
                            edge, 
                            route_segments[0].time_spent, 
                            route_segments[0].cost_spent, 
                            self.get_destiny(vkey), 
                            self.vehicles[vkey].get_timew(),
                            self.vehicles[vkey].get_tollw()])             


    def get_destiny(self, veh_id):
        return self.vehicles[veh_id].get_destiny()

    def get_tollw(self, veh_id):
        return self.vehicles[veh_id].get_tollw()

    def get_timew(self, veh_id):
        return self.vehicles[veh_id].get_timew()

    def could_add_path(self, veh_id):
        # a vehicle should get a new path per execution
        paths = self.vehicles[veh_id].get_paths()
        if len(paths) < self.execution + 1:
            return True
        else:
            return False

    def add_path(self, veh_id, path):
        self.vehicles[veh_id].add_path(path)

    def get_expecience_time(self, veh_id, edge):
        return self.vehicles[veh_id].get_expecience_time(edge)

    def set_free_flow_time(self, veh_id, edge, free_flow):
        self.vehicles[veh_id].set_free_flow_time(edge, free_flow)

    def update_route(self, veh_id, edge_id, timespent, costspent=0):
        route_segment = [rs for rs in self.route_segments[-1] if rs.equal(veh_id, edge_id)]
        if route_segment:
            route_segment[0].update(timespent, costspent)
        else:
            self.add_route_segment(deepcopy(RouteSegment(veh_id, edge_id, timespent, costspent)))

    def terminate(self, emission_path=None):
        if emission_path is not None:
            emission_path = emission_path + '/vehicles.csv'
        self.to_csv(emission_path)

class myEnvironment(Env):
    def __init__(self, env_params, sim_params, network=None, simulator='traci', scenario=None):
        self.database = DataBase()
        self.execution = FIRST_EXECUTION      
        super().__init__(env_params, sim_params, network, simulator, scenario)
        self.freeflow = {}
        self.costs = {}
        self.calculate_freeflow()
        self.calculate_costs(env_params)
        self.database.from_str(self.initial_ids, self.k.network.get_edge_list(), self.freeflow)

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

    def reset(self):    
        if self.execution == FIRST_EXECUTION:
            
            edges_to_csv(self.k.network._edge_list, self.env_params.additional_params['edges_path'])
            junctions_to_csv(self.k.network._junction_list, self.env_params.additional_params['junctions_path'])

        self.database.new_execution()
        self.execution += 1

        observation = super().reset()
        return observation
            
    def terminate(self):
        # TODO: descobrir pq este metodo eh chamado 2x
        emission_path = self.sim_params.emission_path
        self.database.terminate(emission_path)
        super().terminate()

    def get_destiny(self, veh_id):
        return self.database.get_destiny(veh_id)

    def get_timew(self, veh_id):
        return self.database.get_timew(veh_id)

    def get_tollw(self, veh_id):
        return self.database.get_tollw(veh_id)

    def could_add_path(self, veh_id):
        return self.database.could_add_path(veh_id)

    def add_path(self, veh_id, path):
        self.database.add_path(veh_id, path)

    def get_expecience_time(self, veh_id, edge):
        time = self.database.get_expecience_time(veh_id, edge)
        if time == None:
            ff = self.freeflow[edge]
            self.database.set_free_flow_time(veh_id, edge, ff)
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
                self.costs[junction] = deepcopy(junctions_cost)
        else:
            quit()

    def update_vehicle_route(self, veh_id, edge_id):
        self.database.update_route(veh_id, edge_id, self.sim_step, self.costs[edge_id])

    def freeflow_time(self, edge):
        distance = self.k.network.edge_length(edge)
        maxspeed = self.k.network.speed_limit(edge)
        return distance/ maxspeed

    def calculate_freeflow(self):
        edges = self.k.network.get_edge_list()
        for edge in edges:
            self.freeflow[edge] = deepcopy(self.freeflow_time(edge))

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


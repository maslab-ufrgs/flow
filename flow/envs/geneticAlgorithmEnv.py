from flow.envs.base import *
import os
import csv
import pandas as pd


class GeneticAlgorithmEnv(Env):
    def __init__(self, env_params, sim_params, network=None, simulator='traci', scenario=None) -> None:
        super().__init__(env_params, sim_params, network, simulator, scenario)
        # to be more memory efficient, a new database is created for each individual, 
        # but, because the Env class is a singleton,
        # the Env stays the same for all the individuals
        if self.env_params.additional_params["generateFreeFlowTime"]:
            self.freeFlowTime()
    
    def freeFlowTime(self):
        freeFlowPath = self.env_params.additional_params["freeflow_path"]
        weightsPath = self.env_params.additional_params["weights_path"]
        freeFlowTimeDict = {}
        edges = self.k.network.get_edge_list()
        print("edges = {}".format(edges))
        for edge in edges:
            if edge not in self.k.network.get_junction_list():
                distance = self.k.network.edge_length(edge)
                maxSpeed = self.k.network.speed_limit(edge)
                freeFlowTimeDict[edge] = distance/ maxSpeed
        df = pd.read_csv(weightsPath)
        vehicles = list(df['veh_id'])
        print("vehicles = {}".format(vehicles))
        with open(freeFlowPath, "w") as file:
            writer = csv.writer(file)
            header = ["veh_id"] + edges
            writer.writerow(header)
            for veh_id in vehicles:
                data = [veh_id]
                for edge in edges:
                    data.append(freeFlowTimeDict[edge])
                writer.writerow(data)
        exit("Free flow file generated... Exiting this run")
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
    ##############################################################################

class OrtuzarEnv(GeneticAlgorithmEnv):
    def __init__(self, env_params, sim_params, network=None, simulator='traci', scenario=None):
        super().__init__(env_params, sim_params, network, simulator, scenario)
    
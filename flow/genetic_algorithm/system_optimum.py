from copy import deepcopy
from flow.controllers import *
from flow.core import experiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import *
from flow.envs import *
from flow.genetic_algorithm.run import get_value
from flow.networks import *
import pandas as pd
import os
import json
import sys
from flow.genetic_algorithm.parser import *


args = parse_args(sys.argv[1:])
if args.host == 'lab':
    so_path = '/home/lab204/Desktop/marco/maslab/flow/data/{}/data.txt'.format(args.exp_tag)
    individual_id = '/home/lab204/Desktop/marco/maslab/flow/data/{}/'.format(args.exp_tag)
elif args.host == 'home':
    individual_id = '/home/macsilva/Desktop/maslab/flow/data/{}/'.format(args.exp_tag)
    so_path = '/home/macsilva/Desktop/maslab/flow/data/{}/data.txt'.format(args.exp_tag)
else: 
    raise
with open(so_path, 'r') as file:
    data = json.loads(file.read())

vehicles = VehicleParams()
additional_router_params = {
    'routes_path'   :   "../../flow/inputs/routes/system_optimum_routes.txt",
}
vehicles.add(
            veh_id="human",
            acceleration_controller=(IDMController, {}),
            routing_controller=(
                MyGridRouterUsingPredefinedRoutes, 
                additional_router_params),
            num_vehicles=data['num_vehicles'],
            )

sim_params = SumoParams(sim_step=data['sim_step'], emission_path=data['emission_path'], render=True)
initial_config = InitialConfig(bunching=data['bunching'], spacing="custom")
additional_env_params ={
    'emission_path' :       data['emission_path'],
    'weight_path'   :       data['weight_path'],
    'costs_path'    :       data['costs_path'],
    'edges_path'    :       data['edges_path'],
    'junctions_path':       data['junctions_path'],
    'sonet_path'    :       data['sonet_path'],
    'individual_id' :       individual_id,
}
env_params=EnvParams(
        horizon=data['horizon'],
        additional_params=additional_env_params
    )
additional_net_params={
    "num_lanes":        data["num_lanes"],
    "speed_limit":      data['speed_limit'],
    "lane_length":      data["lane_length"],
    "use_input_file_to_get_starting_positions": True,
    # obtido atraves do script de system optimum
    "input_file_path": "../../flow/inputs/networks/system_optimum_starting_positions.txt",
    "weight_path":      data['weight_path'],
    "individual_id":    individual_id,
}
net_params = NetParams(additional_params=additional_net_params)
flow_params = dict(
    exp_tag='geneticalgorithm',
    env_name=myEnvironment,
    network=MyGrid,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)
exp = experiment.Experiment(flow_params)
rldata = exp.run(
    num_runs=1, 
    convert_to_csv=False)
print('\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
print('emission path: {}'.format(data['emission_path']))
print('best value: {}'.format(get_value(data['emission_path'], 1)))
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
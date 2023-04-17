from copy import deepcopy
from flow.controllers import *
from flow.core import experiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.envs import *
from flow.networks import *
import pandas as pd
import os

def run(num_vehicles, individual_id,  exp_tag, weight_path, individual_subfolder='0', host='lab',num_runs=1, horizon=3000, sim_step=0.1, bunching=40, save_data=True):
    vehicles = VehicleParams()
    vehicles.add(
                veh_id="human",
                acceleration_controller=(IDMController, {}),
                routing_controller=(MyGridRouterOnlyWhenVehiclesAreReseting, {}),
                num_vehicles=num_vehicles,
                )
    if host == 'lab':
        parentdir= '/home/lab204/Desktop/marco/maslab/flow/data'
        costs_path= '/home/lab204/Desktop/marco/maslab/flow/flow/inputs/networks/costs.txt'
        edges_path= '/home/lab204/Desktop/marco/maslab/flow/data/{}/edges.csv'.format(exp_tag)
        junctions_path= '/home/lab204/Desktop/marco/maslab/flow/data/{}/junctions.csv'.format(exp_tag)
        systemoptimal_path = '/home/lab204/Desktop/marco/maslab/flow/so_net.txt'
    elif host == 'home':
        parentdir= '/home/macsilva/Desktop/maslab/flow/data'
        costs_path= '/home/macsilva/Desktop/maslab/flow/flow/inputs/networks/costs.txt'
        edges_path= '/home/macsilva/Desktop/maslab/flow/data/{}/edges.csv'.format(exp_tag)
        junctions_path= '/home/macsilva/Desktop/maslab/flow/data/{}/junctions.csv'.format(exp_tag)
        systemoptimal_path = '/home/macsilva/Desktop/maslab/flow/so_net.txt'
    else:
        quit('error -- run -- invalid host')
    emission_path = create_dir(individual_id, individual_subfolder, exp_tag, parentdir=parentdir)
    sim_params = SumoParams(sim_step=sim_step, emission_path=emission_path, render=False)
    initial_config = InitialConfig(bunching=bunching, spacing="custom")
    additional_env_params ={
        'individual_id' :   individual_id,
        'emission_path' :   emission_path,
        'weight_path'   :   weight_path,
        'costs_path'    :   costs_path,
        'edges_path'    :   edges_path,
        'junctions_path':   junctions_path,
        'systemoptimal_path': systemoptimal_path,
    }
    env_params=EnvParams(
            horizon=horizon,
            additional_params=additional_env_params
        )
    additional_net_params={
        "num_lanes": 1,
        "speed_limit": 10,
        "lane_length": 100,
        "use_input_file_to_get_starting_positions": False,
        "input_file_path": "flow/inputs/networks/5by5test.txt",
        "weight_path":weight_path,
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
    rldata = exp.run(num_runs, convert_to_csv=save_data)
    return get_vehicle_value(emission_path)


def create_dir(individual, subfolder, exp_tag, parentdir='/home/lab204/Desktop/marco/maslab/flow/data'):
    # '/home/macsilva/Desktop/maslab/flow/data/<exp_tag>'
    path = os.path.join(parentdir, exp_tag)
    if not exp_tag in os.listdir(parentdir):
        os.mkdir(path)
    ind = 'individual_' + str(individual)
    if not ind in os.listdir(path):
        os.mkdir(path + '/' + ind)
    # '/home/macsilva/Desktop/maslab/flow/data/<exp_tag>/individual_<individual>/'
    path = os.path.join(path, ind)
    # '/home/macsilva/Desktop/maslab/flow/data/<exp_tag>/individual_<individual>/<subfolder>'
    path = os.path.join(path, subfolder)
    os.mkdir(path)
    return path

def get_vehicle_value(emission_path):
    vehicles_path = emission_path + '/vehicles.csv'
    df = pd.read_csv(vehicles_path)
    return df.groupby(['run','veh_id']).mean().mean()['time']

    
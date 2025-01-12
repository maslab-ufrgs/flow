from copy import deepcopy
import json
from flow.controllers import *
from flow.core import experiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import *
from flow.envs import *
from flow.networks import *
import pandas as pd
import os
NUM_LANES = 1
SPEED_LIMIT = 10
LANE_LENGTH = 100

def run(num_vehicles, individual_id,  exp_tag, weight_path, host='lab',num_runs=1, horizon=3000, sim_step=0.1, bunching=40, save_data=False):
    vehicles = VehicleParams()
    vehicles.add(
                veh_id="human",
                acceleration_controller=(IDMController, {}),
                routing_controller=(MyGridRouterOnlyWhenVehiclesAreReseting, {}),
                num_vehicles=num_vehicles,
                )
    if host == 'lab':
        parentdir= '/home/lab204/Desktop/marco/maslab/flow/data'
        costs_path= '/home/lab204/Desktop/marco/maslab/flow/flow/inputs/costs/edgesCost.txt'
        edges_path= '/home/lab204/Desktop/marco/maslab/flow/data/{}/edges.csv'.format(exp_tag)
        junctions_path= '/home/lab204/Desktop/marco/maslab/flow/data/{}/junctions.csv'.format(exp_tag)
        sonet_path = '/home/lab204/Desktop/marco/maslab/flow/data/{}/so_net.txt'.format(exp_tag)
    elif host == 'home':
        parentdir= '/home/macsilva/Desktop/maslab/flow/data'
        costs_path= '/home/macsilva/Desktop/maslab/flow/flow/inputs/costs/edgesCost.txt'
        edges_path= '/home/macsilva/Desktop/maslab/flow/data/{}/edges.csv'.format(exp_tag)
        junctions_path= '/home/macsilva/Desktop/maslab/flow/data/{}/junctions.csv'.format(exp_tag)
        sonet_path = '/home/macsilva/Desktop/maslab/flow/data/{}/so_net.txt'.format(exp_tag)
    else:
        quit('error -- run -- invalid host')
    emission_path = create_dir(exp_tag, parentdir=parentdir)
    # sim_params = SumoParams(sim_step=sim_step, emission_path=emission_path, render=False)
    # sim_params = SumoParams(sim_step=sim_step, emission_path=emission_path, render=False, color_by_speed=True, no_step_log=True)
    sim_params = SumoParams(sim_step=sim_step, render=False, color_by_speed=True, no_step_log=True)
    initial_config = InitialConfig(bunching=bunching, spacing="custom")
    additional_env_params ={
        'individual_id'     :   individual_id,
        'emission_path'     :   emission_path,
        'weight_path'       :   weight_path,
        'costs_path'        :   costs_path,
        'edges_path'        :   edges_path,
        'junctions_path'    :   junctions_path,
        'sonet_path'        :   sonet_path,
    }
    env_params=EnvParams(
            horizon=horizon,
            additional_params=additional_env_params
        )
    additional_net_params={
        "num_lanes"         :   NUM_LANES,
        "speed_limit"       :   SPEED_LIMIT,
        "lane_length"       :   LANE_LENGTH,
        'individual_id'     :   individual_id,
        "weight_path"       :   weight_path,
        "use_input_file_to_get_starting_positions": True,
        "input_file_path": "../../flow/inputs/networks/150vehicles.txt",
    }
    net_params = NetParams( additional_params=additional_net_params)

    flow_params= dict(
        exp_tag=    'geneticalgorithm',
        env_name=   myEnvironment,
        network=    MyGrid,
        simulator=  'traci',
        sim=        sim_params,
        env=        env_params,
        net=        net_params,
        veh=        vehicles,
        initial=    initial_config,
    )

    save_experiment_configurations_into_json(
        emission_path=emission_path,
        num_vehicles=num_vehicles,
        exp_tag=exp_tag,
        weight_path=weight_path,
        parentdir=parentdir,
        costs_path=costs_path,
        edges_path=edges_path,
        junctions_path=junctions_path,
        sonet_path=sonet_path,
        host=host,
        num_runs=num_runs,
        horizon=horizon,
        sim_step=sim_step,
        bunching=bunching,
        save_data=save_data,
        speed_limit = SPEED_LIMIT,
        num_lanes = NUM_LANES,
        lane_length = LANE_LENGTH,
    )

    exp = experiment.Experiment(flow_params)
    rldata = exp.run(num_runs, convert_to_csv=save_data)
    return get_value(emission_path, num_runs)


def create_dir(exp_tag, parentdir='/home/lab204/Desktop/marco/maslab/flow/data'):
    # '/home/macsilva/Desktop/maslab/flow/data/<exp_tag>'
    path = os.path.join(parentdir, exp_tag)
    if not exp_tag in os.listdir(parentdir):
        os.mkdir(path)
    return path

def get_value(emission_path, num_runs):
    vehicles_path = emission_path + '/vehicles.csv'
    df = pd.read_csv(vehicles_path)
    df = df.loc[df['run'] == num_runs - 1]
    time = df.groupby(['veh_id','run']).sum().mean()['time']
    os.remove(vehicles_path)
    return time

def save_experiment_configurations_into_json(
    emission_path,
    num_vehicles,
    exp_tag,
    weight_path,
    parentdir,
    costs_path,
    edges_path,
    junctions_path,
    sonet_path,
    host='lab',
    num_runs=1,
    horizon=3000,
    sim_step=0.1,
    bunching=40,
    save_data=True,
    speed_limit = SPEED_LIMIT,
    num_lanes = NUM_LANES,
    lane_length = LANE_LENGTH,
):
    data = {
        'num_vehicles'      : num_vehicles,
        'exp_tag'           : exp_tag,
        'weight_path'       : weight_path,
        'parentdir'         : parentdir,
        'costs_path'        : costs_path,
        'edges_path'        : edges_path,
        'junctions_path'    : junctions_path,
        'sonet_path'        : sonet_path,
        'host'              : host,
        'num_runs'          : num_runs,
        'horizon'           : horizon,
        'sim_step'          : sim_step,
        'bunching'          : bunching,
        'save_data'         : save_data,
        'speed_limit'       : speed_limit,
        'num_lanes'         : num_lanes,
        'lane_length'       : lane_length,
        'emission_path'     : emission_path,
    }
    with open(emission_path + '/data.txt', 'w') as file:
        json.dump(data, file)
    
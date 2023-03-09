from copy import deepcopy
from flow.controllers import IDMController, MyGridRouterOnlyWhenVehiclesAreReseting as Router
from flow.core import experiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.envs.myEnvironment import myEnvironment
from flow.networks.MyGrid import MyGrid, ADDITIONAL_NET_PARAMS
import pandas as pd
import os

def run(num_vehicles, individual_id,  exp_tag, weight_path, individual_subfolder='0',num_runs=1, horizon=3000, sim_step=0.1, bunching=40, save_data=True):
    vehicles = VehicleParams()
    vehicles.add(
                veh_id="human",
                acceleration_controller=(IDMController, {}),
                routing_controller=(Router, {}),
                num_vehicles=num_vehicles,
                )
    emission_path = create_dir(individual_id, individual_subfolder, exp_tag)
    sim_params = SumoParams(sim_step=sim_step, emission_path=emission_path, render=True)
    initial_config = InitialConfig(bunching=bunching, spacing="custom")
    additional_env_params ={
        'individual_id' :   individual_id,
        'emission_path' :   emission_path,
        'weight_path'   :   weight_path,
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


def create_dir(individual, subfolder, exp_tag, parentdir='/home/macsilva/Desktop/maslab/flow/data'):
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

    
# the TestEnv environment is used to simply simulate the network
import os
from flow.envs import TestEnv

# the Experiment class is used for running simulations
from flow.core.experiment import Experiment

# the base network class
from flow.networks import Network

# all other imports are standard
from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import EnvParams
from flow.core.params import SumoParams 

# specify the routes for vehicles in the network
class TemplateNetwork(Network):

    def specify_routes(self, net_params):
        return {"-32410#3": ["-32410#3"]}

# create some default parameters parameters
env_params = EnvParams()
initial_config = InitialConfig()
vehicles = VehicleParams()
vehicles.add('human', num_vehicles=1)

lust_path = "/home/macsilva/Desktop/maslab/flow/flow/genetic_algorithm/lust/LuSTScenario"

sim_params = SumoParams(render=True, sim_step=1)

# specify the edges vehicles can originate on
initial_config = InitialConfig(
    edges_distribution=["-32410#3"]
)

net_params = NetParams(
    template={
        # network geometry features
        "net": os.path.join(lust_path, "scenario/lust.net.xml"),
        # features associated with the properties of drivers
        "vtype": os.path.join(lust_path, "scenario/vtypes.add.xml"),
        # features associated with the routes vehicles take
        "rou": [os.path.join(lust_path, "scenario/DUARoutes/local.0.rou.xml"),
                os.path.join(lust_path, "scenario/DUARoutes/local.1.rou.xml"),
                os.path.join(lust_path, "scenario/DUARoutes/local.2.rou.xml")]
    }
)

# we no longer need to specify anything in VehicleParams
new_vehicles = VehicleParams()
    

flow_params = dict(
    exp_tag='template',
    env_name=TestEnv,
    network=Network,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=new_vehicles,
    initial=initial_config,
)

# number of time steps
flow_params['env'].horizon = 100000
exp = Experiment(flow_params)

# run the sumo simulation
_ = exp.run(1)
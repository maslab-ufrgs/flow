from flow.controllers import IDMController, MyGridRouterOnlyWhenVehiclesAreReseting as Router
from flow.core import experiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.envs.myEnvironment import myEnvironment
from flow.networks.MyGrid import MyGrid as Network, ADDITIONAL_NET_PARAMS


vehicles = VehicleParams()
vehicles.add(
            veh_id="human",
            acceleration_controller=(IDMController, {}),
            routing_controller=(Router, {}),
            num_vehicles=100,
            )

sim_params = SumoParams(sim_step=0.1)
initial_config = InitialConfig(bunching=40, spacing="custom")
env_params=EnvParams(
        horizon=3000,
    )
net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS.copy())

flow_params = dict(
    exp_tag='selectingroutes',
    env_name=myEnvironment,
    network=Network,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)
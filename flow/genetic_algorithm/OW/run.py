# the TestEnv environment is used to simply simulate the network
import os
from flow.envs.geneticAlgorithmEnv import GeneticAlgorithmEnv

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
from flow.utils.newDijkstra import *

homePrefix = "/home/macsilva/Desktop/maslab/flow/"
labPrefix = "/home/lab204/Desktop/marco/maslab/flow/"
paths = {
    'graphPath' : "flow/data/graphs/OW.net",
    'experiencedTimePath' : "flow/data/vehicles_experiences/experiencedTime.csv",
    'freeFlowPath' : "flow/data/vehicles_experiences/freeFlowTime.csv",
    'costsPath' : "flow/data/costs/costs.csv",
    'weightsPath' : "flow/data/weights/",
    'ow_dir' : "flow/genetic_algorithm/OW/Ortuzar10_1/",
    'routesPath' : "flow/data/routes/dijks.xml",
    'emissionPath' : "flow/data/emission/",
}


class TemplateNetwork(Network):

    def specify_routes(self, net_params):
        return {
            "A1A": ["A1A"],
            "B1B" : ["B1B"],    
            }
def run(individual_id = 0, host='lab',num_runs=1):
    ####################################################
    ## before starting the simulation, set the paths  ##
    ####################################################
    if host == 'lab':
        prefix = labPrefix
    else:
        prefix = homePrefix
    graphPath = prefix + paths['graphPath']
    experiencedTimePath = prefix + paths['experiencedTimePath']
    freeFlowPath = prefix + paths['freeFlowPath']
    costsPath = prefix + paths['costsPath']
    weightsPath = prefix + paths['weightsPath'] + "individual_" + str(individual_id) + ".csv"
    ow_dir = prefix + paths['ow_dir']
    routesPath = prefix + paths['routesPath']
    emissionPath = prefix + paths['emissionPath']
    ####################################################
    for i in range(num_runs):
        print("RUN = {}".format(i))
        emissionDir = os.listdir(emissionPath)
        if i == 0:
            # remove possible trash before starting this simulation
            if os.path.isfile(experiencedTimePath):
                os.remove(experiencedTimePath)
            if len(emissionDir) > 0:
                for f in emissionDir:
                    os.remove(emissionPath + f)
                emissionDir = os.listdir(emissionPath)

        # if there is data from the previous runs, use it.
        if len(emissionDir) > 0:
            emissionFile = emissionPath + emissionDir[0]
        else:
            emissionFile = None

        # create graph that will be used for running dijkstra
        g = Graph(graphPath=graphPath, experiencedTimePath=experiencedTimePath, freeFlowPath=freeFlowPath, costsPath=costsPath, weightsPath=weightsPath, emissionPath=emissionFile)
        # run dijkstra and create xml routes file 
        createRoutesFile(g, routesPath)

        # clear previous simulation trash to get new emissions at the end of this run
        if len(emissionDir) > 0:
            for f in emissionDir:
                os.remove(emissionPath + f)

        env_params = EnvParams(additional_params={
            "freeflow_path":  freeFlowPath,
            "weights_path":weightsPath,
        })

        net_params = NetParams(
            template={
                # network geometry features
                "net": os.path.join(ow_dir, "Network/ortuzar.net.xml"),
                # features associated with the properties of drivers
                "vtype": os.path.join(ow_dir, "Network/vtypes.add.xml"),
                # features associated with the routes vehicles take
                "rou": routesPath,
            }
        )

        flow_params = dict(
            exp_tag='template',
            env_name=GeneticAlgorithmEnv,
            network=Network,
            simulator='traci',
            sim=SumoParams(render=False, sim_step=1),
            env=env_params,
            net=net_params,
            veh=VehicleParams(),
            initial=InitialConfig(edges_distribution="all"),
        )

        # flow_params['env'].horizon = 100000
        flow_params['env'].horizon = 1000
        flow_params['sim'].emission_path = emissionPath
        exp = Experiment(flow_params)

        _ = exp.run(1, convert_to_csv=True)
    emissionDir = os.listdir(emissionPath)
    emissionFile = emissionPath + emissionDir[0]
    df = pd.read_csv(emissionFile)
    time = df.groupby(['id']).count().mean()['time']
    print("individual {} whose value is {}".format(individual_id, time))
    return time

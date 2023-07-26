# the TestEnv environment is used to simply simulate the network
import os
import shutil
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
    'graphPath' : "flow/genetic_algorithm/csv/graphs/OW.net",
    'experiencedTimePath' : "flow/genetic_algorithm/csv/vehicles_experiences/experiencedTime.csv",
    'freeFlowPath' : "flow/genetic_algorithm/csv/vehicles_experiences/freeFlowTime.csv",
    'costsPath' : "flow/genetic_algorithm/csv/costs/costs.csv",
    'weightsPath' : "flow/genetic_algorithm/csv/weights/",
    'ow_dir' : "flow/genetic_algorithm/OW/Ortuzar10_1/",
    'routesPath' : "flow/genetic_algorithm/csv/routes/dijks.xml",
    'emissionPath' : "flow/genetic_algorithm/csv/emission/",
}


class TemplateNetwork(Network):

    def specify_routes(self, net_params):
        return {
            "A1A": ["A1A"],
            "B1B" : ["B1B"],    
            }

def run(individual_id = 0, host='lab',num_runs=1, generateFreeFlowTime=False):
    ####################################################
    ## before starting the simulation, set the paths  ##
    ####################################################
    if host == 'lab':
        prefix = labPrefix
    else:
        prefix = homePrefix
    graphPath = prefix + paths['graphPath']
    experiencedTimePath = prefix + paths['experiencedTimePath'].replace(".csv", "_") + str(individual_id) + ".csv"
    costsPath = prefix + paths['costsPath']
    freeFlowPath = prefix + paths['freeFlowPath']
    weightsPath = prefix + paths['weightsPath'] + "individual_" + str(individual_id) + ".csv"
    ow_dir = prefix + paths['ow_dir']
    routesPath = prefix + paths['routesPath'].replace("dijks", str(individual_id))
    emissionPath = prefix + paths['emissionPath']
    if os.path.exists(emissionPath + str(individual_id) + "/"):
        shutil.rmtree(emissionPath + str(individual_id) + "/")
    os.mkdir(emissionPath + str(individual_id) + "/")
    emissionPath = emissionPath + str(individual_id) + "/"
    ####################################################
    for i in range(num_runs):
        print("RUN = {}, INDIVIDUAL = {}".format(i, individual_id))
        emissionFiles = os.listdir(emissionPath)
        
        # if there is data from the previous runs, use it to generate new routes.
        if len(emissionFiles) > 0:
            emissionFile = emissionPath + emissionFiles[0]
        else:
            emissionFile = None

        # create graph that will be used for running dijkstra
        g = Graph(graphPath=graphPath, experiencedTimePath=experiencedTimePath, freeFlowPath=freeFlowPath, costsPath=costsPath, weightsPath=weightsPath, emissionPath=emissionFile)
        # run dijkstra and create xml routes file 
        createRoutesFile(g, routesPath)

        # After using the data to route generation, clear previous simulation trash to get new emissions at the end of this run.
        if len(emissionFiles) > 0:
            for f in emissionFiles:
                os.remove(emissionPath + f)

        env_params = EnvParams(
            # horizon=100,
            horizon=100000,
            additional_params={
            "freeflow_path":    freeFlowPath,
            "weights_path":     weightsPath,
            "generateFreeFlowTime": generateFreeFlowTime,
        })
        net_params = NetParams(
            template={
                "net": os.path.join(ow_dir, "Network/ortuzar.net.xml"),
                "vtype": os.path.join(ow_dir, "Network/vtypes.add.xml"),
                "rou": routesPath,
            }
        )

        flow_params = dict(
            exp_tag='template',
            env_name=GeneticAlgorithmEnv,
            network=Network,
            simulator='traci',
            sim=SumoParams(render=False, sim_step=1, emission_path=emissionPath, no_step_log=True),
            env=env_params,
            net=net_params,
            veh=VehicleParams(),
            initial=InitialConfig(edges_distribution="all"),
        )

        exp = Experiment(flow_params)
        _ = exp.run(1, convert_to_csv=True)
    
    
    emissionFiles = os.listdir(emissionPath)
    if emissionFiles == None or len(emissionFiles) < 1:
        print(emissionPath)
        exit("LOG: error when generating emissions file for individual {}.".format(individual_id))
    emissionFile = emissionPath + emissionFiles[0]
    df = pd.read_csv(emissionFile)
    time = df.groupby(['id']).count().mean()['time']
    print("LOG: individual {} whose value is {}".format(individual_id, time))
    shutil.rmtree(emissionPath)
    return time

def remove_routes(individual_id:str, host:str):
    if host == 'lab':
        prefix = labPrefix
    else:
        prefix = homePrefix
    routesPath = prefix + paths['routesPath'].replace("dijks", str(individual_id))
    os.remove(routesPath)
    experiencedTimePath = prefix + paths['experiencedTimePath'].replace(".csv", "_") + str(individual_id) + ".csv"
    os.remove(experiencedTimePath)
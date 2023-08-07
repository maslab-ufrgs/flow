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

paths = {
    'graphPath' : "/csv/graphs/OW.net",
    'experiencedTimePath' : "/csv/vehicles_experiences/",
    'costsPath' : "/csv/costs/costs.csv",
    'weightsPath' : "/csv/weights/",
    'routesPath' :  "/csv/routes/",
    'emissionPath' : "/csv/emission/",
}

def run(prefix, networkPath, vTypePath, freeFlowPath, individual_id = 0,num_runs=1, generateFreeFlowTime=False):
    ####################################################
    ## before starting the simulation, set the paths  ##
    ####################################################
    individual_str = "individual_{}".format(individual_id)
    graphPath = prefix + paths['graphPath']
    experiencedTimePath = prefix + paths['experiencedTimePath'] + individual_str + ".csv"
    costsPath = prefix + paths['costsPath']
    weightsPath = prefix + paths['weightsPath'] + individual_str + ".csv"
    routesPath = prefix + paths['routesPath'] + individual_str + ".xml"
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
                "net":      networkPath,
                "vtype":    vTypePath,
                "rou":      routesPath,
            }
        )
        
        sim_params = SumoParams(
            port=individual_id,
            render=False, 
            sim_step=1, 
            emission_path=emissionPath, 
            no_step_log=True)

        flow_params = dict(
            exp_tag='template',
            env_name=GeneticAlgorithmEnv,
            network=Network,
            simulator='traci',
            sim=sim_params,
            env=env_params,
            net=net_params,
            veh=VehicleParams(),
            initial=InitialConfig(edges_distribution="all"),
        )
        print("LOG: starting simulation #{} for individual #{}".format(i, individual_id))
        exp = Experiment(flow_params)
        _ = exp.run(1, convert_to_csv=True)
        print("LOG: ending simulation #{} for individual #{}".format(i, individual_id))
    
    
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

def remove_routes(prefix, individual_id:str):
    routesPath = prefix + paths['routesPath'].replace("dijks", str(individual_id))
    os.remove(routesPath)
    experiencedTimePath = prefix + paths['experiencedTimePath'].replace(".csv", "_") + str(individual_id) + ".csv"
    os.remove(experiencedTimePath)
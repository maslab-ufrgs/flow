import os
import pandas as pd
import csv

class Vehicle:
    def __init__(self, id:str, tollw:float, timew:float, experiencedTime:dict={}) -> None:
        self.id = id
        self.tollw = tollw
        self.timew = timew
        # dict[edgeId:str] -> float
        self.experiencedTime = experiencedTime
    
    def setExperiencedTime(self, experiencedTime:dict):
        self.experiencedTime = experiencedTime
    
    def newExperience(self, edge:str, timeSpentOverEdge:float):
        # Est <- (1 - alpha) * Est + alpha * newEst, with alpha = 0.1
        print("previous experience = [{}], veh = [{}], edge = [{}], time spent = [{}]".format(
            self.experiencedTime[edge],
            self.id,
            edge,
            timeSpentOverEdge
        ))
        self.experiencedTime[edge] = 0.9*self.experiencedTime[edge] + 0.1*timeSpentOverEdge
        print("--> next experience = [{}]".format(self.experiencedTime[edge]))
    
    def __str__(self) -> str:
        return "Vehicle[{}] = [timew={}, tollw={}]".format(self.id, self.timew, self.tollw)

class Edge:
    def __init__(self, id, cost) -> None:
        self.id = id
        self.cost = cost
    
    def __str__(self) -> str:
        return "Edge[{}] = [cost={},]".format(self.id, self.cost)

class RouteSegment:
    def __init__(self, edgeId:str, timeSpent:float, costSpent:float) -> None:
        self.edgeId = edgeId
        self.timeSpent = timeSpent
        self.costSpent = costSpent
    
    def update(self, timeSpent):
        self.timeSpent += timeSpent


    def __str__(self) -> str:
        return "RouteSegment = [edgeId={}, timeSpent={}, costSpent={},]".format(self.edgeId,self.timeSpent, self.costSpent)

class Route:
    def __init__(self, vehId, routeSegments = []) -> None:
        self.vehId = vehId
        self.routeSegments = routeSegments
    
    def update(self, edgeId, timeSpent, costSpent):
        if len(self.routeSegments) > 0 and self.routeSegments[-1].edgeId == edgeId:
            # if there is already a route segment, only updates its value
            self.routeSegments[-1].update(timeSpent)
        else:
            # if there isn't, creates a new one
            self.routeSegments.append(RouteSegment(edgeId, timeSpent, costSpent))


class DataBase:
    def __init__(self, vehicles:dict, edges:dict) -> None:
        # current simulation run (int), start at 0. 
        # the value -1 is just for initialization
        self.execution = -1
        # dictionary[vKey:str] -> Vehicle 
        self.vehicles = vehicles
        # dictionary[eKey:str] -> Edge
        self.edges = edges
        # dictionary[vKey:str] -> Route
        self.routes = {}
        for vKey in self.vehicles:
            self.routes[vKey] = Route(vKey, [])

    def update(self, vehId:str, edgeId:str, timeSpent:float):
        print("LOG = database.update call for {}, {}, {}".format(vehId, edgeId, timeSpent) )
        self.routes[vehId].update(edgeId, timeSpent, self.edges[edgeId].cost)

    def terminate(self, dataPath:str):
        print("LOG = Terminando execução. Atualizar dados dos motoristas e salvá-los em CSV.")
        for vehicle in self.vehicles:
            print("vehicle = {}".format(vehicle))
            for routeSegment in self.routes[vehicle].routeSegments:
                print("rs = {}".format(routeSegment))
                # only updates estimates over edges crossed by the vehicle
                self.vehicles[vehicle].newEstimate(routeSegment.edgeId, routeSegment.timeSpent)
        self.toCSV(dataPath)

    def toCSV(self, dataPath:str):
        print("LOG = Salvando dados em CSV.")
        with open(dataPath, "w") as file:
            writer = csv.writer(file)
            edges = []
            for eKey in self.edges:
                edges.append(eKey)
            header = ["vehicleId"] + edges
            writer.writerow(header)
            for vKey in self.vehicles:
                data = [vKey]
                for edge in edges:
                    data.append(self.vehicles[vKey].experiencedTime[edge])
                writer.writerow(data)


def createDataBase(weightsPath:str, costsPath:str, dataPath:str, freeFlowTime:dict):
    '''
        weightsPath:str;
            it is the path to file that contains all vehicles weights.
        costsPath:str; 
            it is the path to the file that contains all edges' toll costs.
        freeFlowTime:dict = dictionary[edge:str] -> float;
            it is a dictionary that maps an edge key to its free flow time;
            it will be used to initialize each vehicles' dictionary.
    '''
    edges = readEdgesTollCostsFile(costsPath)
    vehicles = readWeightsFile(weightsPath)
    print("LOG = Criando base de dados.")
    if os.path.isfile(dataPath):
        print("LOG = Entrei no IF")
        # if there is previous data, load that data
        df = pd.read_csv(dataPath)
        for vKey in vehicles:
            experiencedTime = {}
            for eKey in edges:
                experiencedTime[eKey] = df.loc[df["vehicleId"] == vKey][eKey].iloc[0]
            vehicles[vKey].setExperiencedTime(experiencedTime)
    else:
        print("LOG = Entrei no ELSE")
        # for the first estimative, use free flow time
        for vKey in vehicles:
            vehicles[vKey].setExperiencedTime(freeFlowTime)
    print("LOG = Parei criar Banco de Dados.")
    return DataBase(vehicles, edges)

def readWeightsFile(weightsPath: str):
    print("LOG = Lendo arquivo de pesos.")
    # dict[vehicleId:str] -> Vehicle
    vehicles = {}
    df = pd.read_csv(weightsPath)
    for index, row in df.iterrows():
        vehicles[row['veh_id']] = Vehicle(row['veh_id'],row['time_weight'], row['toll_weight'])
    return vehicles

def readEdgesTollCostsFile(costsPath:str):
    print("LOG = Lendo arquivo de custos.")
    # dict[edgeId:str] -> Edge
    edges = {}
    df = pd.read_csv(costsPath)
    for index, row in df.iterrows():
        edges[row['edge_id']] = Edge(row['edge_id'], row['cost'])
    return edges
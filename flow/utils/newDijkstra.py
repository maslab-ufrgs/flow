from flow.networks.MyGrid import *
import pandas as pd
import math
import heapq

'''
required files:
    GRAPH DESCRIPTION AT <flow/inputs/graphs/OW.net>
    EDGES COSTS AT <flow/inputs/data/experiencedTimes.csv>
'''
class Graph():
    def __init__(self, experiencedTimePath, graphPath, costsPath, weightsPath, freeFlowPath, emissionPath=None) -> None:
        # list<str> = [n0, n1, ..., nN],
        # where ni: str, i in [N], is a graph's node.
        self.nodes = []
        # dict[node:str] -> list<str> = [n0, n1, ..., nM],
        # where ni: str, i in [M], is neighbor of node.
        self.edges = {}
        # list<tuple<str,str,int>> = [(o0, d0, n0), (o1, d1, n1), ..., (oK, dK, nK)]
        # where oi:str, i in [K], is the origin node of the vehicle,
        # and di:str, i in [K], is the destiny node of the vehicle,
        # and ni:int, i in [K], indicates how many vehicles have this od pair.
        self.odPairs = []
        # DataFrame, whose header is veh_id, edge_id0, edge_id1, edge_id2, ..., edge_idM,
        # where veh_id:str indicates the id of the vehicle,
        # and edge_idi, i in [M], represents the cost perceived by driver whose id is veh_id over the edge edge_idi.
        self.data = pd.DataFrame()
        # Datagrame, whose header is edge_id, cost,
        # where edgeId:str indicates de the id of the edge,
        # and cost is the cost of crossing over edge_id
        self.costs = pd.read_csv(costsPath)
        # Datagrame, whose header is veh_id, time_weight, toll_weight
        # where veh_id:str indicates de the id of the vehicle,
        # and time_weight and toll_weight are both weights used to calculate the vehicle's cost
        self.weights = pd.read_csv(weightsPath)
        if emissionPath is not None:
            """
            we have previous information in the experiencedTimePath file:
            this means that we have at least one previous run;
            so we also have data in the emissions path, and,
            therefore, we must recover it;
            after recovering it, we adjust the our time estimative based on the last run data (on the emissionsPath file).
            after, we save this new information on the experiencedTimePath file to be used on the next run.
            """
            print("LOG: it is not the first run... using previous data!")
            self.emissions = pd.read_csv(emissionPath).groupby(["id", "edge_id"]).count().reset_index()
            self.data = pd.read_csv(experiencedTimePath)
            self.newEstimate()
        else:
            """
            first run:
            - we do not have previous data except for the free flow information (made by the environment)
            - we, now, save the free flow data as if it where this run data in experiencedTimePath
            - we the next run, we will also have the emissions' data to add to this information
            """
            print("LOG: first run!")
            self.emissions = pd.DataFrame()
            self.data = pd.read_csv(freeFlowPath)
        self.data.to_csv(experiencedTimePath, index=False)
        # print("LOG: data")
        # print(self.data)
        self.readGraph(graphPath)

    # newEstimate = (1 - alpha) oldEstimate + (alpha) step, where:
    # newEstimate = this run
    # oldEstimate = data file
    # step = emission file
    # alpha = 0.1 by default
    def newEstimate(self):
        alpha = 0.1
        # print("######## BEFORE UPDATING THE TABLE ########")
        # print(self.data)
        for index in range(len(self.data)):
            vehicle = self.data.iloc[index][0]
            path = list(self.emissions.loc[self.emissions["id"] == vehicle]["edge_id"])
            for column in range(len(self.data.iloc[index])):
                if not column == 0:
                    edge = self.data.columns[column]
                    if edge in path:
                        oldEstimate = float(self.data.iloc[index, column])
                        step = float(self.emissions.loc[self.emissions["id"] == vehicle].loc[self.emissions["edge_id"] == edge]["time"].iloc[0])
                        newEst= ((1- alpha) * oldEstimate) + (alpha*step)
                        self.data.set_value(index, edge, newEst)
        # print("######## AFTER UPDATING THE TABLE ########")
        # print(self.data)


    def getEdgeValue(self, edge:str, vehicleId:str):
        time = self.data[self.data["veh_id"] == vehicleId][edge].iloc[0]
        toll = self.costs[self.costs["edge_id"] == edge]["cost"].iloc[0]
        timew = self.weights[self.weights["veh_id"] == vehicleId]["time_weight"].iloc[0]
        tollw = self.weights[self.weights["veh_id"] == vehicleId]["toll_weight"].iloc[0]
        return timew*time + tollw*toll

    def readGraph(self, graphPath):
        with open(graphPath) as file:
            for line in file.readlines():
                if not line[0] == '#':
                    spltLine = line.replace("\n", "").split(' ')
                    if spltLine[0] == "node":
                        self.nodes.append(spltLine[1])
                    elif spltLine[0] == "edge":
                        if spltLine[2] in self.edges:
                            self.edges[spltLine[2]].append(spltLine[3])
                        else:
                            self.edges[spltLine[2]] = [spltLine[3]]
                    elif spltLine[0] == "od":
                        self.odPairs.append((spltLine[2], spltLine[3], int(spltLine[4])))

    def getNeighbors(self, node):
        if node in self.edges:
            return self.edges[node]
        else:
            return []

    def getVehicles(self):
        return list(self.data["veh_id"])
    
    def getODPairs(self):
        return self.odPairs

class HeapNode():
    def __init__(self, node, distance, previous) -> None:
        self.node = node
        self.distance = distance
        self.previous = previous
    
    def __lt__(self, nxt):
        return self.distance < nxt.distance
        

def dijkstra(graph:Graph, root:str, vehicleId:str):
    # print("LOG: running dijkstra for vehicle {} on the {}".format(vehicleId, root))
    distance = {}
    previous = {}
    for n in graph.nodes:
        distance[n] = math.inf
        previous[n] = None
    distance[root] = 0
    q = [HeapNode(root, 0, None)]
    while q:
        heapNode = heapq.heappop(q)
        node, dist, prev = heapNode.node, heapNode.distance, heapNode.previous
        if dist > distance[node]:
            continue
        previous[node] = prev
        for neighbor in graph.getNeighbors(node):
            edge = node + neighbor
            alt = distance[node] + graph.getEdgeValue(edge, vehicleId)
            if alt < distance[neighbor]:
                distance[neighbor] = alt
                heapq.heappush(q,HeapNode(neighbor, alt, node))

    return previous, distance

def makeRoute(predecessors:dict, destination:str):
    sequenceOfNodes = [destination]
    pred = predecessors[destination]
    while pred is not None:
        sequenceOfNodes.append(pred)
        pred = predecessors[pred]
    sequenceOfNodes = sequenceOfNodes[::-1]

    #############################
    # only for Ortuzar network
    #############################
    if sequenceOfNodes[0] == 'A':
        route = ["A1A"]
    else:
        route = ["B1B"]
    #############################

    for i in range(len(sequenceOfNodes)):
        if 1 <= i < len(sequenceOfNodes):
            edge = sequenceOfNodes[i-1] + sequenceOfNodes[i] 
            route.append(edge)

    #############################
    # only for Ortuzar network
    #############################
    if sequenceOfNodes[-1] == 'L':
        route.append("LL1")
    else:
        route.append("MM1")
    #############################
    return route

def routeToXML(route, vehicleId, vehicleType="default", depart=0.0, departPos="base", departSpeed="0.0", arrivalPos="max", arrivalSpeed="current"):
    strId = "id=\"{}\"".format(vehicleId)
    strType = "type=\"{}\"".format(vehicleType)
    strDepart = "depart=\"{}\"".format(depart)
    strDepartPos = "departPos=\"{}\"".format(departPos)
    strDepartSpeed= "departSpeed=\"{}\"".format(departSpeed)
    strArrivalPos = "arrivalPos=\"{}\"".format(arrivalPos)
    strArrivalSpeed="arrivalSpeed=\"{}\"".format(arrivalSpeed)
    strEdge = ""
    for edge in route:
        strEdge += edge + " "
        strEdges="edges=\"{}\"".format(strEdge[:-1])
    strXmlRoute = "   <vehicle {} {} {} {} {} {} {}>\n".format(
        strId, 
        strType,
        strDepart,
        strDepartPos,
        strDepartSpeed,
        strArrivalPos,
        strArrivalSpeed)
    strXmlRoute += "      <route {}/>\n".format(strEdges)
    strXmlRoute += "   </vehicle>\n"
    return strXmlRoute

def routesFileData(graph:Graph):
    routes = []
    vehicles = graph.getVehicles()
    odPairs = graph.getODPairs()
    counter = 0
    print("LOG: generating routes")
    for origin, destination, numVehicles in odPairs:
        while numVehicles != 0:
            vehicle = vehicles[counter]
            prev, dist = dijkstra(graph, origin, vehicle)
            routes.append(
                routeToXML(
                        route=makeRoute(prev, destination),
                        vehicleId=vehicle, 
                    )
                )
            counter += 1
            numVehicles -= 1
    print("LOG: routes generated")
    return routes



def createRoutesFile(graph:Graph, routesPath:str):
    with open(routesPath, "w") as xmlFile:
        xmlFile.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
        xmlFile.write("<routes xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:noNamespaceSchemaLocation=\"http://sumo.dlr.de/xsd/routes_file.xsd\">\n")
        for route in routesFileData(graph):
            xmlFile.write(route)
        xmlFile.write("</routes>")


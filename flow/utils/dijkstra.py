from flow.networks.MyGrid import *
import math
import heapq

def neighbors(source, edges):
    '''
    input: 
        source: str
        edges: list of str
    output:
        list of neighbors: list of str
    '''
    list_of_neighbors = []
    for edge in edges:
        node_from, node_to = split_string_node_id(edge)
        if node_from == source and node_to not in list_of_neighbors: 
            list_of_neighbors.append(node_to)
    return list_of_neighbors

def split_string_node_id(edge):
    """
        input: 
            edge: str, which format is "fromAtoB"
        output: 
            A: str
            B: str
    """
    edge = edge.replace("from", "")
    splited_string = edge.split("to")
    return splited_string[0], splited_string[1]

def edge_cost(u, v, env, veh_id):
    edge = "from" + u + "to" + v
    time = env.get_expecience_time(veh_id, edge)
    toll = env.get_toll_cost(edge)
    timew = env.get_timew(veh_id)
    tollw = env.get_tollw(veh_id)
    # print('id:{}\ntimew:{},time:{}\ntollw:{},toll:{}\n'.format(veh_id,timew,time,tollw,toll))
    # print('veh_id:{}\nedge: {}\ntime: {}\ntoll: {}\ntimew: {}\ntollw: {}\n'.format(veh_id, edge, time, toll, timew, tollw))
    return timew*time + tollw*toll

# ok
def make_graph(edges):
    nodes_list = []
    for edge in edges:
        node_to, node_from = split_string_node_id(edge)
        if node_to not in nodes_list:
            nodes_list.append(node_to)
        if node_from not in nodes_list:
            nodes_list.append(node_from)
    return nodes_list


def dijkstra(source, env, veh_id, nodes_not_to_be_considered = []):
    # recover edges from environment
    edges_dict = env.k.network._edges
    edges = []
    for key in edges_dict:
        if key not in env.k.network._junction_list:
            edges.append(key)
    # recover nodes from edges
    nodes = make_graph(edges)
    print(nodes)
    print(nodes_not_to_be_considered)
    for n in nodes_not_to_be_considered:
        nodes.remove(n)
    distance = {}
    previous = {}
    undefined = 'undefined'
    for n in nodes:
        distance[n] = math.inf
        previous[n] = undefined
    distance[source] = 0
    q = [(source, 0, undefined)]
    while q:
        node, dist, prev = heapq.heappop(q)
        if dist > distance[node] or node in nodes_not_to_be_considered:
            continue
        previous[node] = prev

        for neighbor in neighbors(node, edges):
            if neighbor not in nodes_not_to_be_considered:
                alt = distance[node] + edge_cost(node, neighbor, env, veh_id)
                if alt < distance[neighbor]:
                    distance[neighbor] = alt
                    heapq.heappush(q,(neighbor, alt, node))

    return previous, distance


def make_path_from_dijkstra(dict_previous, destiny):
    dst = deepcopy(destiny)
    path = [dst]
    undefined = "undefined"
    while dst is not undefined:
        for key in dict_previous:
            if dst == key:
                dst = dict_previous[key]
                if dst is not undefined:
                    path.append(dst)
                break
    path.reverse()
    return path

def make_route(nodes_list):
    route = []
    for index in range(len(nodes_list) -1):
        edge_from = "from" + nodes_list[index]
        edge_to = "to" + nodes_list[index+1]
        route.append(edge_from + edge_to)
    return route

def gen_dijkstra_route(env, edge, veh_id):
    if not edge in env.k.network._junction_list:
        node_from, node_to = split_string_node_id(edge)
        previous, distance = dijkstra(node_to, env, veh_id, [node_from])
        path = [node_from]
        path.extend(make_path_from_dijkstra(previous, env.get_destiny(veh_id)))        
        route = make_route(path)
        return route
    else:
        return None
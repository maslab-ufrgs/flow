"""Contains a list of custom routing controllers."""
from copy import deepcopy
import random
import numpy as np
import linecache

from flow.controllers.base_routing_controller import BaseRouter
from flow.networks.MyGrid import ADDITIONAL_NET_PARAMS
from flow.utils.dijkstra import *

class ContinuousRouter(BaseRouter):
    """A router used to continuously re-route of the vehicle in a closed ring.

    This class is useful if vehicles are expected to continuously follow the
    same route, and repeat said route once it reaches its end.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class.

        Adopt one of the current edge's routes if about to leave the network.
        """
        edge = env.k.vehicle.get_edge(self.veh_id)
        current_route = env.k.vehicle.get_route(self.veh_id)

        if len(current_route) == 0:
            # this occurs to inflowing vehicles, whose information is not added
            # to the subscriptions in the first step that they departed
            return None
        elif edge == current_route[-1]:
            # choose one of the available routes based on the fraction of times
            # the given route can be chosen
            num_routes = len(env.available_routes[edge])
            frac = [val[1] for val in env.available_routes[edge]]
            route_id = np.random.choice(
                [i for i in range(num_routes)], size=1, p=frac)[0]

            # pass the chosen route
            return env.available_routes[edge][route_id][0]
        else:
            return None


class MinicityRouter(BaseRouter):
    """A router used to continuously re-route vehicles in minicity network.

    This class allows the vehicle to pick a random route at junctions.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        vehicles = env.k.vehicle
        veh_id = self.veh_id
        veh_edge = vehicles.get_edge(veh_id)
        veh_route = vehicles.get_route(veh_id)
        veh_next_edge = env.k.network.next_edge(veh_edge,
                                                vehicles.get_lane(veh_id))
        not_an_edge = ":"
        no_next = 0

        if len(veh_next_edge) == no_next:
            next_route = None
        elif veh_route[-1] == veh_edge:
            random_route = random.randint(0, len(veh_next_edge) - 1)
            while veh_next_edge[0][0][0] == not_an_edge:
                veh_next_edge = env.k.network.next_edge(
                    veh_next_edge[random_route][0],
                    veh_next_edge[random_route][1])
            next_route = [veh_edge, veh_next_edge[0][0]]
        else:
            next_route = None

        if veh_edge in ['e_37', 'e_51']:
            next_route = [veh_edge, 'e_29_u', 'e_21']

        return next_route


class GridRouter(BaseRouter):
    """A router used to re-route a vehicle in a traffic light grid environment.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        if len(env.k.vehicle.get_route(self.veh_id)) == 0:
            # this occurs to inflowing vehicles, whose information is not added
            # to the subscriptions in the first step that they departed
            return None
        elif env.k.vehicle.get_edge(self.veh_id) == \
                env.k.vehicle.get_route(self.veh_id)[-1]:
            return [env.k.vehicle.get_edge(self.veh_id)]
        else:
            return None


class BayBridgeRouter(ContinuousRouter):
    """Assists in choosing routes in select cases for the Bay Bridge network.

    Extension to the Continuous Router.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        edge = env.k.vehicle.get_edge(self.veh_id)
        lane = env.k.vehicle.get_lane(self.veh_id)

        if edge == "183343422" and lane in [2] \
                or edge == "124952179" and lane in [1, 2]:
            new_route = env.available_routes[edge + "_1"][0][0]
        else:
            new_route = super().choose_route(env)

        return new_route


class I210Router(ContinuousRouter):
    """Assists in choosing routes in select cases for the I-210 sub-network.

    Extension to the Continuous Router.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        edge = env.k.vehicle.get_edge(self.veh_id)
        lane = env.k.vehicle.get_lane(self.veh_id)

        # vehicles on these edges in lanes 4 and 5 are not going to be able to
        # make it out in time
        if edge == "119257908#1-AddedOffRampEdge" and lane in [5, 4, 3]:
            new_route = env.available_routes[
                "119257908#1-AddedOffRampEdge"][0][0]
        else:
            new_route = super().choose_route(env)

        return new_route

class MyGridRandomRouter(BaseRouter):
    def choose_route(self, env):
        current_edge = env.k.vehicle.get_edge(self.veh_id)
        current_route = env.k.vehicle.get_route(self.veh_id)

        if len(current_route) == 0:
            return None
        else:
            rts_dict = deepcopy(env.k.network.rts)
            available_rts = []
            for key in rts_dict:
                for tup in rts_dict[key]:
                    if current_edge in tup[0]:
                        counter = 0
                        while current_edge != tup[0][counter]:
                            counter += 1
                        del tup[0][0:counter]
                        available_rts.append(tup[0])
            if len(available_rts) == 0:
                return None
            else: 
                route_index = np.random.choice(len(available_rts), size=1)[0]
                return available_rts[route_index]

class MyGridRouterOnlyWhenVehiclesAreReseting(BaseRouter):
    def choose_route(self, env):
        veh_id = self.veh_id
        edge = env.k.vehicle.get_edge(veh_id)
        env.update_vehicle_route(veh_id, edge)
        # if environment is reseting: apply dijkstra to get shortest path.
        if env.ableToApplyDijkstra(veh_id):
            route = gen_dijkstra_route(env, edge, veh_id)
            if len(route) > 0 and route[0] == edge:
                return route
            else:
                return None
        return super().choose_route(env)

class MyGridRouterUsingPredefinedRoutes(BaseRouter):
    def choose_route(self, env):
        veh_id = self.veh_id
        edge = env.k.vehicle.get_edge(veh_id)
        env.update_vehicle_route(veh_id, edge)
        if env.ableToApplyDijkstra(veh_id):
            # if environment is reseting: read routing file.
            route = self.read_routes_file(env, veh_id)
            if len(route) > 0 and route[0] == edge:
                return route
            else:
                return None
        return super().choose_route(env)
    
    def read_routes_file(self, env, veh_id):
        env.applyDijkstra(veh_id)
        veh_id = self.veh_id
        path = self.router_params['routes_path']
        line = linecache.getline(path, int(veh_id.replace('human_', '')) + 1)
        route = line.replace('\n','').split('-')
        return route
    
# must be used with GeneticAlgorithmEnv
class GeneticAlgorithmRouter(BaseRouter):
    def choose_route(self, env):
        # recovers vehicle and edge
        vehId = self.veh_id
        edgeId = env.k.vehicle.get_edge(vehId)
        # updates vehicle's experience over edge edgeId
        print("LOG: env.update for {}, {}".format(vehId, edgeId))
        env.update(vehId, edgeId)
        return super().choose_route(env)
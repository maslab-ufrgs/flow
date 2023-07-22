# Para executar o teste:
# python main.py -c net/one_junction/one_junction.sumo.cfg -n net/one_junction/one_junction.net.xml
#

#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Default libraries
import os
import sys
import subprocess
import argparse
from time import time

# Include access to development libraries
sys.path.append(os.path.join(os.getcwd(),os.path.dirname(__file__), 'src'))

# Import usefull libs for SUMO
# To use autocomplete of their source in Eclipse, add to your project 
# PYTHONPATH a link to SUMO/tools folder. 
import traci
import sumolib

#
# Constants
#

# For Simulation
DEFAULT_PORT = 8813

# For configuration options
PORT = '--port'
MIN_PORT = '-p'
SUMO_CFG = '--sumo-cfg'
MIN_SUMO_CFG = '-c'
NET = '--net'
MIN_NET = '-n'
END_STEP = '--end-step'
MIN_END_STEP = '-es'

def parse():
    """Read the input and put the words in a dictionary."""
    parser = argparse.ArgumentParser(description='Explain me.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    groupSim = parser.add_argument_group('Simulation group','The parameters that define the basics of simulation execution.')
    groupSim.add_argument(MIN_PORT, PORT, dest=PORT, nargs=1, metavar='number', default=DEFAULT_PORT, help='Flag to inform the port to communicate with traci-hub or SUMO.')

    groupStep = parser.add_argument_group('Steps control group','The commands that manipulate the simulation steps.')
    groupStep.add_argument(MIN_END_STEP, END_STEP, dest=END_STEP, nargs=1, type=int, metavar='number', default=-1, help='The end step interval for capture of the simulation data.')

    groupCfg = parser.add_argument_group('Configuration group','The parameters that define the properties for the used algorithm.')
    groupCfg.add_argument(MIN_SUMO_CFG, SUMO_CFG, dest=SUMO_CFG, nargs=1, metavar='file', required=True, help='SUMO configuration file.')
    groupCfg.add_argument(MIN_NET, NET, dest=NET, nargs=1, metavar='file', required=False, help='Captures the file containing the network to be simulated, to obtain data.')
    args = parser.parse_args()
    params = vars(args)
    # Removes the parameter not inserted in the input
#    for key,value in params.items():
##        if value is None:
##            del params[key]
#        if not isinstance(value, list):
#            params[key] = []
#            params[key].append(value)

    return params

def run(port, sumo_cfg, net_file, end_step):
    """ Run the experiment. """

    sumolib_net = sumolib.net.readNet(net_file)
    net_edges = sumolib_net.getEdges()

    #print [edge.getLength() for edge in sumolib_net.getEdges()]

    sumo_binary = os.environ.get("SUMO_BINARY", "sumo")
        
    subprocess.Popen([sumo_binary, "-c", sumo_cfg])
    traci.init(port)

    # Take all edges
    edges = traci.edge.getIDList()
    edges = filter(lambda x: x.find(':') == -1, edges)

    # Take all lanes that has not under junction (connections)
    lanes = traci.lane.getIDList()
    lanes = filter(lambda x: x.find(':') == -1, lanes)

    totalDeparted = 0
    totalArrived = 0
    step = 0
    
    vehicles = dict()
    LOG: open("log.txt", "w")
    log.write("Vehicle ID;O;D;Travel time\n")
	
    while True:
        #print step
        # All vehicles have arrived
        if step > 5 and totalArrived == totalDeparted:
            traci.close()
            break

        # The maximum time step has been reached
        if step > end_step and end_step != -1:
            traci.close()
            break

        try:
            start = time()
            traci.simulationStep()
            totalDeparted += traci.simulation.getDepartedNumber()
            totalArrived += traci.simulation.getArrivedNumber()
            
            for v in traci.simulation.getDepartedIDList():
                vehicles[v] = [step, step, [traci.vehicle.getLaneID(v)]]
            
            toRemove = []
            for v in vehicles:
                try:
                    t0 = vehicles[v][0]
                    od = vehicles[v][2]
                    od.append(traci.vehicle.getLaneID(v))
                    vehicles[v] = [t0, step, od]
                except:
                    log.write(v + ";" + vehicles[v][2][0] + ";" + vehicles[v][2][-1] + ";" + str(vehicles[v][1] - vehicles[v][0]) + "\n")
                    toRemove.append(v)#del vehicles[v]
            
            for r in toRemove:
                del vehicles[r]
            
            step += 1

        except traci.FatalTraCIError as e:
            #print e.message
            traci.close()
            break
    
    
    #log.write("Vehicle ID;O;D;Travel time\n")
    #sys.stdout.flush()
    #for v in vehicles:
        #log.write(v + ";" + vehicles[v][2][0] + ";" + vehicles[v][2][-1] + ";" + str(vehicles[v][1] - vehicles[v][0]) + "\n")
        #sys.stdout.flush()
    log.close()


def main():
    params = parse()
    port = params[PORT]
    sumo_cfg = params[SUMO_CFG].pop()
    net_file = params[NET].pop()
    end_step = params[END_STEP]

    os.system('clear')
    print "RUNNING"
    run(port, sumo_cfg, net_file, end_step)

    exit()

if __name__ == '__main__':
    main()

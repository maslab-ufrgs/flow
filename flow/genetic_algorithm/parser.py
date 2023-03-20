import argparse
from datetime import datetime

def parse_args(args):
    parser = argparse.ArgumentParser(
        description = "Arguments that will be passed to genetic algorithm.",
        epilog= "python genetic_algorithm.py --num_vehicles=INT --pop_size=INT --num_runs=INT --exp_tag=INT --tournament_size=INT --num_generations=INT --host=STR")
    
    parser.add_argument(
        '--num_vehicles', 
        help = "Number of vehicles that will be tested in the network. This represents the size of one idivodual.", 
        required = False, 
        type=int,
        default = 10)  
    parser.add_argument(
        '--pop_size', 
        help = "Number of individuals that a population will have.", 
        required = False, 
        type=int,
        default = 10)
    parser.add_argument(
        '--num_runs', 
        help = "Number of experiment runs, i.e., number of flow executions over an individual to get its value.", 
        required = False,
        type=int, 
        default = 1)
    parser.add_argument(
        '--exp_tag', 
        help="Experiment's name.", 
        required=False, 
        type=str,
        default=str(datetime.now()))
    parser.add_argument(
        '--tournament_size', 
        help="Number of individuals inside the population that will participate in the tournament.", 
        required=False, 
        type=int,
        default=3)
    parser.add_argument(
        '--num_generations', 
        help="Number of populations that will generated apart from the original population.", 
        required=False,
        type=int, 
        default=1)
    parser.add_argument(
        '--host', 
        help="In which machine I am running the experiments (home, lab).", 
        required=False, 
        type=str,
        default='lab')
    return parser.parse_known_args(args)[0]
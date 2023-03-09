from cProfile import label
from copy import deepcopy
from genericpath import isfile
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os

folder_path = "/home/macsilva/Desktop/maslab/flow/data"
edges_csv="/home/macsilva/Desktop/maslab/flow/data/edges.csv"
junctions_csv="/home/macsilva/Desktop/maslab/flow/data/junctions.csv"
file_path="/home/macsilva/Desktop/maslab/flow/data/"
data_path = "/home/macsilva/Desktop/maslab/flow/processed_data"
plots_dir = '/home/macsilva/Desktop/maslab/flow/plots'
csv_extension = ".csv"
png_extension = '.png'
files_not_to_be_processed = ['junctions.csv', "edges.csv", 'vehicles.csv']
dir_not_to_be_read = ['raw']


def read(experiment):
    if exists_processed_data(experiment):
        edges, junctions = get_processed_data(experiment)
    else:
        edges, junctions = process_data(experiment)
    return edges, junctions

def exists_processed_data(experiment):
    for folder in os.listdir(data_path):
        if folder == experiment:
            return True
    return False

def process_data(experiment):
    os.chdir(folder_path)
    edges = get_edges()
    junctions = get_junctions()
    for f in os.listdir():
        if not f in files_not_to_be_processed and f.endswith(".csv"):
            file_name = f.replace(".csv", "")
            process(edges, junctions, file_name, experiment)
    for f in files_not_to_be_processed:
        df = pd.read_csv(file_path + f)
        df.to_csv(data_path + '/' + experiment + '/raw/' + f, index=False)
    return edges, junctions

def get_junctions(path=junctions_csv):
    junctions_df = pd.read_csv(path)
    junctions = []
    for index, row in junctions_df.iterrows():
        junctions.append(row['junction'])
    return junctions

def get_processed_data(experiment):
    dir = data_path + '/' + experiment + '/raw/'
    edges = get_edges(dir + 'edges.csv')
    junctions = get_junctions(dir + 'junctions.csv')
    return edges, junctions

def get_edges(path=edges_csv):    
    edges_df = pd.read_csv(path)
    edges = []
    for index, row in edges_df.iterrows():
        edges.append(row['edge'])
    return edges

# transform a unique and heavy experiment file into many small files, one for each edge.
def process(edges, junctions, file_name, experiment):
    data = edges + junctions
    print(file_path + file_name + csv_extension)
    df = pd.read_csv(file_path + file_name + csv_extension)

    # get time_steps
    time_steps = list(df.groupby(["time"]).mean().sort_values("time", ascending=True).index.values)
    time_steps_dict = {}
    for index in range(len(time_steps)):
        time_steps_dict[time_steps[index]] = index

    # get num_vehicles and vehicles data
    num_vehicles = {}
    vehicles = {}
    for d in data:
        num_vehicles[d] = [0 for time_step in time_steps]
        vehicles[d] = [[] for time_step in time_steps]
    for index, row in df.iterrows():
        num_vehicles[row["edge_id"]][time_steps_dict[row["time"]]] += 1
        vehicles[row["edge_id"]][time_steps_dict[row["time"]]].append(row["id"])

    # create directories
    os.chdir(data_path)
    if not experiment in os.listdir():
        os.mkdir(experiment)
        os.mkdir(experiment + '/raw')
    df.to_csv(data_path + '/'+ experiment +'/raw/'+file_name+csv_extension)
    dir = data_path + '/' + experiment
    os.chdir(dir)
    if not file_name in os.listdir():
        dir += '/' + file_name
        os.mkdir(dir)
        os.mkdir(dir + '/edges')
        os.mkdir(dir + '/junctions')

    # actually processing the data
    dataframes = []
    for d in data:
        data = {
            "edge": d,
            "time_step": time_steps,
            "num_vehicles":num_vehicles[d],
            "vehicles":vehicles[d],
        }
        new_df = pd.DataFrame(
            {
                "time_step": time_steps,
                d:  num_vehicles[d],                
            }
        )
        dataframes.append(new_df)
        df = pd.DataFrame(data)
        if d in edges:
            df.to_csv(dir + '/edges/' + d + csv_extension, index=False)
        else:
            df.to_csv(dir + '/junctions/' + d + csv_extension, index=False)

    
    
def edges(edges, junctions, experiment, var):
    data = edges + junctions
    for edge in data:
        dataframes = []
        os.chdir(data_path + '/' + experiment)
        for exp_run in os.listdir():
            if not exp_run in dir_not_to_be_read:
                if edge in edges:
                    input_dir = '/edges/'
                else:
                    input_dir = '/junctions/'
                file_name = data_path + "/" + experiment + '/' + exp_run + input_dir + edge + csv_extension
                df = pd.read_csv(file_name)
                dataframes.append(df)
        if var == 'edges_heatmap':
            plot_heatmap(dataframes, edge, '/edges_heatmap/', experiment)
        elif var == 'edges_meantime':
            plot_edges_meantime(dataframes, edge, '/edges_meantime/', experiment)
        elif var == 'edges_pressure':
            plot_edges_pressure(dataframes, edge, '/edges_pressure/', experiment)
        else: 
            quit()

def plot_edges_pressure(dataframes, edge, plot_dir, experiment):
    pressures = []
    for df in dataframes:
        pressure = []
        for index, row in df.iterrows():
            current = row['vehicles'].replace('"','').replace('[', '').replace(']', '').replace(' ', '').split(',')
            if not index == 0:
                previous = df.loc[index-1]['vehicles'].replace('"','').replace('[', '').replace(']', '').replace(' ', '').split(',')
            else:
                previous = []
            in_veh = [veh for veh in current if veh not in previous]
            out_veh = [veh for veh in previous if veh not in current]
            pressure.append(len(in_veh) - len(out_veh))
        pressures.append(pressure)

    plt.clf()
    for i in range(len(dataframes)):
        plt.plot([j for j in range(len(pressures[i]))], pressures[i])
    plt.title(edge + ' pressure')
    plt.savefig(get_plot_path(experiment, plot_dir, edge))
            


def plot_edges_meantime(dataframes, edge, plot_dir, experiment):
    data = pd.DataFrame()
    counter = 0
    for df in dataframes:
        df['run'] = counter
        data = pd.concat([data, df], axis=0)
        counter += 1
    plt.clf()
    sns.boxplot(x='run', y='time_step', hue='run', data=data)
    plt.xlabel('run')
    plt.ylabel('time')
    plt.title('vehicles mean time in edge "' + edge +'"')
    plt.savefig(get_plot_path(experiment, plot_dir, edge))

def plot_heatmap(dataframes, edge, plot_dir, experiment):
    data = pd.DataFrame()
    counter = 0
    for df in dataframes:
        df['run'] = counter 
        data = pd.concat([data, df], axis=0, ignore_index=True)
        counter += 1
    plt.clf()
    sns.lineplot(x='time_step', y='num_vehicles', hue='run', data=data)
    plt.ylabel('number of vehicles')
    plt.xlabel('time')
    plt.title(edge + ' heatmap')
    plt.savefig(get_plot_path(experiment, plot_dir, edge))


def get_plot_path(experiment, plot_dir, data):
    os.chdir(plots_dir)
    if experiment in os.listdir():
        os.chdir(plots_dir + experiment)
        if not plot_dir.replace('/', '') in os.listdir(): 
            os.mkdir(plots_dir + experiment + plot_dir)
    else:
        os.mkdir(plots_dir + experiment)
        os.mkdir(plots_dir + experiment + plot_dir)
    return plots_dir + experiment + plot_dir + data + png_extension
            
    
def vehicles(experiment, var):
    file = data_path + '/' + experiment + '/raw/vehicles.csv'
    df = pd.read_csv(file)
    grouped = df.groupby(['veh_id','run']).sum().reset_index()
    max_run = df.max()['run'] + 1
    for i in range(int(len(grouped.index)/max_run)):
        sns.barplot(x='run', y=var, data=grouped.iloc[i*max_run:i*max_run + max_run])
        plt.title("{} {}".format(grouped.iloc[i*max_run]['veh_id'], var))
        plt.savefig(get_plot_path(experiment, '/vehicles_' + var + '/', grouped.iloc[i*max_run]['veh_id']))

def plot_network_meantime(experiment):
    file = data_path + '/' + experiment + '/raw/vehicles.csv'
    df = pd.read_csv(file)
    sns.boxplot(x='run',y='time', data=df)
    plt.title('network mean time')
    plt.xlabel('run')
    plt.ylabel('time (s)')
    plt.savefig(get_plot_path(experiment, '/network_meantime/', 'meantime'))

def plot_network_throughput(experiment):
    file = data_path + '/' + experiment + '/raw/vehicles.csv'
    df = pd.read_csv(file)
    grouped = df.groupby(['run']).max().reset_index()
    num_vehicles = len(df.groupby(['veh_id']).count().index)
    grouped['throughput'] = num_vehicles/ grouped['time'] 
    sns.barplot(x='run',y='throughput', data=grouped)
    plt.title('network throughput')
    plt.xlabel('run')
    plt.ylabel('num_vehicles/ simulation time')
    plt.savefig(get_plot_path(experiment, '/network_throughput/', 'throughput'))

def network(experiment, var):
    if var == 'network_meantime':
        plot_network_meantime(experiment)
    elif var == 'network_throughput':
        plot_network_throughput(experiment)
    else:
        quit()

def main():
    operation, experiment = sys.argv[1],sys.argv[2]
    e, j = read(experiment)
    if operation == 'edges_heatmap':
        edges(e, j, experiment, 'edges_heatmap')
    # ok
    elif operation == 'veh_cost':
        vehicles(experiment, 'cost')
    # ok
    elif operation == 'veh_time':
        vehicles(experiment, 'time')
    # ok
    elif operation == 'edges_meantime':
        edges(e, j, experiment, 'edges_meantime')
    # conferir
    elif operation == 'edges_pressure':
        edges(e, j, experiment, 'edges_pressure')
    # ok
    elif operation == 'network_meantime':
        network(experiment, 'network_meantime')
    # ok
    elif operation == 'network_throughput':
        network(experiment, 'network_throughput')
    else:
        quit()

main()
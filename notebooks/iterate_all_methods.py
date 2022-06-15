import numpy as np
import pandas as pd
import glob
import os
import git
import pm4py
from pm4py.objects.log.importer.xes import factory as xes_import_factory
from eventlog import EventLog
from log_iteration import Iteration

# import importlib
# importlib.reload(from replearn.eventlog import EventLog)

# get all files from Folder Iteration ending with 'json.gz'
event_log_path = '/logs/iteration/*.json.gz'
git_path = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")
final_path = git_path+event_log_path
files = glob.glob(final_path)

column_names = {    0:'Method', 
                    1:'ari',
                    2:'nmi',
                    3:'b3',
                    4:'0',
                    5:'homogeneity',
                    6:'completeness',
                    7:'distribution'}



n = 10
cluster = 'k_means'

combined_results = pd.DataFrame()
combined_results.rename(columns = column_names, inplace = True) 

for filepath in files:

    # load eventlog 
    # event log configuration
    event_log_path = filepath
    file_name = os.path.basename(filepath)

    case_attributes = None # auto-detect attributes
    event_attributes = ['concept:name', 'user'] # use activity name and user
    true_cluster_label = 'cluster'

    # load file
    event_log = EventLog(file_name, case_attributes=case_attributes, event_attributes=event_attributes, true_cluster_label=true_cluster_label)
    event_log.load(event_log_path, False)
    event_log.preprocess()


    # hyperparameters
    n_epochs = n
    n_batch_size = 64
    n_clusters = 5
    vector_size = 32
    cluster_method = cluster
    hyperparameters = [n_epochs,n_batch_size,n_clusters,vector_size,cluster_method]
    
    # get combined results for current file and add filename as first column 
    current_combined_results = Iteration.get_combined_results(event_log,hyperparameters,column_names)
    current_combined_results.insert(loc=0, column='Filename', value=file_name)

    # add current_combined_results to overall combined results df 
    combined_results = combined_results.append(current_combined_results)

print ("pre csv")

combined_results.to_csv(f'tab2_2018_{cluster_method}_{n_epochs}_epochs', encoding='utf-8', index=False, sep=';')

print ("succesful")
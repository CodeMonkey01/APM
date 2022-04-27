import numpy as np
import pandas as pd
import glob
import os
import git
from pm4py.objects.log.importer.xes import factory as xes_import_factory

from replearn.eventlog import EventLog
from replearn.embedding_predict import EmbeddingPredict
from replearn.autoencoder import AutoencoderRepresentation
from replearn.doc2vec import Doc2VecRepresentation
from replearn.clustering import Clustering


# Iterate through files 
def get_combined_results(event_log,hyperparameters):
    column_names = {    0:'Method', 
                        1:'b1',
                        2:'b2',
                        3:'b3',
                        4:'b4',
                        5:'b5',
                        6:'b6',
                        7:'b7'}
    output_df = pd.DataFrame()
    output_df.rename(columns = column_names, inplace = True) 
    
    # Autoencoder
    current_cluster_result = execute_autoencoder(event_log,hyperparameters)
    current_cluster_result = ('Autoencoder',) + current_cluster_result
    temp_df = pd.DataFrame(current_cluster_result).transpose()
    temp_df.rename(columns = column_names, inplace = True)
    output_df = output_df.append(temp_df)

    # TRACE2VEC(event_log,append_case_attr=False, append_event_attr=False)
    current_cluster_result = execute_doc2Vec(event_log,hyperparameters,False, False)
    current_cluster_result = ('TRACE2VEC',) + current_cluster_result
    temp_df = pd.DataFrame(current_cluster_result).transpose()
    temp_df.rename(columns = column_names, inplace = True)
    output_df = output_df.append(temp_df)

    # CASE2VEC(event_log,append_case_attr=False, append_event_attr=True)
    current_cluster_result = execute_doc2Vec(event_log,hyperparameters,False, True)
    current_cluster_result = ('CASE2VEC E',) + current_cluster_result
    temp_df = pd.DataFrame(current_cluster_result).transpose()
    temp_df.rename(columns = column_names, inplace = True)
    output_df = output_df.append(temp_df)

    # CASE2VEC (event+case) (event_log,append_case_attr=True, append_event_attr=True)
    current_cluster_result = execute_doc2Vec(event_log,hyperparameters,True, True)
    current_cluster_result = ('CASE2VEC E+C',) + current_cluster_result
    temp_df = pd.DataFrame(current_cluster_result).transpose()
    temp_df.rename(columns = column_names, inplace = True)
    output_df = output_df.append(temp_df)

    # GRU
    current_cluster_result = execute_gru_lstm(event_log,hyperparameters,'gru')
    current_cluster_result = ('GRU',) + current_cluster_result
    temp_df = pd.DataFrame(current_cluster_result).transpose()
    temp_df.rename(columns = column_names, inplace = True)
    output_df = output_df.append(temp_df)

    # LSTM
    current_cluster_result = execute_gru_lstm(event_log,hyperparameters,'LSTM')
    current_cluster_result = ('LSTM',) + current_cluster_result
    temp_df = pd.DataFrame(current_cluster_result).transpose()
    temp_df.rename(columns = column_names, inplace = True)
    output_df = output_df.append(temp_df)

    return output_df


def execute_autoencoder(event_log,hyperparameters):
    ##################################
    # AUTOENCODER

    # get hyperparameters
    n_epochs = hyperparameters[0]
    n_batch_size = hyperparameters[1]
    n_clusters = hyperparameters[2]
    vector_size = hyperparameters[3]

    # get sequences from event log as one-hot feature vector
    sequences = event_log.event_attributes_flat_onehot_features_2d

    # init and train autoencoder
    autoencoder = AutoencoderRepresentation(event_log)
    autoencoder.build_model(sequences.shape[1], encoder_dim=vector_size)
    autoencoder.fit(batch_size=n_batch_size, epochs=n_epochs, verbose=True)

    feature_vector = autoencoder.predict()

    # cluster feature vector
    cluster_analysis = Clustering(event_log)
    cluster_analysis.cluster(feature_vector, 'agglomerative', n_clusters, 'cosine')
    cluster_result = cluster_analysis.evaluate()

    return cluster_result


def execute_doc2Vec(event_log,hyperparameters,append_case_attr, append_event_attr):
    ##################################
    # TRACE2VEC(event_log,append_case_attr=False, append_event_attr=False)
    # CASE2VEC(event_log,append_case_attr=False, append_event_attr=True)
    # CASE2VEC (event+case) (event_log,append_case_attr=True, append_event_attr=True)

    # get hyperparameters
    n_epochs = hyperparameters[0]
    n_batch_size = hyperparameters[1]
    n_clusters = hyperparameters[2]
    vector_size = hyperparameters[3]

    # train model
    doc2vec = Doc2VecRepresentation(event_log)
    doc2vec.build_model(append_case_attr=append_case_attr, append_event_attr=append_event_attr, vector_size=vector_size, concat=True, epochs=n_epochs)
    doc2vec.fit()

    # infer the vector from the model
    feature_vector = doc2vec.predict(epochs=50)

    # cluster feature vector
    cluster_analysis = Clustering(event_log)
    cluster_analysis.cluster(feature_vector, 'agglomerative', n_clusters, 'cosine')
    cluster_result = cluster_analysis.evaluate()

    return cluster_result

def execute_gru_lstm(event_log,hyperparameters,rnn):
    ##################################
    # GRU (event_log,rnn='gru'):
    # LSTM (event_log,rnn='LSTM')

    # get hyperparameters
    n_epochs = hyperparameters[0]
    n_batch_size = hyperparameters[1]
    n_clusters = hyperparameters[2]
    vector_size = hyperparameters[3]

    # init and train Model
    predictor = EmbeddingPredict(event_log)
    predictor.build_model(embedding_dim=vector_size, gru_dim=vector_size, rnn=rnn)
    predictor.fit(epochs=n_epochs, batch_size=n_batch_size, verbose=True)

    # get feature vector
    pred_model, feature_vector, embedding_vector = predictor.predict()

    # cluster feature vector
    cluster_analysis = Clustering(event_log)
    cluster_analysis.cluster(feature_vector, 'agglomerative', n_clusters, 'cosine')
    cluster_result = cluster_analysis.evaluate()

    return cluster_result


# MAIN
# get all files from Folder Iteration ending with 'json.gz'
event_log_path = '/logs/Iteration/*.json.gz'
git_path = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")
final_path = git_path+event_log_path
files = glob.glob(final_path)

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
    n_epochs = 10
    n_batch_size = 64
    n_clusters = 5
    vector_size = 32
    hyperparameters = [n_epochs,n_batch_size,n_clusters,vector_size]

    print(get_combined_results(event_log,hyperparameters))

    
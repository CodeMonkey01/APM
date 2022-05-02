import pandas as pd

from replearn.autoencoder import AutoencoderRepresentation
from replearn.clustering import Clustering
from replearn.doc2vec import Doc2VecRepresentation
from replearn.embedding_predict import EmbeddingPredict


class Iteration:

    # Iterate through files 
    def get_combined_results(event_log,hyperparameters,column_names):
        # column_names = {    0:'Method', 
        #                     1:'b1',
        #                     2:'b2',
        #                     3:'b3',
        #                     4:'b4',
        #                     5:'b5',
        #                     6:'b6',
        #                     7:'b7'}
        output_df = pd.DataFrame()
        output_df.rename(columns = column_names, inplace = True) 
        
        # Autoencoder
        current_cluster_result = Iteration.execute_autoencoder(event_log,hyperparameters)
        current_cluster_result = ('Autoencoder',) + current_cluster_result
        temp_df = pd.DataFrame(current_cluster_result).transpose()
        temp_df.rename(columns = column_names, inplace = True)
        output_df = output_df.append(temp_df)

        # TRACE2VEC(event_log,append_case_attr=False, append_event_attr=False)
        current_cluster_result = Iteration.execute_doc2Vec(event_log,hyperparameters,False, False)
        current_cluster_result = ('TRACE2VEC',) + current_cluster_result
        temp_df = pd.DataFrame(current_cluster_result).transpose()
        temp_df.rename(columns = column_names, inplace = True)
        output_df = output_df.append(temp_df)

        # CASE2VEC(event_log,append_case_attr=False, append_event_attr=True)
        current_cluster_result = Iteration.execute_doc2Vec(event_log,hyperparameters,False, True)
        current_cluster_result = ('CASE2VEC E',) + current_cluster_result
        temp_df = pd.DataFrame(current_cluster_result).transpose()
        temp_df.rename(columns = column_names, inplace = True)
        output_df = output_df.append(temp_df)

        # CASE2VEC (event+case) (event_log,append_case_attr=True, append_event_attr=True)
        current_cluster_result = Iteration.execute_doc2Vec(event_log,hyperparameters,True, True)
        current_cluster_result = ('CASE2VEC E+C',) + current_cluster_result
        temp_df = pd.DataFrame(current_cluster_result).transpose()
        temp_df.rename(columns = column_names, inplace = True)
        output_df = output_df.append(temp_df)

        # GRU
        current_cluster_result = Iteration.execute_gru_lstm(event_log,hyperparameters,'gru')
        current_cluster_result = ('GRU',) + current_cluster_result
        temp_df = pd.DataFrame(current_cluster_result).transpose()
        temp_df.rename(columns = column_names, inplace = True)
        output_df = output_df.append(temp_df)

        # LSTM
        current_cluster_result = Iteration.execute_gru_lstm(event_log,hyperparameters,'LSTM')
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




        
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os
import json

def get_SOMKNN_cluster_threshold(som_grid, total_hits, confidence = 0.9, show_plot=False):
    """
    Compute the threshold that should  be considered to remove culsters found during the training SOM.
    
    Params:
    -------
    - som_grid: grid activation response after SOM training.
    - confidence: makes reference to the percentage of clusters that the SOM are not anomaly clusters. By default we assume that the 90% of the clusters created are not anomalous.
    - show_plot: whether or not showing the cumulative fraction of Hits.
    
    Return:
    -------
    Threshold to ignore clusters created by SOM.
    """
        
    neuron_hits = som_grid.flatten()
    hit_counts = [np.sum(neuron_hits == i) for i in sorted(np.unique(neuron_hits), reverse=True)]
    cumulative_hits = [sum(hit_counts[:i+1]) for i in range(len(hit_counts))]
    cumulative_fraction = [hit/total_hits for hit in cumulative_hits]
    neuron_hits_cumulative = list(zip(sorted(np.unique(neuron_hits), reverse=True), cumulative_fraction))
    
    threshold = 0
    threshold_confidence = 0
    for i in neuron_hits_cumulative:
        if i[1] >= confidence and (i[1] < threshold_confidence or threshold_confidence == 0):
            threshold = i[0]
            threshold_confidence = i[1]
            
    if show_plot:
        fig, ax = plt.subplots()

        ax.plot(sorted(np.unique(neuron_hits), reverse=True), cumulative_fraction, marker='o')
        ax.set_xlim(max(neuron_hits), min(neuron_hits))
        ax.set_xlabel("Hits of Neuron");
        ax.set_ylabel("Cumulative fraction of Hits");

        print(neuron_hits_cumulative)
        print(f"The selected threshold is {threshold}")
    return threshold

class SOMKNN():
    """
    Implements an hybrid algorithm that combines SOM and KNN to anomaly detection. 
    The original idea comes from the paper entitled "Anomaly detection using self-organizing maps-based K-nearest neighbour algorithm".
    """
    def __init__(self,
                 train_data,
                 x_grid_size,
                 y_grid_size,
                 num_features,
                 num_iterations,
                 sigma,
                 lr,
                 topology,
                 neighborhood,
                 outlier_confidence,
                 KNN_neighbors):
        
        self.train_data = train_data.copy()
        self.train_data = self.train_data.values
        self.x_grid_size = x_grid_size
        self.y_grid_size = y_grid_size
        self.num_features = num_features
        self.iterations = num_iterations
        self.sigma = sigma
        self.lr = lr
        self.topology = topology
        self.nf = neighborhood
        self.n_neighbors = KNN_neighbors
        self.outlier_confidence = outlier_confidence
        self.som = None
        self.denoised_bmus = None
    
    def train(self, whis= 1.5, verbose=False):
        self.som = MiniSom(x=self.x_grid_size, 
                  y=self.y_grid_size, 
                  input_len=self.num_features, 
                  sigma=self.sigma, 
                  learning_rate=self.lr,
                  topology=self.topology,
                  neighborhood_function=self.nf
                 )

        self.som.pca_weights_init(self.train_data)
        
        self.som.train(self.train_data, self.iterations)
        if self.topology == "rectangular":
            print(f"Topographic error: {self.som.topographic_error(self.train_data)}")

        threshold = get_SOMKNN_cluster_threshold(self.som.activation_response(self.train_data), 
                                         (self.x_grid_size * self.y_grid_size),
                                         self.outlier_confidence,
                                         show_plot=False)
        
        self.denoised_bmus = self.som.activation_response(self.train_data)
        self.denoised_bmus[np.where(self.denoised_bmus < threshold)] = 0

        training_data = []
        # positions contains the indices of the selected clusters
        self.positions = np.where(self.denoised_bmus > 0)
        for i,j in zip(self.positions[0], self.positions[1]):
            training_data.append(self.som.get_weights()[i,j])

        self.neigh = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.neigh.fit(training_data)
        
        ############################################################
        #####           quantization error using knn           #####
        ############################################################
        
        self.errors = []
        for x in self.train_data:
            self.errors.append(self.predict(x))

        q3, q1 = np.quantile(self.errors, [0.75, 0.25])
        self.anomaly_threshold = q3+whis*(q3-q1)   
                
        if verbose:
            plt.boxplot(self.errors)
            print(f"q3: {q3}, q1: {q1}")
            print(f"Threshold={self.anomaly_threshold}")

    
    def predict(self, test_data):
        distance, element = self.neigh.kneighbors([test_data])
        denoised_ij = [(self.positions[0][e], self.positions[1][e]) for e in element[0]]
        
        health_indicator = 0
        for dn in denoised_ij:
            health_indicator += np.linalg.norm(self.som.get_weights()[dn[0], dn[1]] - [test_data])
        
        return health_indicator/self.n_neighbors
    
    def detect_anomaly(self, test):
        error = self.predict(test)

        return error > self.anomaly_threshold, error


def som_df(df, 
           map_size=(1,1), 
           Sigma=1, 
           Learning_rate=0.3, 
           Neighborhood_function='gaussian', 
           Topology='rectangular', 
           Activation_distance='euclidean', 
           n_int=1000,
           outlier_confidence=0.9,
           knn_neighbors = 1,
           get_model=False,
           save_plot=None,
           plot_name=None,
           show_plot=False,
           use_pca=False,
           pca_threshold=0.8):
    
    """
    Computes self organizing maps using minisom package to detect anomalies
    Update 1.0: motivated by "PCA filtering and probabilistic SOM for network intrusion detection"
    and "Integration of SOM and PCA for Analyzing Sports Economic Data and Designing a Management System" we 
    updated this method to include an option to reduce the dimensionallity of the data using PCA and then feed
    SOM with this data.
    -----------
    Parameters: 
    
    - df (pandas data frame): data frame containing the time start of each on or off state.
    - scale (bool): whether to scale or not the data.
    - n_int (integer): number of iterations of the training data.
    - get_model (boolean): whether to get som model or not.
    - save_plot (boolean): whether to save som plots or not.
    - knn_neighbors (int): number of neighbors to be used on the hybrid SOM-KNN algorithm. Higher values make
                           the algorithm less sensitive to outliers.
    - outlier_confidence (float): parameter to be used on the hybrid SOM-KNN algorithm. It determines the confidence 
                                  level of the cluster created by SOM. A value of 0.9 means that a 90% of the clusters are
                                  "valid" whereas the remaining 10% clusters should be considered as outliers and they must 
                                  be removed.
    - use_pca (bool): whether to use PCA on training data before feeding SOM.
    - pca_threshold (float): if use_pca is True, pca_threshold defines the minimium variance that must be explained by PCA (is
                             used to select the number of components)
                             
    Arguments to pass to the MiniSom function:
    - map_size (tuple): the size of the map.
    - Sigma (float): the radius of each centroid.
    - Learning_rate (float): to control the convergence/divergence of the SOM neural network.
    - Neigborhood_function (character):the function to calculate the limit of each centroid.
    - Topology (character): type of mapping of the data (hexagonal or rectangular).
    - Activation_distance (character): function to compute the distance to the centroid.
    
    Note: only left or right should be True, if both True, only right would be taken into acount
    ------------
    Return:
    
    Pandas data frame with the information of the activity block and if it is an anomaly or not 
    """ 
    df_cpy = df.copy()
               
    if use_pca:
        n_components=1
        explained_variance = 0
        while explained_variance < pca_threshold:
            pca_pipe = make_pipeline(StandardScaler(),
                                    PCA(n_components=n_components))
            
            components=pca_pipe.fit_transform(df_cpy[selected_columns])
            explained_variance = pca_pipe["pca"].explained_variance_ratio_.sum()

            if explained_variance < pca_threshold: n_components += 1

        print(f"It was used {n_components} components in PCA, that explains a {explained_variance*100}% of the variance of the data.")
        df_copy = components.copy()
        
    data = data_preprocessing(df_cpy).copy()
    
    if map_size == (1, 1):
        num_neurons = int(5*(df.shape[0] ** 0.5))
        num_neurons = int(np.sqrt(num_neurons))
        map_size = (num_neurons, num_neurons)
    
    # parametrization of som model
    somknn = SOMKNN(data,
                map_size[0],
                map_size[1],
                data.shape[1],
                n_int,
                Sigma,
                Learning_rate,
                Topology,
                Neighborhood_function,
                outlier_confidence,
                knn_neighbors)

    somknn.train()
    is_outlier = []
    for row in data.values:
        is_anomaly, error = somknn.detect_anomaly(row)
        is_outlier.append(is_anomaly)

    if show_plot:
        plot_som(somknn, data.values, df, map_size, is_outlier, save_plot, plot_name)

    # if use_pca:
    #     df = pd.DataFrame(df)
        
    df_cpy["anomalies"] = is_outlier
               
    if get_model:
        return somknn, df_cpy
    else:
        return None, df_cpy


def plot_som(somknn, data, df, map_size, is_outlier, save=False, name=None):

    winner_coordinates = np.array(list(map(lambda x: somknn.som.winner(x), data))).T
    
    # assigns the cluster to each data point 
    cluster_index = np.ravel_multi_index(winner_coordinates, map_size)
    frequencies = somknn.som.activation_response(data)
      
    # plotting 
    fig, axs = plt.subplots(1,2, figsize = (10, 5))
    
    axs[0].hist(somknn.errors);
    axs[0].axvline(somknn.anomaly_threshold, color = 'black', linestyle = '--')

    axs[1].pcolor(frequencies.T, cmap = 'Blues');
    fig.colorbar(axs[1].pcolor(frequencies.T, cmap = 'Blues'), ax = axs[1])
        

    axs[0].set_title('Quantization error histogram');
    axs[1].set_title('Activation cell colormap'); 
    
    if name:
        if "/" in name:
            title_name = name.split("/")[-1].split("_")[0]
        else:
            title_name = name

        plt.suptitle("COPD exacerbation analysis")
        fig.tight_layout(pad=5.0)
    
    if save:
        plt.savefig(f"{name}.png")
        
    # plt.close(fig)
        

def som_feature_selection(W, labels, target_index = 0, a = 0.04):
    """ Performs feature selection based on a self organised map trained with the desired variables

    INPUTS: W = numpy array, the weights of the map (X*Y*N) where X = map's rows, Y = map's columns, N = number of variables
            labels = list, holds the names of the variables in same order as in W
            target_index = int, the position of the target variable in W and labels
            a = float, an arbitary parameter in which the selection depends, values between 0.03 and 0.06 work well

    OUTPUTS: selected_labels = list of strings, holds the names of the selected features in order of selection
             target_name = string, the name of the target variable so that user is sure he gave the correct input
    """


    W_2d = np.reshape(W, (W.shape[0]*W.shape[1], W.shape[2])) #reshapes W into MxN assuming M neurons and N features
    target_name = labels[target_index]


    Rand_feat = np.random.uniform(low=0, high=1, size=(W_2d.shape[0], W_2d.shape[1] - 1)) # create N -1 random features
    W_with_rand = np.concatenate((W_2d,Rand_feat), axis=1) # add them to the N regular ones
    W_normed = (W_with_rand - W_with_rand.min(0)) / W_with_rand.ptp(0) # normalize each feature between 0 and 1

    Target_feat = W_normed[:,target_index] # column of target feature

    # Two conditions to check against a
    Check_matrix1 = abs(np.vstack(Target_feat) - W_normed)
    Check_matrix2 = abs(np.vstack(Target_feat) + W_normed - 1)
    S = np.logical_or(Check_matrix1 <= a, Check_matrix2 <= a).astype(int) # applie "or" element-wise in two matrices

    S[:,target_index] = 0 #ignore the target feature so that it is not picked

    selected_labels = []
    while True:

        S2 = np.sum(S, axis=0) # add all rows for each column (feature)

        if not np.any(S2 > 0): # if all features add to 0 kill
            break

        selected_feature_index = np.argmax(S2) # feature with the highest sum gets selected first

        if selected_feature_index > (S.shape[1] - (Rand_feat.shape[1] + 1)): # if random feature is selected kill
            break


        selected_labels.append(labels[selected_feature_index])

        # delete all rows where selected feature evaluates to 1, thus avoid selecting complementary features
        rows_to_delete = np.where(S[:,selected_feature_index] == 1)
        S[rows_to_delete, :] = 0

#     selected_labels = [label for i, label in enumerate(labels) if i in feature_indeces]
    return selected_labels, target_name


def plot_high_dimensional_data(data, target, tag=None, data_class=None, show=True):
    """
    If there are more than 3 features, this method reduces it dimensionality to
    three and show the data in a 3d plot. The method is mainly created for anomaly
    detection representation.
    
    Params:
        - data: dataset
        - target: set of tags that classify an instance as anomaly or not.
    """

    labels = {"0": data.columns[0], "1": data.columns[1], "2": data.columns[2]}
    if data.shape[1] > 3:
        n_components=3

        pca_pipe = make_pipeline(StandardScaler(),
                                PCA(n_components=n_components))
        
        components=pca_pipe.fit_transform(data)
        total_var = pca_pipe["pca"].explained_variance_ratio_.sum() * 100
        labels = {'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
    else:
        components = data.values
        total_var = 100

    
    fig = px.scatter_3d(
        components, x=0, y=1, z=2, 
        color=target, 
        symbol=data_class,
        hover_name=tag,
        title=f'Total Explained Variance: {total_var:.2f}%',
        labels=labels
        )
    
    fig.update_traces(marker_size=5, marker_line_width=1.5)
    fig.update_layout(legend_orientation='h')
    
    if show:
        fig.show(renderer="iframe")
    else: return fig
    

def get_anomalies_from_dbscan(df):
    X = data_preprocessing(df)
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)

    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    # fig = px.line(distances)
    # fig.show(renderer="iframe")

    epsilon = 1.95#float(input("Insert an epsilon value for DBSCAN algorithm"))

    db = DBSCAN(eps=epsilon, min_samples=3).fit(X) # the rule of thumb to minPts is 2*dim
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    dbscan_labels = db.labels_

    return df.iloc[np.where(dbscan_labels == -1)].index, db
    
def get_anomalies_from_isolation_forest(df):
    isolation_model = IsolationForest(
                    n_estimators=750,
                    max_samples='auto',
                    contamination=0.01,
                    n_jobs=-1,
                    random_state=30
    )
    isolation_model.fit(df)
    anomalies = isolation_model.predict(df)

    return anomalies, isolation_model

def get_anomalies_from_SVM(df):
    X = data_preprocessing(df)
    svm = OneClassSVM(kernel="rbf", gamma=0.001, nu=0.05)
    svm.fit(X)
    anomalies = svm.predict(X)

    return anomalies, svm

def ensemble_anomaly_models(df, features, show_clusters = False, show_variable_importance = False):
    #                   |--> DBSCAN --->|
    # input_data -> SOM |--> IFOREST -->| --> anomalies
    #                   |--> SVM ------>|
    data = df[features].copy()

    som_model, df_result = som_df(data,
        (9, 5), 
        Sigma=1.4, 
        Learning_rate=0.4, 
        Neighborhood_function='mexican_hat', 
        Topology='hexagonal', 
        Activation_distance='euclidean', 
        n_int=500,
        outlier_confidence=0.9,
        knn_neighbors = 1,
        get_model=True,
        save_plot=False,
        plot_name="copd analytics",
        use_pca = False,
        pca_threshold = 0.8)

    anomalies_index = list(df_result[df_result.anomalies].index)
    patients = df.iloc[anomalies_index].user_id.unique()
    
    
    filtered_df = df[df['user_id'].isin(patients)].copy()
    dbscan_anomalies, dbscan_model = get_anomalies_from_dbscan(filtered_df[features], 1.95)

    iforest_anomalies, isolation_model = get_anomalies_from_isolation_forest(filtered_df[features])

    svm_anomalies, svm_model = get_anomalies_from_SVM(filtered_df[features])

    users_som = df.iloc[anomalies_index][["user_id", "date"]].copy()
    users_dbscan = df.iloc[list(dbscan_anomalies)][["user_id", "date"]].copy()
    users_iforest= filtered_df.reset_index(drop=True).iloc[iforest_anomalies == -1][["user_id", "date"]].copy()
    users_svm = filtered_df.reset_index(drop=True).iloc[svm_anomalies == -1][["user_id", "date"]].copy()

    all_anomalies = pd.concat([users_som, users_dbscan, users_iforest, users_svm]).reset_index(drop=True)
    anomalies_df = filtered_df[filtered_df["date"].isin(all_anomalies["date"])]

    all_anomalies = (all_anomalies.value_counts() / 4).reset_index()
    all_anomalies.columns = ['user_id', 'date', 'anomaly_confidence']

    df = df.merge(all_anomalies, on=['user_id', 'date'], how='left')
    df.anomaly_confidence = df.anomaly_confidence.fillna(0)
    tag = df.user_id + " - " + df.date.dt.strftime("%Y/%m/%d")

    if show_variable_importance:
        explainer = shap.Explainer(isolation_model, filtered_df[features])
        shap_values = explainer(filtered_df[features])

        shap.plots.bar(shap_values.abs.mean(0))

    fig = None
    if show_clusters:

        list_anomalies = np.repeat(1, df.shape[0])
        list_anomalies[np.where(df.anomaly_confidence > 0.25)[0]] = -1

        fig = plot_high_dimensional_data(df[features], list_anomalies, tag, show=False)        
    
    return all_anomalies, df, fig
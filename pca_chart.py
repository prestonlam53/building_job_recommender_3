import pandas as pd
import numpy as np
import process_data as pda
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import cycle
import pickle5 as pickle

#topic_df, model, vec, topic_list = pda.process_data()

#print(topic_list)
#topic_df.to_pickle('topic_df.pkl')

def create_clusters():
    #topic_df, model, vec, topic_list = pda.process_data()
    pca = PCA(n_components=2)
    topic_df = pd.read_csv('topic_df.csv')
    #topic_df = pd.read_pickle('topic_df.pkl')
    X_train = topic_df.iloc[:, 1:-1]
    y_train = topic_df.iloc[:, -1]
    y_vals = y_train.unique()
    model = pca.fit(X_train)

    return X_train, model.transform(X_train), y_train, y_vals, model



def plot_PCA_2D(data, target, target_names, user_data):
    colors = cycle(['black','g','b','c','m','y','orange','w','aqua','yellow'])
    target_ids = range(len(target_names))
    plt.figure(figsize=(10,10))
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(data[target == i, 0], data[target == i, 1],
                   c=c, label=label, edgecolors='gray')
    plt.scatter(user_data[0][0], user_data[0][1], s = 150, color = 'red')
    plt.title('Job Clusters (You are the Red Dot)')
    plt.xlabel('Marketing Design Words')
    plt.ylabel('Project Management Words')
    plt.legend()

def transform_user_resume(pca_model, resume):
    '''
    take in resume and fit it according to both count vectorizer and PCA model
    '''
    #jobs_df, topic_model, vec, topic_list = pda.process_data()
    vec = pickle.load(open('job_vec.sav', 'rb'))
    topic_model = pickle.load(open('topic_model.sav', 'rb'))

    doc = pda.tokenize_stem(resume)
    doc = vec.transform(doc)
    doc = topic_model.transform(doc)
    doc = pd.DataFrame(doc)
#     X_train = doc.iloc[:, :-1]
#     y_train = doc.iloc[:, -1]
#     y_vals = y_train.unique()
    doc = pca_model.transform(doc)
    return doc

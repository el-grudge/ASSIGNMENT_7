import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import NMF
from surprise import KNNBasic
from surprise.model_selection import cross_validate

my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)

file_path = os.path.expanduser('restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

svd = SVD()
pmf = SVD(biased=False)
nmf = NMF()
ucf = KNNBasic(sim_options={'user_based': True})
icf = KNNBasic(sim_options={'user_based': False})

recommenders = {
    'svd': svd,
    'pmf': pmf,
    'nmf': nmf,
    'ucf': ucf,
    'icf': icf
}

performance = {}
for key, value in recommenders.items():
    print(key)
    performance[key] = cross_validate(value, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

metrics = []
for key in performance:
    for i in range(0, 3):
        metrics.append([i, key, performance[key]['test_rmse'][i], performance[key]['test_mae'][i]])

metrics = pd.DataFrame(metrics, columns=['Fold', 'Algorithm', 'RMSE', 'MAE'])

print(metrics[metrics['Fold'] == 0])
print(metrics[metrics['Fold'] == 1])
print(metrics[metrics['Fold'] == 2])
print(metrics.groupby('Algorithm').mean())

metrics_similarity = []
algorithms = ['ucf', 'icf']
similarity = ['msd', 'cosine', 'pearson']

for i in algorithms:
    for j in similarity:
        if i == 'ucf':
            user_based = True
        else:
            user_based = False
        cf = KNNBasic(sim_options={'name': j, 'user_based': user_based})
        metric = cross_validate(cf, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)
        print(np.mean(metric['test_rmse']))
        metrics_similarity.append([i, j, np.mean(metric['test_rmse']), np.mean(metric['test_mae'])])

metrics_similarity = pd.DataFrame(metrics_similarity, columns=['Algorithm', 'Similarity', 'RMSE', 'MAE'])

metrics_similarity[metrics_similarity.columns.difference(['MAE'])].pivot('Similarity', 'Algorithm', 'RMSE').plot(
    kind='bar')
plt.title('RMSE For User CF vs Item CF')
plt.ylabel('RMSE')
plt.ylim(0.9, 1.1)
plt.xticks(rotation=0)
plt.show()

metrics_similarity[metrics_similarity.columns.difference(['RMSE'])].pivot('Similarity', 'Algorithm', 'MAE').plot(
    kind='bar')
plt.title('MAE For User CF vs Item CF')
plt.ylabel('MAE')
plt.ylim(0.7, 0.9)
plt.xticks(rotation=0)
plt.show()

neighbors = [5, 10, 15, 20, 25, 30, 35, 40, 45]
metrics_neighbors = []

for i in algorithms:
    for k in neighbors:
        if i == 'ucf':
            user_based = True
        else:
            user_based = False
        cf = KNNBasic(k=k, sim_options={'name': 'MSD', 'user_based': user_based})
        metric = cross_validate(cf, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)
        metrics_neighbors.append([i, k, np.mean(metric['test_rmse'])])

metrics_neighbors = pd.DataFrame(metrics_neighbors, columns=['Algorithm', 'Neighbours', 'RMSE'])

metrics_neighbors[metrics_neighbors.columns.difference(['MAE'])].pivot('Neighbours', 'Algorithm', 'RMSE').plot()
plt.title('RMSE For User CF vs Item CF With Different K Values')
#plt.ylim(0.9, 1.0)
plt.ylabel('RMSE')
plt.xticks(rotation=0)
plt.show()


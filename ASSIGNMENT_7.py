# 1: pip install scikit-surprise
# 2: download dataset
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import NMF
from surprise import KNNBasic
from surprise.model_selection import cross_validate
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)

# 3: load data from a file
performance = {}
file_path = os.path.expanduser('restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

# 4: metrics MAE & RMSE
# 5: SVD
svd = SVD()
performance['svd'] = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# 6: PMF
pmf = SVD(biased=False)
performance['pmf'] = cross_validate(pmf, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# 7: NMF
nmf = NMF()
performance['nmf'] = cross_validate(nmf, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# 8: User Based Collaborative Filtering
ucf = KNNBasic(sim_options={
    'user_based': True
})
performance['ucf'] = cross_validate(ucf, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# 9: Item Based Collaborative Filtering
icf = KNNBasic(sim_options={
    'user_based': False
})
performance['icf'] = cross_validate(icf, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# 10: Performance Comparison Fold 1
for key in performance:
    print(key, '=', performance[key]['test_rmse'][0])

# 11: Performance Comparison Fold 2
for key in performance:
    print(key, '=', performance[key]['test_rmse'][1])

# 12: Performance Comparison Fold 3
for key in performance:
    print(key, '=', performance[key]['test_rmse'][2])

# 13: Performance Comparison Mean
for key in performance:
    print(key, '=', np.mean(performance[key]['test_rmse']))

# 14: Similarity

similarity = ['MSD', 'cosine', 'pearson']
rmse = []
mae = []

for i in similarity:
    ucf = KNNBasic(sim_options={
        'name': i,
        'user_based': True
    })
    performance['ucf'] = cross_validate(ucf, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

    icf = KNNBasic(sim_options={
        'name': i,
        'user_based': False
    })
    performance['icf'] = cross_validate(icf, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

    rmse.append([i, np.mean(performance['ucf']['test_rmse']), np.mean(performance['icf']['test_rmse'])])
    mae.append([i, np.mean(performance['ucf']['test_mae']), np.mean(performance['icf']['test_mae'])])

rmse_df = pd.DataFrame(rmse, columns=['Similarity', 'UCF_RMSE', 'ICF_RMSE'])
mae_df = pd.DataFrame(mae, columns=['Similarity', 'UCF_MAE', 'ICF_MAE'])

rmse_df.plot(kind="bar")
plt.ylabel("RMSE")
plt.ylim(0.9, 1.1)
plt.xticks([0, 1, 2], similarity, rotation=0)
plt.show()

mae_df.plot(kind="bar")
plt.ylabel("RMSE")
plt.ylim(0.7, 0.8)
plt.xticks([0, 1, 2], similarity, rotation=0)
plt.show()


# 15: Number of k
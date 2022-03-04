import paddle
import pickle
import numpy as np
from torch import true_divide
import glob
# rank_1_tensor = paddle.to_tensor([2.0, 3.0, 4.0], dtype='float64')
# print(rank_1_tensor.detach())

# f = '/root/aistudio/data/Features_competition_test_B/0be864c276ce4c03ac023eb58aaa6306.pkl'
# video_feat = pickle.load(open(f, 'rb'))
# print(type(video_feat))

# f2 = '/root/aistudio/data/Features_competition_test_B/npy/0be864c276ce4c03ac023eb58aaa6306_0.npy'
# video_data = np.load(f2, allow_pickle=True)
# # print(type(video_data))
# print(f'{3.14565623423434:.3f}')

file_list = glob.glob("/root/aistudio/data/Features_competition_test_B/npy/*.npy")
print(type(file_list))
print(len(file_list))

# for f in file_list:
#     sp = f.split('/')[-1].split('.')[0]
    # print(sp)


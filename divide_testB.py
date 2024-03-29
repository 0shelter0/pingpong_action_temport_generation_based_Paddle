import glob
import pickle
import json

import numpy as np
import paddle
import os
import os.path as osp


file_list = glob.glob("/root/aistudio/data/Features_competition_test_B/*.pkl")

max_frames = 9000

npy_path = ("/root/aistudio/data/Features_competition_test_B/npy/")
if not osp.exists(npy_path):
    os.makedirs(npy_path)

for f in file_list:
    video_feat = pickle.load(open(f, 'rb'))
    tensor = paddle.to_tensor(video_feat['image_feature'])
    pad_num = 9000 - tensor.shape[0]
    pad1d = paddle.nn.Pad1D([0, pad_num])
    tensor = paddle.transpose(tensor, [1, 0])
    tensor = paddle.unsqueeze(tensor, axis=0)
    tensor = pad1d(tensor)
    tensor = paddle.squeeze(tensor, axis=0)
    tensor = paddle.transpose(tensor, [1, 0])

    sps = paddle.split(tensor, num_or_sections=90, axis=0)
    for i, s in enumerate(sps):
        file_name = osp.join(npy_path, f.split('/')[-1].split('.')[0] + f"_{i}.npy")
        np.save(file_name, s.detach().numpy())
    pass
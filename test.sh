# train
# python main.py -c configs/localization/bmn.yaml

# export model
# python tools/export_model.py -c configs/localization/bmn.yaml -p output_train1/BMN/BMN_epoch_00010.pdparams -o inference/BMN

# inference unpon test subset
python tools/predict.py --input_file /root/aistudio/data/Features_competition_test_B/npy \
 --config configs/localization/bmn.yaml \
 --model_file inference/BMN/BMN.pdmodel \
 --params_file inference/BMN/BMN.pdiparams \
 --use_gpu=True \
 --use_tensorrt=False

 # merge annotation
#  python merge_json.py
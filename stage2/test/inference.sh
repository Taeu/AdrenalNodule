#python test_gc.py --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 2 --save_result &
#python test_gc.py --pkl_name det2_set1_epoch_82_250.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 2  --save_result &
#python test_gc.py --pkl_name det2_set1_epoch_82_500.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 2 --save_result &
#python test_gc.py --pkl_name det2_set1_epoch_82_750.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 2 --save_result

# python test_gc.py --data_dir ./ag/data/gc_npy/  --pkl_dir ./ag/result/1103_gc/ \
# --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 5 --save_result &
# python test_gc.py --data_dir ./ag/data/gc_npy/  --pkl_dir ./ag/result/1103_gc/ \
# --pkl_name det2_set1_epoch_82_1000.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 5 --save_result &

python test_gc.py --data_dir .//ag/data/data3/numpy/ --pkl_dir ./ag/result/220223_s/ \
--pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 3 --save_result &
python test_gc.py --data_dir .//ag/data/data3/numpy/ --pkl_dir ./ag/result/220223_s/ \
--pkl_name det2_set1_epoch_82_600.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 3 --save_result
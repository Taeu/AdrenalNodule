#python test_gc3.py --data_dir ./ag/data/gc_npy/  --pkl_dir ./ag/result/1103_gc/ \
#--pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 1 --save_result &
#python test_gc3.py --data_dir ./ag/data/gc_npy/  --pkl_dir ./ag/result/1103_gc/ \
#--pkl_name det2_set1_epoch_82_1000.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 1 --save_result &

python test_gc3.py --data_dir ./ag/data3/numpy/ --pkl_dir ./ag/result/220223_s/ \
--pkl_name 220419_save_result_det2_set1_dc_and_fc_with_input_160_0_det2_set1_epoch_82_0.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 3 --save_result 
#python test_gc3.py --data_dir ./ag/data3/numpy/ --pkl_dir ./ag/result/220223_s/ \
#--pkl_name det2_set1_epoch_82_600.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 3 --save_result &
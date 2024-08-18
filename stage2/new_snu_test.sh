
python test_gc.py --data_dir ./ag/data4/numpy/ --pkl_dir ./ag/result/220428_snu/ \
 --pkl_name det2_set1_epoch_82_0.pkl --exp_name 220506_det2_set1_dc_and_fc_with_input_160_unet_squeeze --gpu_id 5 --save_result --squeeze &

python test_gc.py --data_dir ./ag/data4/numpy/ --pkl_dir ./ag/result/220428_snu/ \
 --pkl_name det2_set1_epoch_82_0.pkl --exp_name 220506_det2_set1_dc_and_fc_with_input_160_psp_squeeze --gpu_id 5 --save_result --squeeze &

python test_gc.py --data_dir ./ag/data4/numpy/ --pkl_dir ./ag/result/220428_snu/ \
 --pkl_name det2_set1_epoch_82_0.pkl --exp_name 220506_det2_set1_dc_and_fc_with_input_160_hrnet_ocr --gpu_id 5 --save_result
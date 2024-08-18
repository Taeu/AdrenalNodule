save_dir=./ag/result/stage2/new_finetune1/
ckpt_path=./ag/result/stage2/checkpoints/det2_dc_and_fc_with_input_160_finetune1/best_epoch.pth

# re-run 23.11.22. for SNU and SNUB
# python test/test_231112.py --gpu_id 3 \
#     --data_dir  ./ag/data/data1/Numpy2/ \
#     --pkl_dir  ./ag/result/stage1/ \
#     --pkl_name snu_snub.pkl  \
#     --checkpoint_path ${ckpt_path} \
#     --save_dir ${save_dir} \
#     --save_output 

# re-run 23.11.22. for SC
# python test/test_231112.py --gpu_id 3 \
#     --data_dir  ./ag/data/data3/numpy/ \
#     --pkl_dir  ./ag/result/stage1/ \
#     --pkl_name new_SC.pkl  \
#     --checkpoint_path ${ckpt_path} \
#     --save_dir ${save_dir} \
#     --save_output 

# re-run for GC
# 146
# python test/test_231112.py --gpu_id 3 \
#     --data_dir  ./ag/data/data5/numpy/ \
#     --pkl_dir  ./ag/result/220530_new_gc/ \
#     --pkl_name det2_set1_epoch_82_0.pkl  \
#     --checkpoint_path ${ckpt_path} \
#     --save_dir ${save_dir} \
#     --save_output 
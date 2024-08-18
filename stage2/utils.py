import argparse
def make_parser():
    args = argparse.ArgumentParser()
    #exp
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--save_dir", type=str, default='./ag/result/stage2_0916/')
    args.add_argument("--exp_name", type=str, default = 'unet_1')
    args.add_argument("--train_batch_size", type=int, default=32)
    args.add_argument("--eval_batch_size", type=int, default=64)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument('--class_head', action='store_true', default=False)

    #dataset
    args.add_argument("--root", type=str, default='./ag/data/stage2_0916/')
    args.add_argument("--image_dir", type=str, default='stage1_input_train_unet_crop')
    args.add_argument("--label_dir", type=str, default='stage1_label_train_unet_crop')
    args.add_argument("--output_dir", type=str, default='stage1_output_train_unet_crop')

    args.add_argument("--image_size", type=int, default=160)
    args.add_argument("--advprop", type=bool, default=False)

    #model
    args.add_argument("--model_name", type=str, default="unet")
    args.add_argument("--transfer", type=str, default=None)
    args.add_argument("--dropout", type=float, default=0.2)
    args.add_argument("--num_classes", type=int, default=1)
    #hparams
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--step", type=float, default=15)
    args.add_argument("--num_epochs", type=int, default=60)
    # args.add_argument("--val_same_epoch", type=int, default=20)
    args.add_argument("--weight_decay", type=float, default=1e-5)
    args.add_argument("--optim", type=str, default="rangerlars")
    args.add_argument("--scheduler", type=str, default="cosine")
    args.add_argument("--warmup", type=int, default=5)
    args.add_argument("--cutmix_alpha", type=float, default=1)
    args.add_argument("--cutmix_prob", type=float, default=0.5)

    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--gpu_id", type=str, default="0")
    args.add_argument('--focal_loss', action='store_true', default=False)
    args.add_argument('--no_weights', action='store_true', default=False)
    args.add_argument('--amp', action='store_true', default=False)

    args.add_argument("--test_for_train", type=bool, default=False)
    config = args.parse_args()
    return config
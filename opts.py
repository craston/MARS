import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--frame_dir',
        default='/tmp-network/user/ncrasto/video_db/HMDB51/HMDB51_flows_min256_q',
        type=str,
        help='path of jpg files')
    parser.add_argument(
        '--annotation_path',
        default='/tmp-network/user/ncrasto/video_db/HMDB51/HMDB51_labels',
        type=str,
        help='label paths')
    parser.add_argument(
        '--result_path',
        default='',
        type=str,
        help='result_path')
    parser.add_argument(
        '--only_RGB', 
        action='store_true', 
        help='Extracted only RGB frames')
    parser.set_defaults(only_RGB = False)
    parser.add_argument(
        '--dataset',
        default='HMDB51',
        type=str,
        help='(HMDB51, UCF101, Kinectics)')
    parser.add_argument(
        '--split',
        default=1,
        type=str,
        help='(for HMDB51 and UCF101)')
    parser.add_argument(
        '--modality',
        default='RGB',
        type=str,
        help='(RGB, Flow)')
    parser.add_argument(
        '--input_channels',
        default=3,
        type=int,
        help='(3, 2)')
    parser.add_argument(
        '--n_classes',
        default=400,
        type=int,
        help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument(
        '--n_finetune_classes',
        default=51,
        type=int,
        help=
        'Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
    parser.add_argument(
        '--resize',
        default=360,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--sample_size',
        default=112,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--sample_duration',
        default=16,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--learning_rate',
        default=0.1,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--momentum', 
        default=0.9, 
        type=float, 
        help='Momentum')
    parser.add_argument(
        '--dampening', 
        default=0.9, 
        type=float, 
        help='dampening of SGD')
    parser.add_argument(
        '--weight_decay', 
        default=1e-3, 
        type=float, 
        help='Weight Decay')
    parser.add_argument(
        '--nesterov', 
        action='store_true', 
        help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        '--MARS', 
        action='store_true', 
        help='test MARS')
    parser.set_defaults(MARS=False)
    parser.add_argument(
        '--training', 
        action='store_true', 
        help='training/testing')
    parser.set_defaults(training=True)
    parser.add_argument(
        '--freeze_BN', 
        action='store_true', 
        help='freeze_BN/testing')
    parser.set_defaults(freeze_BN=False)
    parser.add_argument(
        '--optimizer',
        default='sgd',
        type=str,
        help='Currently only support SGD')
    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument(
        '--batch_size', 
        default=20, 
        type=int, 
        help='Batch Size')
    parser.add_argument(
        '--n_workers', 
        default=4, 
        type=int, 
        help='Number of workers for dataloader')
    parser.add_argument(
        '--pretrain_path', 
        default='', 
        type=str, 
        help='Pretrained model (.pth)')
    parser.add_argument(
        '--MARS_pretrain_path', 
        default='', 
        type=str, 
        help='Pretrained model (.pth)')
    parser.add_argument(
        '--MARS_resume_path', 
        default='', 
        type=str, 
        help='MARS resume model (.pth)')
    parser.add_argument(
        '--MARS_alpha', 
        default=1, 
        type=float, 
        help='Weight of Flow augemented MSE loss')
    parser.add_argument(
        '--n_epochs',
        default=400,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument(
        '--resume_path1',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--resume_path2',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--resume_path3',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--log',
        default=1,
        type=int,
        help='Log training and validation')
    parser.add_argument(
        '--checkpoint',
        default=10,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--model_depth',
        default=50,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--output_layers',
        action='append',
        help='layer to output on forward pass')
    parser.set_defaults(output_layers=[])
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument(
        '--resnext_cardinality',
        default=32,
        type=int,
        help='ResNeXt cardinality')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument(
        '--random_seed', default=False, type=bool, help='Manually set random seed of sampling validation clip')
    parser.add_argument(
        '--ft_begin_index',
        default=0,
        type=int,
		help='Begin block index of fine-tuning')
    args = parser.parse_args()

    return args

# _base_ =[
#     '../_base_/datasets/imagenet_bs64_swin_224.py'
# ]
gpus = (0,)
log_dir = 'exp/p2pnet'
workers = 8
print_freq = 20
seed = 42

network = dict(
    module="p2pnet",
    model="P2PNet",
    backbone="VGGBackbone",
    sub_arch='vgg16_bn',
    row = 2,
    line = 2,

    # gaussian kernel configs
    sigma = [4],
    gau_kernel_size = 11,
    baseline_loss = False,
    pretrained_backbone="PretrainedModels/vgg/",
    pred_mask=False,

    matcher=dict(
        type='HungarianMatcher2',
        set_cost_class=1.0,
        set_cost_point=0.05
    ),
    criterion=dict(
        type="SetCriterion_P2PNet",
        num_classes=1,
        eos_coef= 0.5, # 0.15, # 0.5,
        ce_loss_coef=1.0,
        point_loss_coef=0.0002
    )
)

dataset = dict(
    name='SHHA_P2PNet',
    root='ProcessedData/SHTech_P2PNet/part_A/',
    test_set='test.txt',
    train_set='train.txt',
    loc_gt = 'test_gt_loc.txt',
    collate='p2pnet_collate',
    num_classes=2,
    patch=True,
    patch_size=4, # 每次裁剪个数
    extra_train_set =None,
)

train = dict(
    trainer="Trainer_Points_P2PNet",
    counter='normal',
    image_size=(128, 128),  # height width
    route_size=(256, 256),  # height, width
    base_size=2048,
    batch_size_per_gpu=8,
    shuffle=True,
    begin_epoch=0,
    end_epoch=2000,
    extra_epoch=0,
    extra_lr=0,
    #  RESUME: true
    resume_path='exp/p2pnet/SHHA_P2PNet/VGGBackbone_vgg16_bn/SHHA_p2pnet_2024-05-26-00-35',
    flip=True,
    multi_scale=True,
    scale_factor=(0.5, 1/0.5),
    val_start=300,
    # val_span=[-2000], # 测试用
    val_span=[-2000,
              # -1900, -1900,
              # -1800, -1800,
              # -1700, -1700,
              # -1600, -1600,
              # -1500, -1500,
              # -1400, -1400,
              # -1300, -1300,
              # -1200, -1200,
              # -1100, -1100,
              # -1000, -1000,
              # -900, -900,
              # -800, -800,
              # -600, -600,
              # -400, -400,
              # -200, -200,
              # -100, -100
              ],
    save_vis = True,
    save_vis_freq = 200,
    downsamplerate= 1,
    ignore_label= 255
)

test = dict(
    tester = "Tester_Points",
    validator='Validator_Points_P2PNet',
    image_size=(1024, 2048),  # height, width
    no_crop=True, # if set, crop_size will not works
    crop_size=(256, 256), # crop size: height, width
    base_size=2048,
    loc_base_size=(768,2048),
    loc_threshold=0.2,
    batch_size_per_gpu=1,
    patch_batch_size=16,
    flip_test=False,
    multi_scale=False,
    save_vis = True,
    save_vis_freq = 60,

    # For crowd counting
    # model_file= './exp/SHHA/MocHRBackbone_hrnet48/SHHA_HR_2022-10-25-20-1_251_mae_54.5_mse_86.9/Ep_251_mae_54.571960030021245_mse_86.92610575458059.pth'

    # For localization
    model_file='', # './exp/SHHA/MocHRBackbone_hrnet48/SHHA_HR_2022-10-25-20-1_251_mae_54.5_mse_86.9/final_state.pth'
)

optimizer = dict(
    NAME='adam_p2pnet',
    BASE_LR=1e-4,
    BASE_LR_BACKBONE=1e-5,
    BETAS=(0.9, 0.999),
    WEIGHT_DECAY=1e-2,
    EPS= 1.0e-08,
    MOMENTUM= 0.9,
    AMSGRAD = False,
    NESTEROV= True,
    )


lr_config = dict(
    NAME='step',
    WARMUP_METHOD='linear',
    DECAY_EPOCHS=3500,
    DECAY_RATE = 0.1,
    WARMUP_EPOCHS= 0, # 10,   # the number of epochs to warmup the lr_rate
    WARMUP_LR= 1e-4, # 5.0e-07,
    MIN_LR= 1.0e-07,
    no_adjust=True
  )


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

CUDNN = dict(
    BENCHMARK= True,
    DETERMINISTIC= False,
    ENABLED= True)



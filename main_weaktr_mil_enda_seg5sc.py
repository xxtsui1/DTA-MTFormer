import argparse
import datetime
import os.path
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from datagen_endata_seg import ListDataset
from engine_base_mil_enda_seg_sc import train_one_epoch, evaluate, generate_attention_maps_ms
import models_wt_mil_base_seg5s16sc_tiny#models_wt_mil_base_seg5s16sc
import utils_wt
import random
import numpy as np

import mlflow
#from evaluation import whole_eval
import yaml

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--seed', type=int, default=504)
    parser.add_argument('--epoch-save-ckpt', action="store_true")
    parser.add_argument('--deterministic', action="store_true", default=True)
    parser.add_argument('--no-deterministic', action="store_false", dest="deterministic")
    parser.add_argument('--nb_classes', type=int, default=1,help='num classes')

    # distributed parameters
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--gpu', default=None)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist-url', default=None)
    parser.add_argument('--dist-backend', default=None)
    parser.add_argument('--local_rank', default=None, type=int)
    parser.add_argument('--distributed', action="store_true")

    # Model parameters
    parser.add_argument('--model', default='deit_small_WeakTr_patch16_224_mil_base_seg5', type=str, metavar='MODEL',
                        help='Name of model to train')
    #parser.add_argument('--finetune', default='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth', help='finetune from checkpoint')#model deit_small_patch16_224
    #parser.add_argument('--finetune', default='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth', help='finetune from checkpoint')#model deit_tiny_patch16_224 --data-path /path/to/imagenet
    #parser.add_argument('--finetune', default='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth', help='finetune from checkpoint')# --checkpoint_weaktr1_enda_milbaseseg5s16_vitb_best
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # AAF parameters
    parser.add_argument("--reduction", type=float, default=4)
    parser.add_argument("--pool-type", type=str, default="avg")
    parser.add_argument("--feat-reduction", default=8, type=int)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--filter-bias-and-bn', action='store_true', default=True)
    parser.add_argument('--no-filter-bias-and-bn', action='store_false', dest='filter_bias_and_bn')

    parser.add_argument('--skiplist', action='store_true', default=True)
    parser.add_argument('--no-skiplist', action='store_false', dest='skiplist')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=4e-4, metavar='LR',
                        help='learning rate (default: 4e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    #parser.add_argument('--finetune', default='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth', help='finetune from checkpoint')#model deit_small_patch16_224
    parser.add_argument('--finetune', default='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth', help='finetune from checkpoint')#model deit_tiny_patch16_224 --data-path /path/to/imagenet
    #parser.add_argument('--finetune', default='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth', help='finetune from checkpoint')# --checkpoint_weaktr1_enda_milbaseseg5s16_vitb_best
    parser.add_argument('--extra-token', action="store_true", default=True)
    parser.add_argument('--no-extra-token', action="store_false", dest="extra-token")

    # Dataset parameters
    parser.add_argument('--data-path', default='', type=str, help='dataset path')
    parser.add_argument('--img-list', default='', type=str, help='image list path')
    parser.add_argument('--data-set', default='', type=str, help='dataset')

    parser.add_argument('--output_dir', default='seg5test_tiny',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # generating attention maps
    parser.add_argument('--gen_attention_maps', action='store_true')
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--attention-dir', type=str, default=None)
    parser.add_argument('--layer-index', type=int, default=12, help='extract attention maps from the last layers')

    parser.add_argument('--img-ms-list', type=str, default=None)
    parser.add_argument('--patch-attn-refine', action='store_true', default=True)
    parser.add_argument('--no-patch-attn-refine', action='store_false', dest="patch_attn_refine")

    parser.add_argument('--visualize-cls-attn', action='store_true', default=True)
    parser.add_argument('--no-visualize-cls-attn', action='store_false', dest="visualize-cls-attn")

    parser.add_argument('--gt-dir', type=str, default='SegmentationClass')
    parser.add_argument('--cam-npy-dir', type=str, default=None)
    parser.add_argument("--scales", nargs='+', type=float, default=[1.0])
    parser.add_argument('--label-file-path', type=str, default=None)
    parser.add_argument('--attention-type', type=str, default='fused')

    parser.add_argument('--if_eval_miou', action='store_true', default=True)
    parser.add_argument('--no-if_eval_miou', action='store_false', dest="if_eval_miou")

    parser.add_argument('--eval_miou_threshold_start', type=int, default=30)
    parser.add_argument('--eval_miou_threshold_end', type=int, default=60)

    return parser


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    print(args)

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # save config
    variant_file = "variant_attn.yml" if args.gen_attention_maps else "variant.yml"
    variant_str = yaml.dump(args.__dict__)
    with open(output_dir / variant_file, "w") as f:
        f.write(variant_str)

    device = torch.device(args.device)
    if args.local_rank is not None:
        utils_wt.init_distributed_mode(args)
        args.batch_size = args.batch_size // args.world_size
        if args.distributed:
            device = torch.device(args.device, args.local_rank)
    
    if_dataloader_reproduce = False
    if args.seed != None:
        if_dataloader_reproduce = True
        g, worker_init_fn = utils_wt.fix_random_seeds(args.seed, if_warning_only=(not args.input_size == 224), deterministic=args.deterministic)
    else:
        cudnn.benchmark = True
    
    '''
    # Step0: Set our run name in mlflow
    run_name = args.output_dir.split("/")[-1]
    mlflow.start_run(run_name=run_name)
    # Step1: Log our parameters into mlflow
    # Log our parameters into mlflow
    for key, value in vars(args).items():
        mlflow.log_param(key, value)
'''
    trainset = ListDataset(root=['/home/cuixiaoxiao/knowledge_adaptation/endata/atrophy/image', '/home/cuixiaoxiao/knowledge_adaptation/endata/atrophy/label','/home/cuixiaoxiao/knowledge_adaptation/endata/normal/image'],
                           list_file='train.txt',input_size=224, state='Train')
    valset = ListDataset(root=['/home/cuixiaoxiao/knowledge_adaptation/endata/atrophy/image', '/home/cuixiaoxiao/knowledge_adaptation/endata/atrophy/label','/home/cuixiaoxiao/knowledge_adaptation/endata/normal/image'],
                           list_file='valid.txt',input_size=224, state='Valid')


    data_loader_train = torch.utils.data.DataLoader(trainset, batch_size=24, shuffle=False, num_workers=4)
    data_loader_val = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=4)

    print(f"Creating model: {args.model}")

    model_params = dict(
        model_name=args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        #reduction=args.reduction,
        #pool=args.pool_type
    )
    if "AttnFeat" in args.model:
        model_params["feat_reduction"] = args.feat_reduction
    model = create_model(**model_params)

    if args.finetune and not args.gen_attention_maps:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        try:
            checkpoint_model = checkpoint['model']
        except:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        if args.extra_token:
            num_extra_tokens = 1
        else:
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)

        new_size = int(num_patches ** 0.5)

        if args.extra_token:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens].repeat(1, args.nb_classes, 1)
        else:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]

        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        if args.extra_token:
            cls_token_checkpoint = checkpoint_model['cls_token']
            new_cls_token = cls_token_checkpoint.repeat(1, args.nb_classes, 1)
            checkpoint_model['cls_token'] = new_cls_token

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    #mlflow.log_param("n_parameters", n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils_wt.get_world_size() / 512.0
    args.lr = linear_scaled_lr

    model_params = model.parameters()
    if args.weight_decay and (args.skiplist or args.filter_bias_and_bn):
        skip = {}
        if hasattr(model, 'no_weight_decay') and args.skiplist:
            skip = model.no_weight_decay()
        from utils_wt import add_weight_decay
        model_params = add_weight_decay(model, args.weight_decay, skip,
                                        args.filter_bias_and_bn)
        args.weight_decay = 0.

    optimizer = create_optimizer(args, model_params)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"mAP of the network on the {len(data_loader_val)} test images: {test_stats['mAP'] * 100:.1f}%")
        return

    if args.gen_attention_maps:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        if args.distributed:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        generate_attention_maps_ms(data_loader_train, model, device, args)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_max_mIoU = 0.0
    max_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.epoch_save_ckpt and os.path.exists(output_dir / f'checkpoint{epoch}.pth'):
            checkpoint = torch.load(output_dir / f'checkpoint{epoch}.pth', map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_stats = checkpoint["train_stats"]
        else:
            train_stats = train_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad
            )

        lr_scheduler.step(epoch)

        test_stats = evaluate(data_loader_val, model, device)
        if test_stats["mAP"]>=max_accuracy :
            max_accuracy =test_stats["mAP"]
            torch.save({'model': model.state_dict(), 'epoch': epoch}, output_dir / 'checkpoint_my21a2cfsc_best_segsss.pth')
    torch.save({'model': model.state_dict(), 'epoch': epoch}, output_dir / 'checkpoint_my21a6cfsc_segsss.pth')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

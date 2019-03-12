from common_utils import *
from network import *
from config import *
from dataset import *
from train import *
from arguments import *
def main():
    '''overwrite config parameters with argument parameters
    '''
    print(args)
    device = torch.device("cuda:1" if (torch.cuda.is_available() and args['device']=='gpu') else "cpu")
    criterion = nn.CrossEntropyLoss()
    #network=SimpleNetwork(args)
    network = LSTMNetwork(args)
    #network = VariableLSTMNetwork(args)
    # if device.type != 'cpu':
    #     network = nn.DataParallel(network)
    network.to(device)
    #optimizer = torch.optim.SGD(network.parameters(), lr=args['lr'])#,momentum=args['momentum'],weight_decay=args['wd'])
    #optimizer = torch.optim.Adam(network.parameters(), lr=args['lr'], amsgrad=True)#,momentum=args['momentum'],weight_decay=args['wd'])
    optimizer = torch.optim.Adam(network.parameters(), lr=args['lr'])#,momentum=args['momentum'],weight_decay=args['wd'])
    scheduler = StepLR(optimizer,step_size=100000,gamma=0.1)    
    dataset = LSTMXORDataset(args)
    #dataset = VariableLSTMXORDataset(args)
    dataloader = DataLoader(dataset,batch_size=args['bs'],shuffle=True,num_workers=8)
    for epoch in range(100000):
        scheduler.step()
        train_LSTM(dataloader,network,criterion,optimizer,epoch,device)
        #print(scheduler.get_lr())
    
main()
#     args = parser.parse_args()

#     if args.seed is not None:
#         random.seed(args.seed)
#         torch.manual_seed(args.seed)
#         cudnn.deterministic = True
#         warnings.warn('You have chosen to seed training. '
#                       'This will turn on the CUDNN deterministic setting, '
#                       'which can slow down your training considerably! '
#                       'You may see unexpected behavior when restarting '
#                       'from checkpoints.')

#     if args.gpu is not None:
#         warnings.warn('You have chosen a specific GPU. This will completely '
#                       'disable data parallelism.')

#     if args.dist_url == "env://" and args.world_size == -1:
#        args.world_size = int(os.environ["WORLD_SIZE"])

#     args.distributed = args.world_size > 1 or args.multiprocessing_distributed

#     ngpus_per_node = torch.cuda.device_count()
#     if args.multiprocessing_distributed:
#         # Since we have ngpus_per_node processes per node, the total world_size
#         # needs to be adjusted accordingly
#         args.world_size = ngpus_per_node * args.world_size
#         # Use torch.multiprocessing.spawn to launch distributed processes: the
#         # main_worker process function
#         mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
#     else:
#         # Simply call main_worker function
#         main_worker(args.gpu, ngpus_per_node, args)

# def main_worker(gpu, ngpus_per_node, args):
#         global best_acc1
#         args.gpu = gpu

#         if args.gpu is not None:
#             print("Use GPU: {} for training".format(args.gpu))

#         if args.distributed:
#             if args.dist_url == "env://" and args.rank == -1:
#                 args.rank = int(os.environ["RANK"])
#             if args.multiprocessing_distributed:
#                 # For multiprocessing distributed training, rank needs to be the
#                 # global rank among all the processes
#                 args.rank = args.rank * ngpus_per_node + gpu
#             dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
#                                     world_size=args.world_size, rank=args.rank)
#         # create model
#         if args.pretrained:
#             print("=> using pre-trained model '{}'".format(args.arch))
#             model = models.__dict__[args.arch](pretrained=True)
#         else:
#             print("=> creating model '{}'".format(args.arch))
#             model = models.__dict__[args.arch]()

#         if args.distributed:
#             # For multiprocessing distributed, DistributedDataParallel constructor
#             # should always set the single device scope, otherwise,
#             # DistributedDataParallel will use all available devices.
#             if args.gpu is not None:
#                 torch.cuda.set_device(args.gpu)
#                 model.cuda(args.gpu)
#                 # When using a single GPU per process and per
#                 # DistributedDataParallel, we need to divide the batch size
#                 # ourselves based on the total number of GPUs we have
#                 args.batch_size = int(args.batch_size / ngpus_per_node)
#                 args.workers = int(args.workers / ngpus_per_node)
#                 model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
#             else:
#                 model.cuda()
#                 # DistributedDataParallel will divide and allocate batch_size to all
#                 # available GPUs if device_ids are not set
#                 model = torch.nn.parallel.DistributedDataParallel(model)
#         elif args.gpu is not None:
#             torch.cuda.set_device(args.gpu)
#             model = model.cuda(args.gpu)
#         else:
#             # DataParallel will divide and allocate batch_size to all available GPUs
#             if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
#                 model.features = torch.nn.DataParallel(model.features)
#                 model.cuda()
#             else:
#                 model = torch.nn.DataParallel(model).cuda()

#         # define loss function (criterion) and optimizer
#         criterion = nn.CrossEntropyLoss().cuda(args.gpu)

#         optimizer = torch.optim.SGD(model.parameters(), args.lr,
#                                     momentum=args.momentum,
#                                     weight_decay=args.weight_decay)

#         # optionally resume from a checkpoint
#         if args.resume:
#             if os.path.isfile(args.resume):
#                 print("=> loading checkpoint '{}'".format(args.resume))
#                 checkpoint = torch.load(args.resume)
#                 args.start_epoch = checkpoint['epoch']
#                 best_acc1 = checkpoint['best_acc1']
#                 model.load_state_dict(checkpoint['state_dict'])
#                 optimizer.load_state_dict(checkpoint['optimizer'])
#                 print("=> loaded checkpoint '{}' (epoch {})"
#                     .format(args.resume, checkpoint['epoch']))
#             else:
#                 print("=> no checkpoint found at '{}'".format(args.resume))

#         cudnn.benchmark = True

#         # Data loading code
#         traindir = os.path.join(args.data, 'train')
#         valdir = os.path.join(args.data, 'val')
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                         std=[0.229, 0.224, 0.225])

#         train_dataset = datasets.ImageFolder(
#             traindir,
#             transforms.Compose([
#                 transforms.RandomResizedCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 normalize,
#             ]))

#         if args.distributed:
#             train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
#         else:
#             train_sampler = None

#         train_loader = torch.utils.data.DataLoader(
#             train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
#             num_workers=args.workers, pin_memory=True, sampler=train_sampler)

#         val_loader = torch.utils.data.DataLoader(
#             datasets.ImageFolder(valdir, transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 normalize,
#             ])),
#             batch_size=args.batch_size, shuffle=False,
#             num_workers=args.workers, pin_memory=True)

#         if args.evaluate:
#             validate(val_loader, model, criterion, args)
#             return

#         for epoch in range(args.start_epoch, args.epochs):
#             if args.distributed:
#                 train_sampler.set_epoch(epoch)
#             adjust_learning_rate(optimizer, epoch, args)

#             # train for one epoch
#             train(train_loader, model, criterion, optimizer, epoch, args)

#             # evaluate on validation set
#             acc1 = validate(val_loader, model, criterion, args)

#             # remember best acc@1 and save checkpoint
#             is_best = acc1 > best_acc1
#             best_acc1 = max(acc1, best_acc1)

#             if not args.multiprocessing_distributed or (args.multiprocessing_distributed
#                     and args.rank % ngpus_per_node == 0):
#                 save_checkpoint({
#                     'epoch': epoch + 1,
#                     'arch': args.arch,
#                     'state_dict': model.state_dict(),
#                     'best_acc1': best_acc1,
#                     'optimizer' : optimizer.state_dict(),
#                 }, is_best)
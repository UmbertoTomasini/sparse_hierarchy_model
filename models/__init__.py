import torch
import torch.backends.cudnn as cudnn

from .fcn import DenseNet
from .cnn import ConvNetGAPMF # ConvNet2L
from .lcn import LocallyHierarchicalNet
from .cnn2 import CNN2
from .fcn2 import FCN2
from .gcnn import GCNN
from .lcn_shared import LocallyHierarchicalNetShared
from .cnn2_shared import CNN2_shared
from .efficientnet import *
from .vgg import *
from .resnet import *


def model_initialization(args, input_dim, ch,s):
    """
    Neural netowrk initialization.
    :param args: parser arguments
    :return: neural network as torch.nn.Module
    """
    #for new architectures
    image_size = (s*(args.s0+1))**args.net_layers
    num_classes = args.num_classes
    
    num_outputs = 1 if args.loss == "hinge" else args.num_classes

    ### Define network architecture ###
    torch.manual_seed(args.seed_net)

    net = None

    ### Define network architecture ###
    if args.seed_net != -1:
        torch.manual_seed(args.seed_net)
    net = None
    #if not args.pretrained: # and not args.scattering_mode
        
    if args.net == "fcn":
        net = DenseNet(
            n_layers=args.net_layers,
            input_dim=input_dim * ch,
            h=args.width,
            out_dim=num_outputs,
            batch_norm=args.batch_norm,
            # bias=args.bias,
        )
    elif args.net == "cnn":
        net = ConvNetGAPMF(
            n_blocks=args.net_layers,
            input_ch=ch,
            h=args.width,
            filter_size=args.filter_size,
            stride=args.stride,
            pbc=args.pbc,
            out_dim=num_outputs,
            batch_norm=args.batch_norm,
        )

    ### The next 4 architectures are built to have the same *effective* number of parameters ###
    elif args.net == "lcn":
        net = LocallyHierarchicalNet(
            num_layers=args.net_layers,
            input_channels=ch,
            h=args.width,
            # filter_size=args.filter_size,
            out_dim=num_outputs,
            s = args.s,
            s0 = args.s0,
            bias=args.bias,
        )
    elif args.net == "lcn_shared":
        net = LocallyHierarchicalNetShared(
            num_layers=args.net_layers,
            input_channels=ch,
            h=args.width,
            # filter_size=args.filter_size,
            out_dim=num_outputs,
            s = args.s,
            s0 = args.s0,
            sharing = args.sharing, 
            bias=args.bias,
        )
    elif (args.net == "cnn2" or args.net =="cnn2-hom"):
        net = CNN2(
            num_layers=args.net_layers,
            input_channels=ch,
            h=args.width,
            # filter_size=args.filter_size,
            out_dim=num_outputs,
            s0 = args.s0,
            s = s,
            bias=args.bias,
        )
    elif args.net == "cnn2_shared":
        net = CNN2_shared(
            num_layers=args.net_layers,
            input_channels=ch,
            h=args.width,
            # filter_size=args.filter_size,
            out_dim=num_outputs,
            s0 = args.s0,
            s = s,
            sharings = args.sharings, 
            bias=args.bias,
        )
    elif args.net == "fcn2":
        net = FCN2(
            num_layers=args.net_layers,
            input_channels=ch,
            h=args.width,
            out_dim=num_outputs,
            s = args.s,
            s0 = args.s0,
            bias=args.bias,
        )
    elif args.net == "gcnn":
        net = GCNN(
            num_layers=args.net_layers,
            input_channels=ch,
            h=args.width,
            out_dim=num_outputs,
            bias=args.bias,
        )

    #NEW ONES
    if 'VGG' in args.net:
        if 'bn' in args.net:
            bn = True
            net_name = args.net[:-2]
        else:
            bn = False
            net_name = args.net
        net = VGG1D(
            net_name,
            num_ch= ch,
            num_classes=num_classes,
            pooling="max")
            #batch_norm=bn,
            #param_list=args.param_list,
             #args.pooling)
            #width_factor=args.width_factor,
            #stride=args.stride)

    if args.net == 'AlexNet':
        net = AlexNet(num_ch=num_ch, num_classes=num_classes)
    if args.net == 'ResNet18':
        net = ResNet18(num_ch=ch, num_classes=num_classes)
    if args.net == 'ResNet34':
        net = ResNet34(num_ch=ch, num_classes=num_classes)
    if args.net == 'ResNet50':
        net = ResNet50(num_ch=ch, num_classes=num_classes)
    if args.net == 'ResNet101':
        net = ResNet101(num_ch=ch, num_classes=num_classes)
    if args.net == 'LeNet':
        net = LeNet(num_ch=num_ch, num_classes=num_classes, stride=args.stride)
    if args.net == 'GoogLeNet':
        net = GoogLeNet(num_ch=num_ch, num_classes=num_classes)
    if args.net == 'MobileNetV2':
        net = MobileNetV2(num_ch=num_ch, num_classes=num_classes)
    if args.net == 'DenseNet121':
        net = DenseNet121(num_ch=num_ch, num_classes=num_classes)
    if args.net == 'EfficientNetB0':
        net = EfficientNetB0(num_ch=ch, num_classes=num_classes)
    if args.net == 'ConvNextT':
        net = ConvNextT(num_ch=num_ch, num_classes=num_classes)
    if args.net == 'MinCNN':
        net = MinCNN(num_ch=num_ch, num_classes=num_classes, h=args.width, fs=args.filter_size, ps=args.pooling_size, param_list=args.param_list)
    if args.net == 'LCN':
        net = MinCNN(num_ch=num_ch, num_classes=num_classes, h=args.width, fs=args.filter_size, ps=args.pooling_size)
    if args.net == 'DenseNetL2':
        net = DenseNetL2(num_ch=num_ch * image_size ** 2, num_classes=num_classes, h=args.width)
    if args.net == 'DenseNetL4':
        net = DenseNetL4(num_ch=num_ch * image_size ** 2, num_classes=num_classes, h=args.width)
    if args.net == 'DenseNetL6':
        net = DenseNetL6(num_ch=num_ch * image_size ** 2, num_classes=num_classes, h=args.width)
    if args.net == 'ConvGAP':
        net = ConvNetGAPMF(n_blocks=args.depth, input_ch=num_ch, h=args.width, filter_size=args.filter_size,
                           stride=args.stride, pbc=args.pbc, out_dim=num_classes, batch_norm=args.batch_norm)

    if args.net == 'ScatteringLinear':
        net = ScatteringLinear(n=image_size, ch=num_ch, J=args.J, L=args.L, num_classes=num_classes)
    assert net is not None, 'Network architecture not in the list!'
    net = net.to(args.device)
    
  
        
    assert net is not None, "Network architecture not in the list!"

    if args.random_features:
        for param in [p for p in net.parameters()][:-2]:
            param.requires_grad = False

    net = net.to(args.device)

    if args.device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    return net

from architectures import get_architecture, IMAGENET_CLASSIFIERS, AUTOENCODER_ARCHITECTURES, DENOISERS_ARCHITECTURES
from core import Smooth
from datasets import get_dataset, DATASETS, get_num_classes
from time import time

import argparse
import datetime
import os
import torch
from architectures import CNNModel
from datasets import Medical_Classification
from torch.utils.data import DataLoader
from robustness import datasets as dataset_r
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy
import foolbox
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Certify many examples')
# parser.add_argument("--dataset", choices=DATASETS, help="which dataset")
parser.add_argument("--dataset", help="which dataset")
parser.add_argument("--attack_type", type=str, help="attack type", choices=['FGSM', 'PGD', 'DeepFool'])
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--philly_imagenet_path', type=str, default='',
                    help='Path to imagenet on philly')
parser.add_argument('--azure_datastore_path', type=str, default='',
                    help='Path to imagenet on azure')

parser.add_argument('--l2radius', type=float, help='l2 radius')

# Model Arch & Checkpoint
parser.add_argument('--model_type', default='DS', type=str,
                    help="Denoiser + (AutoEncoder) + classifier/reconstructor",
                    choices=['DS', 'AE_DS'])
parser.add_argument("--base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument('--pretrained-denoiser', type=str, default='',
                    help='Path to a denoiser to attached before classifier during certificaiton.')

parser.add_argument('--pretrained-encoder', default='', type=str,
                    help='path to a pretrained encoder')
parser.add_argument('--pretrained-decoder', default='', type=str,
                    help='path to a pretrained decoder')

parser.add_argument('--encoder_arch', type=str, default='cifar_encoder', choices=AUTOENCODER_ARCHITECTURES)
parser.add_argument('--decoder_arch', type=str, default='cifar_decoder', choices=AUTOENCODER_ARCHITECTURES)
parser.add_argument('--arch', type=str, choices=DENOISERS_ARCHITECTURES)

parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

args = parser.parse_args()

if __name__ == "__main__":

    # --------------------- Dataset Loading ----------------------
    if args.dataset == 'cifar10' or args.dataset == 'stl10' or args.dataset == 'mnist':
        pin_memory = (args.dataset == "imagenet")
        test_dataset = get_dataset(args.dataset, 'test')
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1,
                                 num_workers=args.workers, pin_memory=pin_memory)

    elif args.dataset == 'restricted_imagenet':
        in_path = '/localscratch2/damondemon/datasets/imagenet'
        in_info_path = '/localscratch2/damondemon/datasets/imagenet_info'
        in_hier = ImageNetHierarchy(in_path, in_info_path)

        superclass_wnid = ['n02084071', 'n02120997', 'n01639765', 'n01662784', 'n02401031', 'n02131653', 'n02484322',
                           'n01976957', 'n02159955', 'n01482330']

        class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)
        custom_dataset = dataset_r.CustomImageNet(in_path, class_ranges)
        _, test_loader = custom_dataset.make_loaders(workers=4, batch_size=1)
    
    elif 'medical' in args.dataset.lower():
        if 'alzheimer' in args.dataset.lower():
            data_root = '/home/chaunm/Projects/medical-adversarial-defense/defense/datasets/Alzheimer/'
        elif 'chestcancer' in args.dataset.lower():
            data_root = '/home/chaunm/Projects/medical-adversarial-defense/defense/datasets/ChestCancer/'
        elif 'kidneycancer' in args.dataset.lower():
            data_root = '/home/chaunm/Projects/medical-adversarial-defense/defense/datasets/KidneyCancer/'
        IMG_SIZE = 64

        # trainset = Origdataset(args, datasets_root_dir,  split="train",  transform=transform_train)
        trainset = Medical_Classification(data_root + 'train', IMG_SIZE)

        # testset = Origdataset(args, datasets_root_dir,  split="val",  transform=transform_test)
        testset = Medical_Classification(data_root + 'val', IMG_SIZE)

        train_loader = DataLoader(trainset, shuffle=True, batch_size=args.batch,
                                  num_workers=args.workers)
        test_loader = DataLoader(testset, shuffle=False, batch_size=args.batch,
                                 num_workers=args.workers)

    # --------------------- Model Loading -------------------------
    # a) Classifier / Reconstructor
    checkpoint = torch.load(args.base_classifier)
    # base_classifier = get_architecture(checkpoint['arch'], args.dataset)
    # base_classifier.load_state_dict(checkpoint['state_dict'])
    num_classes = len(trainset.get_classes())
    base_classifier = CNNModel(num_classes).cuda()
    base_classifier.load_state_dict(checkpoint)

    # b) Denoiser

    # if args.pretrained_denoiser:
    #     checkpoint = torch.load(args.pretrained_denoiser)
    #     assert checkpoint['arch'] == args.arch
    #     denoiser = get_architecture(checkpoint['arch'], args.dataset)
    #     denoiser.load_state_dict(checkpoint['state_dict'])
    # else:
    #     denoiser = get_architecture(args.arch, args.dataset)

    if args.pretrained_denoiser:
        checkpoint = torch.load(args.pretrained_denoiser)
        assert args.arch == args.arch
        denoiser = get_architecture(args.arch, args.dataset)
        denoiser.load_state_dict(checkpoint['state_dict'])
    else:
        denoiser = get_architecture(args.arch, args.dataset)

    # c) AutoEncoder
    if args.model_type == 'AE_DS':
        checkpoint = torch.load(args.pretrained_encoder)
        assert checkpoint['arch'] == args.encoder_arch
        encoder = get_architecture(checkpoint['arch'], args.dataset)
        encoder.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load(args.pretrained_decoder)
        assert checkpoint['arch'] == args.decoder_arch
        decoder = get_architecture(checkpoint['arch'], args.dataset)
        decoder.load_state_dict(checkpoint['state_dict'])

        base_classifier = torch.nn.Sequential(denoiser, encoder, decoder, base_classifier)

    else:
        base_classifier = torch.nn.Sequential(denoiser, base_classifier)

    base_classifier = base_classifier.eval().cuda()

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # ------------------ Adversarial Attacks ----------------------
    class FoolBoxLib():
        def __init__(self, model, attack_type):
            self.fmodel = foolbox.PyTorchModel(model, bounds=(0, 1), device='cuda')
            self.epsilons = 8/255

            print('Creating', attack_type, 'method.')
            if attack_type == 'FGSM':
                self.attack = foolbox.attacks.LinfFastGradientAttack()
            elif attack_type == 'PGD':
                self.attack = foolbox.attacks.LinfPGD()
            elif attack_type == 'DeepFool':
                self.attack = foolbox.attacks.LinfDeepFoolAttack()
            

        def __call__(self, images, targets):
            # images = images.unsqueeze(0)
            # targets = targets.unsqueeze(0)
            _, advs, is_adv = self.attack(self.fmodel, images, targets, epsilons=self.epsilons)
            return advs

    fb = FoolBoxLib(base_classifier, args.attack_type)

    advs_images_lst = []
    advs_labels_lst = []
    print('Poisoining images')
    for imgs, targets in tqdm(train_loader):
        imgs = imgs.cuda()
        targets = targets.cuda()
        adv_images = fb(imgs, targets)
        advs_images_lst.append(adv_images)
        advs_labels_lst.append(targets)
    advs_images_lst = torch.concat(advs_images_lst, dim=0)
    advs_labels_lst = torch.concat(advs_labels_lst, dim=0)
    train_dataset = torch.utils.data.TensorDataset(advs_images_lst, advs_labels_lst)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch)

    advs_images_lst = []
    advs_labels_lst = []
    for imgs, targets in tqdm(test_loader):
        imgs = imgs.cuda()
        targets = targets.cuda()
        adv_images = fb(imgs, targets)
        advs_images_lst.append(adv_images)
        advs_labels_lst.append(targets)
    advs_images_lst = torch.concat(advs_images_lst, dim=0)
    advs_labels_lst = torch.concat(advs_labels_lst, dim=0)
    test_dataset = torch.utils.data.TensorDataset(advs_images_lst, advs_labels_lst)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch)

    # prepare output file
    if not os.path.exists(args.outfile.split('sigma')[0]):
        os.makedirs(args.outfile.split('sigma')[0])

    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tSta_correct\ttime\tcount\tSta_count", file=f, flush=True)
    print("idx\tlabel\tpredict\tradius\tSta_correct\ttime\tcount\tSta_count", flush=True)
    f.close()

    # iterate through the dataset
    count = 0
    sta_count = 0
    for i, (x, label) in enumerate(test_loader):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        # prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)

        prediction = base_classifier(x).argmax(1)
        radius = 0
        
        after_time = time()
        # correct = int(prediction == label)
        
        # correct = int(prediction == label and radius > args.l2radius)
        # sta_correct = int(prediction == label)

        correct = int(sum(prediction == label))
        sta_correct = int(sum(prediction == label))

        count += correct
        sta_count += sta_correct

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

        f = open(args.outfile, 'a')
        print("{}\t{}\t{}\t{}".format(
            i, time_elapsed, count, sta_count), file=f, flush=True)
        print("{}\t{}\t{}\t{}".format(
            i, time_elapsed, count, sta_count), flush=True)
        # print("{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{}".format(
        #     i, label, prediction, radius, sta_correct, time_elapsed, count, sta_count), file=f, flush=True)
        # print("{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{}".format(
        #     i, label, prediction, radius, sta_correct, time_elapsed, count, sta_count), flush=True)
        f.close()
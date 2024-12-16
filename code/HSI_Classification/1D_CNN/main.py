import argparse
import os
import tensorflow as tf

from data_loader import Data
from model import Model

from tensorflow.python.keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='1D CNN for hyperspectral image classification')

parser.add_argument('--result',dest='result',default='result')
parser.add_argument('--log',dest='log',default='log')
parser.add_argument('--model',dest='model',default='model')
parser.add_argument('--tfrecords',dest='tfrecords',default='tfrecords')
parser.add_argument('--data_name',dest='data_name',default='WHU_Hi_HanChuan')#dftc
parser.add_argument('--data_path',dest='data_path',default="/home/storage/Dataset/WHU-Hi-HanChuan")

parser.add_argument('--use_lr_decay',dest='use_lr_decay',default=True)
parser.add_argument('--decay_rete',dest='decay_rete',default=0.90)
parser.add_argument('--decay_steps',dest='decay_steps',default=10000)
parser.add_argument('--learning_rate',dest='lr',default=0.001)
parser.add_argument('--train_num',dest='train_num',default=200) # intger for number and decimal for percentage
parser.add_argument('--batch_size',dest='batch_size',default=100)
parser.add_argument('--fix_seed',dest='fix_seed',default=False)
parser.add_argument('--seed',dest='seed',default=666)
parser.add_argument('--test_batch',dest='test_batch',default=5000)
parser.add_argument('--iter_num',dest='iter_num',default=100001)
parser.add_argument('--cube_size',dest='cube_size',default=1)
parser.add_argument('--save_decode_map',dest='save_decode_map',default=True)
parser.add_argument('--save_decode_seg_map',dest='save_decode_seg_map',default=True)
parser.add_argument('--load_model',dest='load_model',default=False)


args = parser.parse_args()
if not os.path.exists(args.model):
    os.mkdir(args.model)
if not os.path.exists(args.log):
    os.mkdir(args.log)
if not os.path.exists(args.result):
    os.mkdir(args.result)
if not os.path.exists(args.tfrecords):
    os.mkdir(args.tfrecords)


def main():

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    for i in range(1):
        args.id = str(i)
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session(config=config) as sess:
            args.result = os.path.join(args.result,args.id)
            args.log = os.path.join(args.log, args.id)
            args.model = os.path.join(args.model, args.id)
            args.tfrecords = os.path.join(args.tfrecords, args.id)
            if not os.path.exists(args.model):
                os.mkdir(args.model)
            if not os.path.exists(args.log):
                os.mkdir(args.log)
            if not os.path.exists(args.result):
                os.mkdir(args.result)
            if not os.path.exists(args.tfrecords):
                os.mkdir(args.tfrecords)
            dataset = Data(args)
            dataset.read_data()
            model = Model(args,sess)
            if not args.load_model:
                model.train(dataset)
            else:
                model.load(args.model)
            model.test(dataset)
            if args.save_decode_map:
                model.save_decode_map(dataset)
            if args.save_decode_seg_map:
                model.save_decode_seg_map(dataset)
            args.result = 'result'
            args.log = 'log'
            args.tfrecords = 'tfrecords'
            args.model = 'model'



if __name__ == '__main__':
    main()
"""
ttest end!
1 class: ( 151593 / 371578 ) 0.40797087018068884
2 class: ( 168521 / 292996 ) 0.575164848666876
3 class: ( 147460 / 184056 ) 0.801169209371061
4 class: ( 19139 / 70399 ) 0.27186465716842567
5 class: ( 198133 / 259374 ) 0.7638892101752681
6 class: ( 8458 / 10802 ) 0.7830031475652657
7 class: ( 9483 / 11651 ) 0.8139215517981289
confusion matrix:
[[151593  21892   5110  15661  11014   1004    590]
 [ 35599 168521  19513  11867  12254    340    168]
 [ 14744  31464 147460   3985   3278    107     26]
 [ 75777  43836   7598  19139  22578    563    559]
 [ 22431  12558   3198   8928 198133    133    617]
 [ 38205  10205    954   4564    947   8458    208]
 [ 33229   4520    223   6255  11170    197   9483]]
total right num: 702787
oa: 0.5852383633008454
aa: 0.6309976421322449
kappa: 0.49236242536277736
Groundtruth map get finished
test end!
decode map get finished
test end!
seg decode map get finished



"""
import argparse
import torchtext.data as data
from torchtext.vocab import Vectors
import getmydata
import getrawData
import CNN
import torch
import datetime
import os
import train
import jieba

# ********************************************************************* #

trainDataPath = "./data/Training data for Emotion Classification.xml"
testDataPath = "./data/EmotionClassficationTest.xml"

# ********************************************************************* #


dataroot = './data'


parse = argparse.ArgumentParser(description="text classificer")

#training hyparemeter

parse.add_argument('-lr',type=float,default=0.001,help="initial learning rate [default:0.001]")
parse.add_argument('-epochs',type=int,default=256,help="numbers of epochs for train [default:256]")
parse.add_argument('-batch-size',type=int,default=64,help="batch size for training")
parse.add_argument('-log-interval',type=int,default=1,help="how many steps to wait before logging training status")
parse.add_argument('-test-interval',type=int,default=100,help="how many steps to wait before testing")
parse.add_argument('-save-interval',type=int,default=500,help="how many steps to wait before saving")
parse.add_argument('-save-dir',type=str,default='snapshot',help="where to save snapshot")
parse.add_argument('-early-stop',type=int,default=1000,help="iteration numbers to stop without performance increasing")
parse.add_argument('-save-best',type=bool,default=True,help="wether to save when get best performance")
parse.add_argument('-loadPretrainedVector',type=bool,default=True,help="wether to load your own vector")
parse.add_argument('-wordvector',type=str,default='D:/code/NLPdataset/wordvector/sgns.weibo.word',help="if you choose to load your own wordvector,this is its path")




#data
parse.add_argument('-shuffle',action='store_true',default=True,help="shuffle the data every epochs")

#model
parse.add_argument('-dropout',type=float,default=0.5,help="the probability for dropout")
parse.add_argument('-max-norm',type=float,default=3.0,help="l2 constraint of parameters")
parse.add_argument('-embed-dim',type=int,default=300,help="number of embedding dimension")
parse.add_argument('-static',action='store_true',help="fix the embedding")

##### CNN #####
parse.add_argument('-kernel-num',type=int,default=100,help="number of each kind of kernel")
parse.add_argument('-kernel-sizes',type=str,default='3,4,5',help='comma-separated kernel size to use for convolution')


#device
parse.add_argument('-device',type=int,default=-1,help="device to use for iterate data, -1 mean cpu")
parse.add_argument('-no-cuda',action='store_true',default=False,help="disable the gpu")


#option
parse.add_argument('-snapshot',type=str,default=None,help='filename of model snapchat')
parse.add_argument('-predict',type=str,default=None,help="predict the sentence given")
parse.add_argument('-test',action='store_true',default=False,help="train or test")

args = parse.parse_args()


#load own dataset
def weibo(text_field, label_field,rawdata, **kargs):
    train, dev, test = getmydata.mydata.getdataSplit(text_field, label_field, rawdata)
    ###使用自己的词向量###
    
    vector = Vectors(name='./vector_cache/sgns.weibo.word')

    text_field.build_vocab(train, dev, test, vectors=vector)

    label_field.build_vocab(train, dev, test)

    word_embedding = text_field.vocab.vectors##在此函数中返回词向量值

    print("the embedding",word_embedding)

    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
                                                                 batch_sizes=(args.batch_size, len(dev), len(test)),
                                                                 **kargs)

    return train_iter, dev_iter, test_iter, word_embedding

def tokenizer(text):
    return list(jieba.cut(text))


# load data
print("\nloading data......")

text_field = data.Field(lower=True, tokenize=tokenizer, fix_length=120)
label_field = data.Field(sequential=False)

traindata = getrawData.getRaw(trainDataPath)
testdata = getrawData.getRaw(testDataPath)
rawdata = traindata + testdata


train_iter, dev_iter, test_iter, word_embedding = weibo(text_field,label_field,rawdata,device=-1,repeat=False)

#print("vocabulary",text_field.vocab.stoi)


# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab)-1
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir,datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
args.word_embedding = word_embedding


print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


#model
cnn = CNN.CNN(args, text_field)
if args.snapshot is not None:
    print('\nLoading model from {}'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()

if args.predict is not None:
    label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.eval(test_iter, cnn, args)
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    try:
        train.train(train_iter, dev_iter, cnn, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')

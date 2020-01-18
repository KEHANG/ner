import os
import argparse
from torch import optim
from datetime import datetime

import tagger.loader
import tagger.models.crf

parser = argparse.ArgumentParser()

parser.add_argument('--dset_dir', default='../../data', type=str,
                    help='dataset filename')
parser.add_argument('--dset_file', default='mini_train.txt', type=str,
                    help='dataset filename for train')
parser.add_argument('--dset_file_dev', default='mini_dev.txt', type=str,
                    help='dataset filename for dev')
parser.add_argument('--dset_file_test', default='mini_test.txt', type=str,
                    help='dataset filename for test')
parser.add_argument('--batch_size', default=1, type=int,
                    help='batch size')
parser.add_argument('--num_workers', default=1, type=int,
                    help='_ num_workers')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of epochs')
parser.add_argument('--lr', default=0.01, type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay rate')
parser.add_argument('--print_period', default=3600, type=float,
                    help='print period')
parser.add_argument('--embedding_dim', default=10, type=int,
                    help='embedding dimension of each word.')
parser.add_argument('--hidden_dim', default=8, type=int,
                    help='hiddlen dimension.')

args = parser.parse_args()

def main(args):

    # create work folder
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    work_folder = 'model-save-{0}'.format(now)
    os.mkdir(work_folder)

    # load data
    data = tagger.loader.return_data(args.dset_dir,
                                     args.dset_file,
                                     args.batch_size,
                                     args.num_workers,
                                     args.dset_file_dev,
                                     args.dset_file_test)
    dataloader, dataloader_dev, word_to_ix, tag_to_ix = data

    # build model
    net = tagger.models.crf.BiLSTM_CRF(vocab_size=len(word_to_ix),
                                       embedding_dim=args.embedding_dim,
                                       hidden_dim=args.hidden_dim,
                                       lstm_num_layers=1,
                                       tag_to_ix=tag_to_ix)

    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          weight_decay=args.weight_decay)

    # start training
    best_f1 = 0.0
    for epoch in range(args.epochs):
        # train step
        train_loss, nb_tr_steps = net.train_one_epoch(dataloader, optimizer)
        print("Epoch={0} Train loss: {1}".format(epoch+1, train_loss/nb_tr_steps))

        # evaluation step
        f1 = net.f1_eval(dataloader_dev)
        print('Epoch={0} Validation F1: {1:.3f}'.format(epoch+1, f1))

        if f1 > best_f1:
          best_f1 = f1
          output_dir = os.path.join(work_folder, "epoch-{0:05d}".format(epoch+1))
          net.save(output_dir)

if __name__ == '__main__':
    main(args)

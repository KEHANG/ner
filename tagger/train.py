import os
import argparse
import torch
from torch import optim
from datetime import datetime

import tagger.model
import tagger.loader 
from tagger.constant import PAD

parser = argparse.ArgumentParser()

parser.add_argument('--dset_dir', default='../data', type=str,
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
parser.add_argument('--epochs', default=1000, type=int,
                    help='number of epochs')
parser.add_argument('--lr', default=0.01, type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay rate')
parser.add_argument('--print_period', default=2, type=float,
                    help='print period')
parser.add_argument('--embedding_dim', default=10, type=int,
                    help='embedding dimension of each word.')
parser.add_argument('--hidden_dim', default=8, type=int,
                    help='hiddlen dimension of BiLSTM.')

args = parser.parse_args()

def match_eval(tag_seqs_pred, tag_seqs_true, tag_to_ix):

    assert(tag_seqs_pred.shape==tag_seqs_true.shape)

    pos_pred_num = torch.sum((~torch.eq(tag_seqs_pred, tag_to_ix['O'])) & 
                             (~torch.eq(tag_seqs_pred, tag_to_ix[PAD])))

    pos_true_num = torch.sum((~torch.eq(tag_seqs_true, tag_to_ix['O'])) & 
                             (~torch.eq(tag_seqs_true, tag_to_ix[PAD])))

    pos_match_num = torch.sum((torch.eq(tag_seqs_pred, tag_seqs_true)) & 
                              (~torch.eq(tag_seqs_true, tag_to_ix['O'])) &
                              (~torch.eq(tag_seqs_true, tag_to_ix[PAD])))

    return pos_pred_num.item(), pos_true_num.item(), pos_match_num.item()

def dev_eval(net, dataloader_dev):

  pred_sum, true_sum, match_sum = [0, 0, 0]
  for dev_batch in dataloader_dev:
      sentences, lengths, tag_seqs = dev_batch
      tag_seqs_pred= net.predict(sentences, lengths)
      pred_num, true_num, match_num = match_eval(tag_seqs_pred, 
                                                 tag_seqs,
                                                 net.tag_to_ix)

      pred_sum += pred_num
      true_sum += true_num
      match_sum += match_num

  if match_sum == 0:
      f1 = 0.0
  else:
      precision = match_sum/1./(pred_sum)
      recall = match_sum/1./true_sum
      f1 = 2*precision*recall/(precision+recall)
  return f1

def main(args):

    # create work folder
    work_folder = 'training_folder-{0}'.format(datetime.now())
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
    net = tagger.model.BiLSTM(len(word_to_ix), 
                                  tag_to_ix, 
                                  args.embedding_dim, 
                                  args.hidden_dim,
                                  batch_size=args.batch_size)
    
    optimizer = optim.SGD(net.parameters(), 
                          lr=args.lr, 
                          weight_decay=args.weight_decay)
    
    # start training
    best_f1 = 0.0
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            net.zero_grad()

            # Step 2. Run our forward pass.
            sentences, lengths, tag_seqs = batch
            loss = torch.mean(net.neg_log_likelihood(sentences, lengths, tag_seqs))

            # Step 3. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()   

            # Print statistics
            running_loss += loss.item()
            if i % args.print_period == (args.print_period-1):
                f1 = dev_eval(net, dataloader_dev)
                print('[epoch=%d, batches=%d] train-loss: %.3f dev-f1: %.3f' %
                      (epoch + 1, i + 1, running_loss/(i + 1), f1))

                if f1 > best_f1:
                  best_f1 = f1
                  output_dir = os.path.join(work_folder, "epoch-{0}-batch-{1}".format(epoch+1, i+1))
                  net.save(output_dir)

if __name__ == '__main__':
    main(args)

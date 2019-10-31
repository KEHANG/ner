import os
import argparse
import torch
from torch import optim
from datetime import datetime
from tqdm import trange
from pytorch_pretrained_bert import BertTokenizer

import tagger.loader
import tagger.models.bert

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
parser.add_argument('--epochs', default=1, type=int,
                    help='number of epochs')
parser.add_argument('--lr', default=3e-5, type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=1e-2, type=float,
                    help='weight decay rate')

args = parser.parse_args()

def main(args):

    # create workspace folder
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    work_folder = 'model-save-{0}'.format(now)
    os.mkdir(work_folder)

    # load train and dev data
    train_dataset = tagger.loader.NERDataset(root=args.dset_dir, filename=args.dset_file, word_to_ix=None, tag_to_ix=None)
    dev_dataset = tagger.loader.NERDataset(root=args.dset_dir, filename=args.dset_file_dev, word_to_ix=None, tag_to_ix=None)
    tag_to_ix = {'[PAD]': 0,
                 'B-ORG': 1,
                 'O': 2,
                 'B-MISC': 3,
                 'B-PER': 4,
                 'I-PER': 5,
                 'B-LOC': 6,
                 'I-ORG': 7,
                 'I-MISC': 8,
                 'I-LOC': 9,
                 'X': 10}

    # create training/validation-friendly dataloaders
    bert_model = 'bert-base-cased'
    do_lower_case = False
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
    
    train_dataloader = tagger.loader.prepare_dataloader(train_dataset,
                                                        tokenizer,
                                                        tag_to_ix,
                                                        args.batch_size,
                                                        mode='train')

    dev_dataloader = tagger.loader.prepare_dataloader(dev_dataset,
                                                      tokenizer,
                                                      tag_to_ix,
                                                      args.batch_size,
                                                      mode='dev')

    # build model
    net = tagger.models.bert.BertNER.from_pretrained('bert-base-cased', tag_to_ix)
    
    # configurate optimizer
    param_optimizer = list(net.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.lr)

    # start training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {0}.".format(device))
    net.to(device)
    best_f1 = 0.0
    for epoch in trange(args.epochs, desc="Epoch"):
        # train step
        train_loss, nb_tr_steps = net.train_one_epoch(train_dataloader, optimizer, device)
        print("Epoch={0} Train loss: {1}".format(epoch+1, train_loss/nb_tr_steps))

        # evaluation step
        f1 = net.f1_eval(dev_dataloader, device)
        print('Epoch={0} Validation F1: {1:.3f}'.format(epoch+1, f1))

        if f1 > best_f1:
          best_f1 = f1
          output_dir = os.path.join(work_folder, "epoch-{0:05d}".format(epoch+1))
          net.save(output_dir)

if __name__ == '__main__':
    main(args)

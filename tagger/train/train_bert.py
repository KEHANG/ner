import os
import argparse
import torch
from torch import optim
from datetime import datetime
from tqdm import tqdm, trange
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

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

    # create work folder
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    work_folder = 'model-save-{0}'.format(now)
    os.mkdir(work_folder)

    # load data
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

    # featurize data
    bert_model = 'bert-base-cased'
    do_lower_case = False
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
    ## 2.2 tokenize sentences
    train_tokenized_sentences = []
    train_label_ids = []
    for (ex_index, textlist) in enumerate(train_dataset.word_sequences):
        labellist = train_dataset.tag_sequences[ex_index]
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        ntokens = []
        segment_ids = []
        label_ids = []
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(tag_to_ix[labels[i]])
        
        train_tokenized_sentences.append(ntokens)
        train_label_ids.append(torch.tensor(label_ids, dtype=torch.long))


    ## 2.2.1 convert tokens to ids
    train_input_ids = [torch.tensor(tokenizer.convert_tokens_to_ids(sent), dtype=torch.long) for sent in train_tokenized_sentences]
    ## 2.2.2 pad to the same length
    train_padded_input_ids = torch.nn.utils.rnn.pad_sequence(train_input_ids, batch_first=True)
    train_padded_label_ids = torch.nn.utils.rnn.pad_sequence(train_label_ids, batch_first=True)
    ## 2.2.3 get padding masks
    train_attention_masks = torch.tensor([[float(i>0) for i in ii] for ii in train_padded_input_ids])

    ## 2.3 tokenize dev sentences
    dev_tokenized_sentences = []
    dev_label_ids = []
    for (ex_index, textlist) in enumerate(dev_dataset.word_sequences):
        labellist = dev_dataset.tag_sequences[ex_index]
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        ntokens = []
        segment_ids = []
        label_ids = []
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(tag_to_ix[labels[i]])
        
        dev_tokenized_sentences.append(ntokens)
        dev_label_ids.append(torch.tensor(label_ids, dtype=torch.long))

    ## 2.3.1 convert tokens to ids
    dev_input_ids = [torch.tensor(tokenizer.convert_tokens_to_ids(sent), dtype=torch.long) for sent in dev_tokenized_sentences]
    ## 2.3.2 pad to the same length
    dev_padded_input_ids = torch.nn.utils.rnn.pad_sequence(dev_input_ids, batch_first=True)
    dev_padded_label_ids = torch.nn.utils.rnn.pad_sequence(dev_label_ids, batch_first=True)
    ## 2..3.3 get padding masks
    dev_attention_masks = torch.tensor([[float(i>0) for i in ii] for ii in dev_padded_input_ids])

    bs = args.batch_size
    train_data = TensorDataset(train_padded_input_ids, train_attention_masks, train_padded_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

    valid_data = TensorDataset(dev_padded_input_ids, dev_attention_masks, dev_padded_label_ids)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

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
        # TRAIN loop
        net.train()
        train_loss = 0.0
        nb_tr_steps = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # forward pass
            loss = net(b_input_ids, token_type_ids=None,
                         attention_mask=b_input_mask, labels=b_labels)
            # backward pass
            loss.backward()
            # track train loss
            train_loss += loss.item()
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=1.0)
            # update parameters
            optimizer.step()
            net.zero_grad()
        # print train loss per epoch
        print("Epoch={0} Train loss: {1}".format(epoch+1, train_loss/nb_tr_steps))
            
        # Eval loop
        net.eval()
        f1 = net.f1_eval(valid_dataloader, device)
        print('Epoch={0} Validation F1: {1:.3f}'.format(epoch+1, f1))

        if f1 > best_f1:
          best_f1 = f1
          output_dir = os.path.join(work_folder, "epoch-{0:05d}".format(epoch+1))
          net.save(output_dir)

if __name__ == '__main__':
    main(args)

import os
import codecs
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tagger.constant import START_TAG, STOP_TAG, PAD

class PadSequence:
    def __call__(self, batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        # Get each sequence and pad it
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.tensor([len(x) for x in sequences])
        # Don't forget to grab the labels of the *sorted* batch
        labels = [x[1] for x in sorted_batch]
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        return sequences_padded, lengths, labels_padded

class NERDataset(Dataset):
    def __init__(self, root, filename,
                 word_to_ix, tag_to_ix,
                 tag_column_idx=-1):

        self.word_sequences = list()
        self.tag_sequences = list()
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

        datafile = os.path.join(root, filename)
        with codecs.open(datafile, 'r', 'utf-8') as f:
            lines = f.readlines()
        
        curr_words = list()
        curr_tags = list()
        for k, line_raw in enumerate(lines):
            line = line_raw.strip()

            # new sentence or new document
            if len(line) == 0 or line.startswith('-DOCSTART-'):
                if len(curr_words) > 0:
                    self.word_sequences.append(curr_words)
                    self.tag_sequences.append(curr_tags)
                    curr_words = list()
                    curr_tags = list()
                continue
            strings = line.split(' ')
            word = strings[0]
            tag = strings[tag_column_idx]
            curr_words.append(word)
            curr_tags.append(tag)

            if k == len(lines) - 1:
                self.word_sequences.append(curr_words)
                self.tag_sequences.append(curr_tags)
        
        print('Loaded {0} samples from {1}.'.format(len(self.word_sequences), datafile))

    def __getitem__(self, index):

        words = self.word_sequences[index]
        tags = self.tag_sequences[index]
        word_ixs = torch.tensor([self.word_to_ix[word] for word in words])
        tag_ixs = torch.tensor([self.tag_to_ix[tag] for tag in tags])
        return [word_ixs, tag_ixs]

    def __len__(self):
        return len(self.word_sequences)

def get_word_and_tag_to_ix(dset_dir, dset_file, dset_file_dev, dset_file_test, tag_column_idx=-1):

    word_to_ix = {PAD: 0}
    tag_to_ix = {PAD: 0, START_TAG: 1, STOP_TAG: 2}

    
    for filename in [dset_file, dset_file_dev, dset_file_test]:
        datafile = os.path.join(dset_dir, filename)
        with codecs.open(datafile, 'r', 'utf-8') as f:
            lines = f.readlines()
        
        for line_raw in lines:
            line = line_raw.strip()

            # new sentence or new document
            if len(line) == 0 or line.startswith('-DOCSTART-'):
                continue
            strings = line.split(' ')
            word = strings[0]
            tag = strings[tag_column_idx]

            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    return word_to_ix, tag_to_ix

def return_data(dset_dir, dset_file, batch_size, num_workers,
                dset_file_dev, dset_file_test):

    
    word_to_ix, tag_to_ix = get_word_and_tag_to_ix(dset_dir, dset_file,
                                                   dset_file_dev, dset_file_test,
                                                   tag_column_idx=-1)
    train_data = NERDataset(root=dset_dir, filename=dset_file,
                            word_to_ix=word_to_ix, tag_to_ix=tag_to_ix)

    dev_data = NERDataset(root=dset_dir, filename=dset_file_dev,
                          word_to_ix=word_to_ix, tag_to_ix=tag_to_ix)
    
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              collate_fn=PadSequence())
    dev_loader = DataLoader(dev_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=True,
                            collate_fn=PadSequence())

    return train_loader, dev_loader, word_to_ix, tag_to_ix

def prepare_dataloader(dataset, tokenizer, tag_to_ix, batch_size, mode='train'):

    input_ids_for_all_sentences = []
    tag_ids_for_all_sentences = []
    for sentence_idx, words_in_a_sentence in enumerate(dataset.word_sequences):
        tags_for_a_sentence = dataset.tag_sequences[sentence_idx]
        subwords_in_a_sentence = []
        subtags_for_a_sentence = []
        for word_idx, word in enumerate(words_in_a_sentence):
            subwords_from_a_word = tokenizer.tokenize(word)
            subwords_in_a_sentence.extend(subwords_from_a_word)
            tag0 = tags_for_a_sentence[word_idx]
            for m in range(len(subwords_from_a_word)):
                if m == 0:
                    subtags_for_a_sentence.append(tag0)
                else:
                    subtags_for_a_sentence.append("X")
        
        segment_ids = []
        tag_ids = []
        for subword_idx, _ in enumerate(subwords_in_a_sentence):
            segment_ids.append(0)
            tag_ids.append(tag_to_ix[subtags_for_a_sentence[subword_idx]])

        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(subwords_in_a_sentence), dtype=torch.long)

        input_ids_for_all_sentences.append(input_ids)
        tag_ids_for_all_sentences.append(torch.tensor(tag_ids, dtype=torch.long))

    ## pad to the same length
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_for_all_sentences, batch_first=True)
    padded_tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids_for_all_sentences, batch_first=True)
    ## get padding masks
    attention_masks = torch.tensor([[float(i>0) for i in ii] for ii in padded_input_ids])

    dataset = TensorDataset(padded_input_ids, attention_masks, padded_tag_ids)
    if mode == 'train':
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataloader

if __name__ == '__main__':
    dset_dir = '../data'
    dset_file = 'mini_train.txt'
    dset_file_dev = 'mini_dev.txt'
    dset_file_test = 'mini_test.txt'
    batch_size = 2
    num_workers = 1

    dataloader, dataloader_dev, word_to_ix, tag_to_ix = return_data(dset_dir, dset_file, batch_size, num_workers,
                                                                    dset_file_dev, dset_file_test)
    print(tag_to_ix)
    for i_batch, sample_batched in enumerate(dataloader_dev):
        print(sample_batched)
        break


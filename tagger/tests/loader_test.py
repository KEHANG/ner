import os
import unittest

import tagger.loader

class TestLoader(unittest.TestCase):

    def setUp(self):
        test_base = os.path.abspath(os.path.dirname(__file__))
        self.dset_dir = os.path.join(test_base, 'data')
        self.dset_file = 'mini_train.txt'
        self.dset_file_dev = 'mini_dev.txt'
        self.dset_file_test = 'mini_test.txt'

    def test_get_word_and_tag_to_ix(self):

    	word_to_ix, tag_to_ix = tagger.loader.get_word_and_tag_to_ix(self.dset_dir, 
    						self.dset_file, self.dset_file_dev, self.dset_file_test)

    	self.assertEqual(len(word_to_ix), 14987)
    	self.assertEqual(len(tag_to_ix), 12)
        
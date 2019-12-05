import random
from torchtext import data
import re


class mydata(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, datas, examples=None, **kwargs):
        """ Create own dataset instance given a path and fields.

            Arguments:
                text_field: The field that will be used for text data.
                label_field: The field that will be used for label data.
                data: Raw data.
                examples: The examples contain all the data.
                Remaining keyword arguments: Passed to the constructor of
                    data.Dataset.
        """
        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """


            #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            #string = re.sub(r"\'s", " \'s", string)
            #string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            # string = re.sub(r",", " , ", string)
            # string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            # string = re.sub(r"\s{2,}", " ", string)
            #print("in clean",string.strip())
            return string.strip()

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field),('label', label_field)]

        if examples is None:
            examples = []
            # for weibo in datas:
            #     formdat = [str(weibo['text']),str(weibo['label'])]
            #     print("formdat",formdat)
            #     exam = list(data.Example.fromlist(formdat, fields))
            #     examples += exam

            examples += [data.Example.fromlist([weibo['text'],weibo['label']], fields) for weibo in datas]
            print("in examples",len(examples))
        super(mydata, self).__init__(examples, fields, **kwargs)

    @classmethod
    def getdataSplit(cls, text_field, label_field, rawdata, dev_ratio=.1, test_ratio=.1, shuffle=True, **kwargs):
        """Create dataset objects for splits of the MR dataset.

            Arguments:
                text_field: The field that will be used for the sentence.
                label_field: The field that will be used for label data.
                dev_ratio: The ratio that will be used to get split validation dataset.
                shuffle: Whether to shuffle the data before split.
                train: The filename of the train data. Default: 'train.txt'.
                Remaining keyword arguments: Passed to the splits method of
                    Dataset.
        """

        examples = cls(text_field, label_field,datas=rawdata,**kwargs).examples
        if shuffle: random.shuffle(examples)
        dev_index = int(dev_ratio * len(examples))
        test_index = dev_index + int(test_ratio * len(examples))

        return (cls(text_field,label_field,rawdata,examples=examples[test_index:]),
                cls(text_field,label_field,rawdata,examples=examples[:dev_index]),
                cls(text_field,label_field,rawdata,examples=examples[dev_index:test_index]))



import sys
if not 'texar_repo' in sys.path:
  sys.path += ['texar_repo']
from config import *
from texar_repo.examples.bert.utils import tokenization
import tensorflow as tf
import os
import csv
import jsonlines
import collections

class InputExample():
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence.
                For single sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second
                sequence. Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.src_txt = text_a
        self.tgt_txt = text_b


class InputFeatures():
    """A single set of features of data."""

    def __init__(self, src_input_ids,src_input_mask,src_segment_ids,tgt_input_ids,tgt_input_mask,tgt_labels):
        self.src_input_ids = src_input_ids
        self.src_input_mask = src_input_mask
        self.src_segment_ids = src_segment_ids
        self.tgt_input_ids = tgt_input_ids
        self.tgt_input_mask = tgt_input_mask 
        self.tgt_labels = tgt_labels
        
       
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i = 0
            for line in reader:
                lines.append(line)
        return lines

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        with open(input_file, "r") as fr:
            lines = []
            for line in fr:
                lines.append(line.strip())
        return lines
      
      
# class CNNDailymail(DataProcessor):
#     """Processor for the CoLA data set (GLUE version)."""
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_file(os.path.join(data_dir, "train_story.txt")), self._read_file(os.path.join(data_dir, "train_summ.txt")),
#             "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_file(os.path.join(data_dir, "eval_story.txt")),self._read_file(os.path.join(data_dir, "eval_summ.txt")),
#             "dev")
#
#     def get_test_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_file(os.path.join(data_dir, "test_story.txt")), self._read_file(os.path.join(data_dir, "test_summ.txt")),
#             "test")
#
#     def _create_examples(self, src_lines, tgt_lines, set_type):
#         examples = []
#         for i, data in enumerate(zip(src_lines, tgt_lines)):
#             guid = "%s-%s" % (set_type, i)
#             if set_type == "test" and i == 0:
#                 continue
#             else:
#                 #print(data)
#                 if len(data[0]) == 0 or len(data[1]) == 0:
#                   continue
#                 src_lines = tokenization.convert_to_unicode(data[0][0])
#                 tgt_lines = tokenization.convert_to_unicode(data[1][0])
#                 examples.append(InputExample(guid=guid, text_a=src_lines, text_b=tgt_lines))
#         return examples
  
#
# def file_based_convert_examples_to_features(examples, max_seq_length_src,max_seq_length_tgt, tokenizer, output_file):
#     """Convert a set of `InputExample`s to a TFRecord file."""
#     writer = tf.python_io.TFRecordWriter(output_file)
#     for (ex_index, example) in enumerate(examples):
#         if (ex_index + 1) % 1000 == 0:
#             print("------------processed..{}...examples".format(ex_index))
#
#         feature = convert_single_example(example, max_seq_length_src, max_seq_length_tgt, tokenizer)
#
#         def create_int_feature(values):
#             return tf.train.Feature(
#                 int64_list=tf.train.Int64List(value=list(values)))
#
#         features = collections.OrderedDict()
#         features["src_input_ids"] = create_int_feature(feature.src_input_ids)
#         features["src_input_mask"] = create_int_feature(feature.src_input_mask)
#         features["src_segment_ids"] = create_int_feature(feature.src_segment_ids)
#
#         features["tgt_input_ids"] = create_int_feature(feature.tgt_input_ids)
#         features["tgt_input_mask"] = create_int_feature(feature.tgt_input_mask)
#         features['tgt_labels'] = create_int_feature(feature.tgt_labels)
#
#         tf_example = tf.train.Example(features=tf.train.Features(feature=features))
#         writer.write(tf_example.SerializeToString())


def convert_example_to_feature(example, writer,  max_seq_length_src, max_seq_length_tgt, tokenizer):
    """Convert a set of `InputExample`s to a TFRecord file."""
    feature = convert_single_example(example, max_seq_length_src, max_seq_length_tgt, tokenizer)

    def create_int_feature(values):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(values)))

    features = collections.OrderedDict()
    features["src_input_ids"] = create_int_feature(feature.src_input_ids)
    features["src_input_mask"] = create_int_feature(feature.src_input_mask)
    features["src_segment_ids"] = create_int_feature(feature.src_segment_ids)

    features["tgt_input_ids"] = create_int_feature(feature.tgt_input_ids)
    features["tgt_input_mask"] = create_int_feature(feature.tgt_input_mask)
    features['tgt_labels'] = create_int_feature(feature.tgt_labels)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())


def convert_single_example(example, max_seq_length_src,max_seq_length_tgt, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    """
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    """
    tokens_a = tokenizer.tokenize(example.src_txt)
    tokens_b = tokenizer.tokenize(example.tgt_txt)

    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    if len(tokens_a) > max_seq_length_src - 2:
            tokens_a = tokens_a[0:(max_seq_length_src - 2)]
    
    if len(tokens_b) > max_seq_length_tgt - 2:
            tokens_b = tokens_b[0:(max_seq_length_tgt - 2)]

    tokens_src = []
    segment_ids_src = []
    tokens_src.append("[CLS]")
    segment_ids_src.append(0)
    for token in tokens_a:
        tokens_src.append(token)
        segment_ids_src.append(0)
    tokens_src.append("[SEP]")
    segment_ids_src.append(0)

    tokens_tgt = []
    segment_ids_tgt = []
    tokens_tgt.append("[CLS]")
    #segment_ids_tgt.append(0)
    for token in tokens_b:
        tokens_tgt.append(token)
        #segment_ids_tgt.append(0)
    tokens_tgt.append("[SEP]")
    #segment_ids_tgt.append(0)

    input_ids_src = tokenizer.convert_tokens_to_ids(tokens_src)
    input_ids_tgt = tokenizer.convert_tokens_to_ids(tokens_tgt)

    labels_tgt = input_ids_tgt[1:]
    
    #Adding begiining and end token
    input_ids_tgt = input_ids_tgt[:-1] 
    
    input_mask_src = [1] * len(input_ids_src)
    input_mask_tgt = [1] * len(input_ids_tgt)

    while len(input_ids_src) < max_seq_length_src:
        input_ids_src.append(0)
        input_mask_src.append(0)
        segment_ids_src.append(0)

    while len(input_ids_tgt) < max_seq_length_tgt:
        input_ids_tgt.append(0)
        input_mask_tgt.append(0)
        segment_ids_tgt.append(0)
        labels_tgt.append(0)

    feature = InputFeatures( src_input_ids=input_ids_src,src_input_mask=input_mask_src,src_segment_ids=segment_ids_src,
        tgt_input_ids=input_ids_tgt,tgt_input_mask=input_mask_tgt,tgt_labels=labels_tgt)
    return feature


def file_based_input_fn_builder(input_file, max_seq_length_src,max_seq_length_tgt, is_training,
                                drop_remainder, is_distributed=False):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    name_to_features = {
        "src_input_ids": tf.FixedLenFeature([max_seq_length_src], tf.int64),
        "src_input_mask": tf.FixedLenFeature([max_seq_length_src], tf.int64),
        "src_segment_ids": tf.FixedLenFeature([max_seq_length_src], tf.int64),
        "tgt_input_ids": tf.FixedLenFeature([max_seq_length_tgt], tf.int64),
        "tgt_input_mask": tf.FixedLenFeature([max_seq_length_tgt], tf.int64),
        "tgt_labels" : tf.FixedLenFeature([max_seq_length_tgt], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        print(example)
        print(example.keys())

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            if is_distributed:
                import horovod.tensorflow as hvd
                tf.logging.info('distributed mode is enabled.'
                                'size:{} rank:{}'.format(hvd.size(), hvd.rank()))
                # https://github.com/uber/horovod/issues/223
                d = d.shard(hvd.size(), hvd.rank())

                d = d.repeat()
                d = d.shuffle(buffer_size=100)
                d = d.apply(
                    tf.contrib.data.map_and_batch(
                        lambda record: _decode_record(record, name_to_features),
                        batch_size=batch_size//hvd.size(),
                        drop_remainder=drop_remainder))
            else:
                tf.logging.info('distributed mode is not enabled.')
                d = d.repeat()
                d = d.shuffle(buffer_size=100)
                d = d.apply(
                    tf.contrib.data.map_and_batch(
                        lambda record: _decode_record(record, name_to_features),
                        batch_size=batch_size,
                        drop_remainder=drop_remainder))
        else:
            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder))
        return d
    return input_fn
  

def process(mode, tokenizer, src_file, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    index = -1
    with open(src_file, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            index += 1
            summary_list = item.get("summary", [])
            text_list = item.get("text", [])
            summary_str = "".join(summary_list)
            content_str = "".join(text_list)

            pos = summary_str.find("<?xml:")
            if pos >= 0:
                continue
            summary_str = summary_str.replace("\n", ".").strip()
            content_str = content_str.replace("\n", ".").strip()
            guid = "%s-%s" % (mode, index)
            if mode == "test" and index == 0:
                continue
            else:
                src_lines = tokenization.convert_to_unicode(content_str)
                tgt_lines = tokenization.convert_to_unicode(summary_str)
                cur_example = InputExample(guid=guid, text_a=src_lines, text_b=tgt_lines)
                convert_example_to_feature(cur_example, writer, max_seq_length_src, max_seq_length_tgt, tokenizer)


def newsroom_2_tfrecoder(
                tokenizer,
                data_dir,
                max_seq_length_src,
                max_seq_length_tgt,
                batch_size,
                mode,
                output_dir,
                is_distributed=False):
    src_file = os.path.join(src_data_dir, "%s.label.info.jsonl" % mode)
    output_file = os.path.join(output_dir, "%s.tf_record" % mode)
    if os.path.exists(output_file):
        print("%s already exist!" % output_file)
        return
    process(mode, tokenizer, src_file, output_file)
    if mode == 'train':
        dataset = file_based_input_fn_builder(
            input_file=output_file,
            max_seq_length_src=max_seq_length_src,
            max_seq_length_tgt =max_seq_length_tgt,
            is_training=True,
            drop_remainder=True,
            is_distributed=is_distributed)({'batch_size': batch_size})
    elif mode == 'dev':
        dataset = file_based_input_fn_builder(
            input_file=output_file,
            max_seq_length_src=max_seq_length_src,
            max_seq_length_tgt =max_seq_length_tgt,
            is_training=False,
            drop_remainder=True,
            is_distributed=is_distributed)({'batch_size': batch_size})
    elif mode == 'test':
        dataset = file_based_input_fn_builder(
            input_file=output_file,
            max_seq_length_src=max_seq_length_src,
            max_seq_length_tgt =max_seq_length_tgt,
            is_training=False,
            drop_remainder=True,
            is_distributed=is_distributed)({'batch_size': batch_size})
    return dataset


if __name__ == "__main__":
    tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_pretrain_dir, 'vocab.txt'),
        do_lower_case=True)

    vocab_size = len(tokenizer.vocab)

    # train_dataset = newsroom_2_tfrecoder(tokenizer, data_dir, max_seq_length_src, max_seq_length_tgt, batch_size, 'train', data_dir)
    eval_dataset =  newsroom_2_tfrecoder(tokenizer, data_dir, max_seq_length_src, max_seq_length_tgt, eval_batch_size, 'test', data_dir)
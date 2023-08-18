import os, random
import numpy as np

from .base import DataProcessor, InputExample

class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, r):
        """See base class."""
        data_dir = os.path.join(data_dir, 'MRPC')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", r=r)

    def get_dev_examples(self, data_dir):
        """See base class."""
        data_dir = os.path.join(data_dir, 'MRPC')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, r=None):
        """Creates examples for the training and dev sets."""
        random.seed(0)
        np.random.seed(0)
        os.environ["PYTHONHASHSEED"] = str(0)
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            if r is not None and np.random.random() < r:
                new_label = label
                num = len(self.get_labels())
                while new_label == label:
                    new_label = self.get_labels()[np.random.randint(num)]
                label = new_label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, r):
        """See base class."""
        data_dir = os.path.join(data_dir, 'MNLI')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", r=r)

    def get_dev_examples(self, data_dir):
        """See base class."""
        data_dir = os.path.join(data_dir, 'MNLI')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type, r=None):
        """Creates examples for the training and dev sets."""
        random.seed(0)
        np.random.seed(0)
        os.environ["PYTHONHASHSEED"] = str(0)
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            if r is not None and np.random.random() < r:
                new_label = label
                num = len(self.get_labels())
                while new_label == label:
                    new_label = self.get_labels()[np.random.randint(num)]
                label = new_label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        data_dir = os.path.join(data_dir, 'MNLI')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")

class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir, r):
        """See base class."""
        data_dir = os.path.join(data_dir, 'CoLA')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", r=r)

    def get_dev_examples(self, data_dir):
        """See base class."""
        data_dir = os.path.join(data_dir, 'CoLA')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, r=None):
        """Creates examples for the training and dev sets."""
        random.seed(0)
        np.random.seed(0)
        os.environ["PYTHONHASHSEED"] = str(0)
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            if r is not None and np.random.random() < r:
                new_label = label
                num = len(self.get_labels())
                while new_label == label:
                    new_label = self.get_labels()[np.random.randint(num)]
                label = new_label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir, r):
        """See base class."""
        data_dir = os.path.join(data_dir, 'SST-2')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", r=r)

    def get_dev_examples(self, data_dir):
        """See base class."""
        data_dir = os.path.join(data_dir, 'SST-2')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, r=None):
        """Creates examples for the training and dev sets."""
        random.seed(0)
        np.random.seed(0)
        os.environ["PYTHONHASHSEED"] = str(0)
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            if r is not None and np.random.random() < r:
                new_label = label
                num = len(self.get_labels())
                while new_label == label:
                    new_label = self.get_labels()[np.random.randint(num)]
                label = new_label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir, r):
        """See base class."""
        data_dir = os.path.join(data_dir, 'STS-B')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        data_dir = os.path.join(data_dir, 'STS-B')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        random.seed(0)
        np.random.seed(0)
        os.environ["PYTHONHASHSEED"] = str(0)
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class QqpProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir, r):
        """See base class."""
        data_dir = os.path.join(data_dir, 'QQP')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", r=r)

    def get_dev_examples(self, data_dir):
        """See base class."""
        data_dir = os.path.join(data_dir, 'QQP')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, r=None):
        """Creates examples for the training and dev sets."""
        random.seed(0)
        np.random.seed(0)
        os.environ["PYTHONHASHSEED"] = str(0)
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            if r is not None and np.random.random() < r:
                new_label = label
                num = len(self.get_labels())
                while new_label == label:
                    new_label = self.get_labels()[np.random.randint(num)]
                label = new_label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class QnliProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir, r):
        """See base class."""
        data_dir = os.path.join(data_dir, 'QNLI')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", r=r)

    def get_dev_examples(self, data_dir):
        """See base class."""
        data_dir = os.path.join(data_dir, 'QNLI')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type, r=None):
        """Creates examples for the training and dev sets."""
        random.seed(0)
        np.random.seed(0)
        os.environ["PYTHONHASHSEED"] = str(0)
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            if r is not None and np.random.random() < r:
                new_label = label
                num = len(self.get_labels())
                while new_label == label:
                    new_label = self.get_labels()[np.random.randint(num)]
                label = new_label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir, r):
        """See base class."""
        data_dir = os.path.join(data_dir, 'RTE')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", r=r)

    def get_dev_examples(self, data_dir):
        """See base class."""
        data_dir = os.path.join(data_dir, 'RTE')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type, r=None):
        """Creates examples for the training and dev sets."""
        random.seed(0)
        np.random.seed(0)
        os.environ["PYTHONHASHSEED"] = str(0)
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            if r is not None and np.random.random() < r:
                new_label = label
                num = len(self.get_labels())
                while new_label == label:
                    new_label = self.get_labels()[np.random.randint(num)]
                label = new_label
            if i < 10:
                print (label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, r):
        """See base class."""
        data_dir = os.path.join(data_dir, 'WNLI')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", r=r)

    def get_dev_examples(self, data_dir):
        """See base class."""
        data_dir = os.path.join(data_dir, 'WNLI')
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, r=None):
        """Creates examples for the training and dev sets."""
        random.seed(0)
        np.random.seed(0)
        os.environ["PYTHONHASHSEED"] = str(0)
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            if r is not None and np.random.random() < r:
                new_label = label
                num = len(self.get_labels())
                while new_label == label:
                    new_label = self.get_labels()[np.random.randint(num)]
                label = new_label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
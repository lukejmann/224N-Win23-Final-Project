import datasets
from datasets.tasks import TextClassification
import os

_DESCRIPTION = """\
Biden text classification dataset. This dataset is a collection of transcripts from Joe Biden's media appearances scraped from SITE. As well as a collection of transcripts from other random media appearances scraped from the same website. T
"""

_CITATION = """\
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Mann, Luke and Spector, Ori},
  title     = {Joe Biden's Media Appearances},
}
"""


# file sturcture will be:
# data/
#   subject/
#     train/
#       pos/
#         0.txt
#         1.txt
#         ...
#       neg/
#         0.txt
#         1.txt
#         ...
#     test/
#       pos/
#         0.txt
#         1.txt
#         ...
#       neg/
#         0.txt
#         1.txt
#         ...


class SubjectClassifierConfig(datasets.BuilderConfig):
    """BuilderConfig for SubjectClassifier."""

    def __init__(self, **kwargs):
        super(SubjectClassifierConfig, self).__init__(
            version=datasets.Version("1.0.0"), **kwargs
        )


class SubjectClassifier(datasets.GeneratorBasedBuilder):
    """SubjectClassifier dataset."""

    BUILDER_CONFIGS = [
        SubjectClassifierConfig(
            name="plain_text",
            description="Plain text",
        )
    ]

    def __init__(self, *args, writer_batch_size=None, subject="biden", **kwargs):
        self.subject = subject

        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=["neg", "pos"]),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
            task_templates=[
                TextClassification(text_column="text", label_column="label")
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        # load from local
        train_paths = [
            "dataset/biden_classifier/data/train/pos",
            "dataset/biden_classifier/data/train/neg",
        ]
        test_paths = [
            "dataset/biden_classifier/data/test/pos",
            "dataset/biden_classifier/data/test/neg",
        ]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "files": dl_manager.iter_files(train_paths),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "files": dl_manager.iter_files(test_paths),
                },
            ),
        ]

    def _generate_examples(self, files, split, labeled=True):
        # label_mapping = {"pos": 1, "neg": 0}
        # print(files)
        label_mapping = {"pos": 1, "neg": 0}
        for path in files:
            # if path.startswith(f"data/{split}"):
            label = label_mapping[path.split("/")[-2]]
            f = open(path, "r")
            if label is not None:
                yield path, {"text": f.read(), "label": label}
        # else:
        #     for path, f in files:
        #         if path.startswith(f"aclImdb/{split}"):
        #             if path.split("/")[2] == "unsup":
        #                 yield path, {"text": f.read().decode("utf-8"), "label": -1}

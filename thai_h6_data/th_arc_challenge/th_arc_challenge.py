# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
"""



import csv
import json
import os

import datasets


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {th_ai2_arc},
author={huggingface, Inc.
},
year={2023}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URL = "th_arc_challenge.zip"
_VERSION = "0.0.1"

class Ai2ArcConfig(datasets.BuilderConfig):
    """BuilderConfig for Ai2ARC."""

    def __init__(self, **kwargs):
        """BuilderConfig for Ai2Arc.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        # version 0.0.0 - initial release
        super(Ai2ArcConfig, self).__init__(version=datasets.Version(_VERSION, ""), **kwargs)

class Ai2Arc(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version(_VERSION)

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # BUILDER_CONFIGS = [
    #     Ai2ArcConfig(
    #         name="ARC-Challenge",
    #         description="""
    #         Challenge Set of 2590 “hard” questions (those that both a retrieval and a co-occurrence method fail to answer correctly)
    #         """,
    #     ),
    #     Ai2ArcConfig(
    #         name="ARC-Easy",
    #         description="""
    #       Easy Set of 5197 questions
    #       """,
    #     ),
    # ]

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "choices": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "label": datasets.Value("string"),
                        }
                    ),
                    "answerKey": datasets.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            ),
            supervised_keys=None,
            homepage="https://allenai.org/data/arc",
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        data_dir = dl_manager.download_and_extract(_URL)

        # subset = self.config.name.replace('-', '_')
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"th_arc_challenge__few-shot.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"th_arc_challenge__few-shot.json"),
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"th_arc_challenge__test.json"),
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        """Yields examples."""
        # open json file
        with open(filepath, encoding="utf-8") as f:
            json_data = json.load(f)
            for data in json_data:
                answerkey = data["answerKey"]

                id_ = data["id"]
                question = data["th_question"]
                text_choices = data["choices"]["th_text"]
                label_choices = data["choices"]["label"]
                yield id_, {
                    "id": id_,
                    "answerKey": answerkey,
                    "question": question,
                    "choices": {
                        "text": text_choices,
                        "label": label_choices
                    },
                }

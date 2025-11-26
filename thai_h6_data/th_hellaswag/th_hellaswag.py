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
title = {th_hellaswag},
author={huggingface, Inc.
},
year={2023}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
HellaSwag: Can a Machine Really Finish Your Sentence? is a new dataset for commonsense NLI. A paper was published at ACL2019.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = "th_hellaswag.zip"
_VERSION = "0.0.1"

class Hellaswag(datasets.GeneratorBasedBuilder):
    """TODO(hellaswag): Short description of my dataset."""

    # TODO(hellaswag): Set up version.
    VERSION = datasets.Version(_VERSION)

    def _info(self):
        # TODO(hellaswag): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    # These are the features of your dataset like images, labels ...
                    "ind": datasets.Value("int32"),
                    "activity_label": datasets.Value("string"),
                    "ctx_a": datasets.Value("string"),
                    "ctx_b": datasets.Value("string"),
                    "ctx": datasets.Value("string"),
                    "endings": datasets.features.Sequence(datasets.Value("string")),
                    "source_id": datasets.Value("string"),
                    "split_type": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            # Homepage of the dataset for documentation
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(hellaswag): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        urls = _URLS
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"th_hellaswag__few-shot.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"th_hellaswag__validation.json"),
                    "split": "validation",
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     gen_kwargs={
            #         "filepath": os.path.join(data_dir, f"th_hellaswag__test.json"),
            #         "split": "test"
            #     },
            # ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        # TODO(hellaswag): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            json_data = json.load(f)
            for key, data in enumerate(json_data):
                yield key, {
                    "ind": int(data["ind"]),
                    "activity_label": data["th_activity_label"],
                    "ctx_a": data.get("th_ctx_a", ""),
                    "ctx_b": data.get("th_ctx_b", ""),
                    "ctx": data["th_ctx"],
                    "endings": data.get("th_endings", []),
                    "source_id": data["source_id"],
                    "split_type": data["split_type"],
                    "label": str(data.get("label", "")),
                }

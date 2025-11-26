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
"""TruthfulQA dataset."""


import os
import json

import datasets


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {th_truthful_qa},
author={huggingface, Inc.
},
year={2023}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
TruthfulQA is a benchmark to measure whether a language model is truthful in
generating answers to questions. The benchmark comprises 817 questions that
span 38 categories, including health, law, finance and politics. Questions are
crafted so that some humans would answer falsely due to a false belief or
misconception. To perform well, models must avoid generating false answers
learned from imitating human texts.
"""


# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "multiple_choice": "th_truthfulqa.zip",
}
_VERSION = "0.0.1"


class TruthfulQaConfig(datasets.BuilderConfig):
    """BuilderConfig for TruthfulQA."""

    def __init__(self, features, **kwargs):
        """BuilderConfig for TruthfulQA.
        Args:
          features: *list[string]*, list of features that'll appear in the feature dict.
          **kwargs: keyword arguments forwarded to super.
        """
        super(TruthfulQaConfig, self).__init__(version=datasets.Version(_VERSION, ""), **kwargs)
        self.features = features


class TruthfulQa(datasets.GeneratorBasedBuilder):
    """TruthfulQA is a benchmark to measure whether a language model is truthful in generating answers to questions."""

    VERSION = datasets.Version(_VERSION)
    BUILDER_CONFIGS = [
        # TruthfulQaConfig(
        #     name="generation",
        #     features=datasets.Features(
        #         {
        #             "type": datasets.Value("string"),
        #             "category": datasets.Value("string"),
        #             "question": datasets.Value("string"),
        #             "best_answer": datasets.Value("string"),
        #             "correct_answers": datasets.features.Sequence(datasets.Value("string")),
        #             "incorrect_answers": datasets.features.Sequence(datasets.Value("string")),
        #             "source": datasets.Value("string"),
        #         }
        #     ),
        #     description="The Generation TruthfulQA (main) task tests a model's ability to generate 1-2 sentence answers for a given question truthfully.",
        # ),
        TruthfulQaConfig(
            name="multiple_choice",
            features=datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "mc1_targets": {
                        "choices": datasets.features.Sequence(datasets.Value("string")),
                        "labels": datasets.features.Sequence(datasets.Value("int32")),
                    },
                    "mc2_targets": {
                        "choices": datasets.features.Sequence(datasets.Value("string")),
                        "labels": datasets.features.Sequence(datasets.Value("int32")),
                    },
                }
            ),
            description="The Multiple-Choice TruthfulQA task provides a multiple-choice option to test a model's ability to identify true statements.",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self.config.features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)
        subset = self.config.name.replace('-', '_')

        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"th_truthful_qa__{subset}__validation.json"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        if self.config.name == "multiple_choice":
            # Multiple choice data
            with open(filepath, encoding="utf-8") as f:
                json_data = json.load(f)
                for key, data in enumerate(json_data):
                    yield key, {
                        "question": data["th_question"],
                        "mc1_targets": {
                            "choices": data["mc1_targets"]["th_choices"],
                            "labels": data["mc1_targets"]["labels"]
                        },
                        "mc2_targets": {
                            "choices": data["mc2_targets"]["th_choices"],
                            "labels": data["mc2_targets"]["labels"]
                        },
                    }
        # elif self.config.name == "generation":
        #     # Generation data
        #     with open(filepath, encoding="utf-8") as f:
        #         json_data = json.load(f)
        #         for key, data in enumerate(json_data):
        #             if not data["th_correct_answers"] or not data["th_incorrect_answers"]:
        #                 continue
        #             yield key, {
        #                 "type": data["type"],
        #                 "category": data["category"],
        #                 "question": data["th_question"],
        #                 "best_answer": data["th_best_answer"],
        #                 "correct_answers": data["th_correct_answers"],
        #                 "incorrect_answers": data["th_incorrect_answers"],
        #                 "source": data["source"],
        #             }

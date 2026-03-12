"""
CUTE
"""

from typing import List, Union
from functools import partial

from oe_eval.components.instances import RequestInstance
from oe_eval.components.requests import LoglikelihoodRequest, RequestType
from oe_eval.metrics.metric import GreedyAccuracyMetric, MCAccuracy
from oe_eval.tasks.base_task import Task
from oe_eval.tasks.utils import map_indexed

_CITATION = """
@article{edman2024cute,
  title={CUTE: Measuring LLMs' Understanding of Their Tokens},
  author={Edman, Lukas and Schmid, Helmut and Fraser, Alexander},
  journal={arXiv preprint arXiv:2409.15452},
  year={2024}
}
"""


class CUTE(Task):
    VERSION = 0
    REQUEST_TYPE = RequestType.LOGLIKELIHOOD
    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": "leukas/cute",
        "native_id_field": "id",  # like GPQA, auto index
        "primary_metric": "greedy_acc",
        "split": "test",
        "context_kwargs": None,
        "compute_gold_bpb": True,
    }

    def make_metrics(self):
        self._metrics = [GreedyAccuracyMetric(**self.task_config["metric_kwargs"])]
        self._metrics.append(MCAccuracy(**self.task_config.get("metric_kwargs", {})))
        return self._metrics

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        data = []

        for split in self.dataset:
            data.extend(map_indexed(partial(self._process_doc, split=split), self.dataset[split]))

        return data

    def _process_doc(self, doc, index=-1, split=None):
        answer_prefix = "\"" if split in {"contains_char", "contains_word"} else "\" "

        # skip the closing quote since it runs into tokenization bias problems
        # e.g. dolma2 tokenizer has a "\"\n" token so "\"" is not the greedy continuation
        # and other tokenizers might have other tokens taking precedence
        answer_text = answer_prefix + doc["answer"]
        query = doc["prompt"].strip() + " Answer:"

        category = doc.get("split", None)
        example_id = index if split is None else f"{split}_{index}"
        if category is not None:
            example_id = f"{category}_{example_id}"

        out_doc = {
            "id": example_id,
            "query": query,
            "answers_text": [answer_text],  # expect list valued
            "choices": [answer_text],  # for perplexity eval
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + doc["answers_text"][0]

    def construct_requests(
        self, doc: dict, ctx: Union[str, list, dict], doc_id: int
    ) -> List[RequestInstance]:
        target = self.doc_to_target(doc)

        return [
            RequestInstance(
                request_type=RequestType.LOGLIKELIHOOD.value,
                doc=doc,
                request=LoglikelihoodRequest(
                    context=ctx,
                    continuation=target,
                ),
                idx=0,
                task_name=self.task_name,
                doc_id=doc_id,
                native_id=doc.get(self.task_config.get("native_id_field", "id")),
                label=doc["answers_text"],
            )
        ]
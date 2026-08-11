# Copied from eleuther_huggingface.py.
# Changes marked with NOTE(BOLMO).

import copy
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F
import transformers
from lm_eval import utils
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import Collator, pad_and_concat, stop_sequences_criteria
from tqdm import tqdm

from oe_eval.components.instances import RequestInstance
from oe_eval.components.requests import (
    GenerateUntilAndLoglikelihoodRequest,
    GenerateUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
)
from oe_eval.utilities.model_results_collation import collate_results
from oe_eval.utils import cut_at_stop_sequence

# Minimally modified version of model inference code from lm_eval, (models/huggingface.py)
# adding _verbose versions of various methods to return additional information

eval_logger = utils.eval_logger


class HFLM_Bolmo_Verbose(HFLM):
    def __init__(
        self,
        pretrained: Optional[Union[str, transformers.PreTrainedModel]] = None,
        device: Optional[str] = None,
        device_map_option: Optional[str] = "auto",
        dtype: Optional[Union[str, torch.dtype]] = None,
        # Force patch boundaries instead of letting the model predict them.
        #   None                  -> unchanged behaviour (model predicts its own)
        #   "self"                -> the model's own predicted mask, fed back in (null control)
        #   <hf tokenizer id>     -> that tokenizer's segmentation, e.g. "google/gemma-2-2b"
        # Declared explicitly rather than read out of **kwargs, because anything left in
        # kwargs is forwarded to HFLM.__init__, which rejects unknown arguments.
        boundary_source: Optional[str] = None,
        # Path to write the patch boundaries actually used, one JSON object per scored
        # sequence. Requires `boundary_source`; pass "self" to capture what the model
        # predicts for itself, which is bit-identical to leaving boundaries unforced.
        boundary_dump: Optional[str] = None,
        # Ablation: drop the expanded-embedding channel, which adds a learned
        # dolma2-subword feature to every byte independently of where patches fall.
        disable_expanded_embeddings: bool = False,
        **kwargs,
        # pretrained: Optional[Union[str, transformers.PreTrainedModel]] = "gpt2",
        # backend: Optional[Literal["default", "causal", "seq2seq"]] = "default",
        # override whether the model should be treated as decoder-only (causal) or encoder-decoder (seq2seq)
        # revision: Optional[str] = "main",
        # subfolder: Optional[str] = None,
        # tokenizer: Optional[
        #    Union[
        #        str,
        #        transformers.PreTrainedTokenizer,
        #        transformers.PreTrainedTokenizerFast,
        #    ]
        # ] = None,
        # truncation: Optional[bool] = False,
        # logits_cache: bool = True,
        # max_length: Optional[int] = None,
        # device: Optional[str] = "cuda",
        # dtype: Optional[Union[str, torch.dtype]] = "auto",
        # batch_size: Optional[Union[int, str]] = 1,
        # max_batch_size: Optional[int] = 64,
        # trust_remote_code: Optional[bool] = False,
        # use_fast_tokenizer: Optional[bool] = True,
        # add_bos_token: Optional[bool] = False,
        # prefix_token_id: Optional[int] = None,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # parallelize: Optional[bool] = False,
        # device_map_option: Optional[str] = "auto",
        # max_memory_per_gpu: Optional[Union[int, str]] = None,
        # max_cpu_memory: Optional[Union[int, str]] = None,
        # offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        # PEFT, delta weights and quantization options
        # peft: Optional[str] = None,
        # delta: Optional[str] = None,
        # autogptq: Optional[Union[bool, str]] = False,
        # **kwargs,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.device_count() > 0 else "cpu"
        if device_map_option is None:
            if torch.cuda.device_count() > 1:
                device_map_option = "balanced_low_0"
            elif torch.cuda.device_count() > 0:
                device_map_option = "auto"
        if "revision" in kwargs and kwargs["revision"] is None:
            # Hack to deal with Eleuther using "main" as a default for "revision"
            kwargs = kwargs.copy()
            del kwargs["revision"]
        if torch.cuda.device_count() > 1:
            kwargs["parallelize"] = True
        super().__init__(
            pretrained, device=device, dtype=dtype, device_map_option=device_map_option, **kwargs
        )
        self.tokenizer_size = len(self.tokenizer)

        # NOTE(BOLMO): byte ids are laid out as the BLT special tokens followed by the 256
        # byte values, doubled for the fused patch-ending half (BolmoTokenizerConfig), so a
        # raw byte b is encoded as b + n_special_ids within each half. Read off the
        # tokenizer rather than restated here, since both bounds are checkpoint facts.
        assert (
            self.tokenizer.config.special_tokens_first
        ), "byte-id layout assumes the special tokens come first"
        self.n_special_ids = self.tokenizer.offset
        self.vocab_size_bolmo = self.tokenizer.config.vocab_size // 2
        assert self.vocab_size_bolmo == self.n_special_ids + 256

        self.newline_id = self.tok_encode("\n")[-1]

        # NOTE(BOLMO): forced-boundary evaluation. Loaded once per run rather than per batch.
        self.boundary_source = boundary_source
        if boundary_source is not None and boundary_source != "self":
            self.boundary_tokenizer = transformers.AutoTokenizer.from_pretrained(
                boundary_source, use_fast=True
            )
        else:
            self.boundary_tokenizer = None

        self.boundary_dump = boundary_dump
        if boundary_dump is not None:
            if boundary_source is None:
                raise ValueError(
                    "boundary_dump requires boundary_source; use 'self' to capture the "
                    "model's own boundaries without changing its behaviour"
                )
            # truncate any previous run so a re-run does not append to stale rows
            open(boundary_dump, "w").close()

        # NOTE(BOLMO): a plain attribute on the local encoder gates both the addition in
        # _embed and the lazy derivation of expanded_input_ids, so flipping it is enough.
        if disable_expanded_embeddings:
            self.model.model.local_encoder.add_expanded_embeddings = False  # type: ignore
            eval_logger.info("BOLMO: expanded-embedding channel disabled")

    def unload_model(self):
        # Unload model from GPU, following advice in https://stackoverflow.com/questions/69357881
        import gc

        del self._model  # type: ignore
        self._model = None
        gc.collect()
        torch.cuda.empty_cache()

    def _unmask_tokens(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Returns the token and a mask for correct labels (or Ellipsis if no mask is needed)"""
        mask = tokens >= 0
        if mask.all():
            # all tokens are valid, no mask needed
            # Ellipsis is useful because it can be used for indexing without adding extra computation
            return (tokens, Ellipsis)
        else:
            return (torch.where(mask, tokens, tokens + self.tokenizer_size), mask)

    def _mask_tokens(
        self,
        tokens: List[int],
        masked_spans: List[Tuple[int, int]],
        token_spans: List[Tuple[int, int]],
    ) -> List[int]:
        if not masked_spans:
            return tokens

        tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=self.device)
        masked_spans_tensor = torch.tensor(masked_spans, dtype=torch.long, device=self.device)
        for token_offset in range(0, len(token_spans), 8192):  # process tokens in chunks of 8192
            token_spans_chunk = torch.tensor(token_spans, dtype=torch.long, device=self.device)

            token_starts_after_mask_ends = (
                token_spans_chunk[:, None, 0] >= masked_spans_tensor[None, :, 1]
            )
            token_ends_before_mask_starts = (
                token_spans_chunk[:, None, 1] <= masked_spans_tensor[None, :, 0]
            )
            no_overlap = token_starts_after_mask_ends | token_ends_before_mask_starts
            has_overlap = ~torch.all(no_overlap, dim=1)

            indices = torch.where(has_overlap)[0] + token_offset
            tokens_tensor[indices] -= self.tokenizer_size
        return tokens_tensor.cpu().tolist()

    def _mask_text(self, text: str, masked_spans: List[Tuple[int, int]]) -> str:
        output = ""
        start = 0
        for span in masked_spans:
            end = span[0]
            output += text[start:end]
            start = span[1]
        output += text[start:]
        return output

    # NOTE(BOLMO): the inherited HFLM._model_call forwards only attn_mask/labels, so a
    # boundary mask cannot reach the model through it. Without a mask we delegate to the
    # parent unchanged, which keeps runs without `boundary_source` identical to before.
    def _model_call(self, inps, attn_mask=None, labels=None, boundary_mask=None):
        if boundary_mask is None:
            return super()._model_call(inps, attn_mask=attn_mask, labels=labels)

        if attn_mask is not None or labels is not None:
            raise NotImplementedError("boundary_mask is not supported for seq2seq models")

        assert self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
        with torch.no_grad():
            return self.model(inps, boundary_mask=boundary_mask).logits

    def _row_bytes_and_positions(self, ids: List[int]) -> Tuple[bytes, List[int], List[int]]:
        """Split one row of byte ids into its raw bytes, their row positions, and specials."""
        data = bytearray()
        byte_positions: List[int] = []
        special_positions: List[int] = []
        for position, token_id in enumerate(ids):
            # the modulo folds the fused patch-ending half back onto the plain one
            base = token_id % self.vocab_size_bolmo
            if base >= self.n_special_ids:
                data.append(base - self.n_special_ids)
                byte_positions.append(position)
            else:
                special_positions.append(position)
        return bytes(data), byte_positions, special_positions

    @staticmethod
    def boundary_dump_key(context: str, continuation: str) -> str:
        """Stable join key between a dumped row and the request that produced it."""
        digest = hashlib.sha1(f"{context}\x00{continuation}".encode("utf-8"))
        return digest.hexdigest()

    def _dump_boundaries(self, chunk, boundary_mask: torch.Tensor, inplens: List[int]) -> None:
        """Append the boundaries actually used for each sequence in this batch.

        Rows are emitted in the collator's sorted order, not request order, so each carries
        a content hash rather than an index; join against the requests file offline.
        Boundary positions are stored rather than a full bool vector, which is both smaller
        and directly readable.
        """
        with open(cast(str, self.boundary_dump), "a") as handle:
            for (request_str, _, _), row, inplen in zip(chunk, boundary_mask, inplens):
                context, continuation = request_str
                positions = torch.nonzero(row[: int(inplen)], as_tuple=False)
                handle.write(
                    json.dumps(
                        {
                            "key": self.boundary_dump_key(context, continuation),
                            "n_bytes": int(inplen),
                            "boundaries": positions.flatten().tolist(),
                        }
                    )
                    + "\n"
                )

    def _predicted_boundary_mask(self, batched_inps: torch.Tensor) -> torch.Tensor:
        """The mask the model would have predicted for itself, for the null control.

        Deliberately not `prefill_boundary_prediction_forward`: that passes a
        `boundary_state`, which makes the local encoder overwrite the final position
        (`boundary_mask[:, -1] = boundary_state.mask`). The loglikelihood path calls the
        model with no boundary_state, so reproducing it means leaving that out.
        """
        inner = self.model.model  # type: ignore
        with torch.no_grad():
            expanded_input_ids = None
            if inner.local_encoder.add_expanded_embeddings:
                # BolmoModel.forward derives these lazily when they are not supplied;
                # expand_byte_ids emits one id per byte, so the rows stay rectangular
                expanded_input_ids = torch.tensor(
                    [inner.tokenizer.expand_byte_ids(row) for row in batched_inps.tolist()],
                    dtype=torch.long,
                    device=batched_inps.device,
                )

            _, _, _, boundary_mask = inner.local_encoder(
                input_ids=batched_inps,
                expanded_input_ids=expanded_input_ids,
                true_boundary_mask=None,
                boundary_state=None,
                pad_state=None,
            )

        return cast(torch.Tensor, boundary_mask)

    def _build_boundary_mask(
        self, batched_inps: torch.Tensor, inplens: List[int]
    ) -> torch.Tensor:
        """Forced patch boundaries for an already-batched input; True ends a patch.

        `batched_inps` is [batch, padding_len], right-padded, taken after `_unmask_tokens`.
        With `boundary_source="self"` this returns the model's own predicted mask, so
        feeding it back in must reproduce the unforced result exactly.
        """
        if self.boundary_source == "self":
            return self._predicted_boundary_mask(batched_inps)

        assert self.boundary_tokenizer is not None
        batch_size, padding_len = batched_inps.shape
        mask = torch.zeros(
            batch_size, padding_len, dtype=torch.bool, device=batched_inps.device
        )

        for row, (ids, inplen) in enumerate(zip(batched_inps.tolist(), inplens)):
            data, byte_positions, special_positions = self._row_bytes_and_positions(
                ids[: int(inplen)]
            )

            # BOS is always a boundary (the model forces position 0 regardless), and any
            # other special token stands as its own patch
            mask[row, 0] = True
            for position in special_positions:
                mask[row, position] = True

            if not data:
                continue

            text = data.decode("utf-8", errors="replace")
            offsets = self.boundary_tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True
            )["offset_mapping"]

            # offset_mapping is in characters but the model indexes bytes, and the two
            # disagree on any multi-byte character, so convert through a prefix table
            char_to_byte = [0]
            for character in text:
                char_to_byte.append(char_to_byte[-1] + len(character.encode("utf-8")))

            for _, char_end in offsets:
                byte_end = char_to_byte[char_end]
                # the mask marks the last byte OF a patch, not the gap after it
                if 0 < byte_end <= len(byte_positions):
                    mask[row, byte_positions[byte_end - 1]] = True

        return mask

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer,
            stop_sequences=stop,
            # bolmo.generate applies stopping criteria only over the continuation
            initial_decoder_input_length=0,
            # bolmo.generate applies stopping criteria separately per example
            batch_size=1,
        )
        return self.model.generate(
            context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )

    def tok_encode(
        self,
        text: str,
        masked_spans: Optional[List[Tuple[int, int]]] = None,
        left_truncate_len=None,
        add_special_tokens=None,
    ) -> List[int]:
        if add_special_tokens is None:
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                add_special_tokens = False
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                add_special_tokens = True

        if masked_spans:
            outputs = self.tokenizer.encode_plus(
                text, return_offsets_mapping=True, add_special_tokens=add_special_tokens
            )
            encoding = self._mask_tokens(
                outputs["input_ids"], masked_spans, outputs["offset_mapping"]
            )
        else:
            encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            if self.add_bos_token:
                encoding = [self.tokenizer.bos_token_id] + encoding

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def loglikelihood_rolling_verbose(
        self,
        requests: List[LoglikelihoodRollingRequest],
        disable_tqdm: bool = False,
        verbose_token_logits: bool = False,
    ) -> List[dict]:
        loglikelihoods = []

        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        rolling_token_windows = []
        window_request_ids = []

        for i, request in enumerate(tqdm(requests, disable=(disable_tqdm or (self.rank != 0)))):
            request_rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(
                            request.context, masked_spans=request.masked_context_spans
                        ),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            request_rolling_token_windows = [(None,) + x for x in request_rolling_token_windows]
            rolling_token_windows.extend(request_rolling_token_windows)
            window_request_ids.extend([i] * len(request_rolling_token_windows))

        pad_amnt = 0
        if self.world_size > 1:
            # We pad out the external document-level iterator so the inner iterator doesn't hang
            mytensor = torch.tensor(len(rolling_token_windows), device=self.device)
            gathered = self.accelerator.gather(mytensor).cpu().detach().numpy().tolist()

            pad_amnt = max(gathered) - gathered[self.rank]
            if pad_amnt > 0:
                rolling_token_windows += pad_amnt * [rolling_token_windows[0]]

        string_nll = self._loglikelihood_tokens_verbose(
            requests=rolling_token_windows,
            disable_tqdm=True,
            override_bs=adaptive_batch_size,
            verbose_token_logits=verbose_token_logits,
        )

        if (self.world_size > 1) and (pad_amnt > 0):
            string_nll = string_nll[:-pad_amnt]

        for i, request in enumerate(requests):
            # TODO: Make this a utility function to share with other model types
            model_output: Dict[str, Any] = {
                "sum_logits": 0,
                "num_tokens": 0,
                "num_tokens_all": 0,
            }
            if verbose_token_logits:
                model_output["tokens"] = []
                model_output["logits"] = []

            target_text = self._mask_text(request.context, request.masked_context_spans)
            model_output["num_chars"] = len(target_text)
            model_output["num_words"] = len(re.split(r"\s+", target_text))
            model_output["num_bytes"] = len(target_text.encode("utf-8"))

            request_nll = [
                string_nll[j]
                for j in range(len(rolling_token_windows))
                if window_request_ids[j] == i
            ]
            for result in request_nll:
                model_output["sum_logits"] += result["sum_logits"]
                model_output["num_tokens"] += result["num_tokens"]
                model_output["num_tokens_all"] += result["num_tokens_all"]
                if verbose_token_logits:
                    model_output["tokens"] += result["tokens"]
                    model_output["logits"] += result["logits"]
            loglikelihoods.append(model_output)

        return loglikelihoods

    def loglikelihood_verbose(
        self,
        requests: List[LoglikelihoodRequest],
        override_bs: Optional[int] = None,
        disable_tqdm: bool = False,
    ) -> List[dict]:
        new_reqs = []
        for req in requests:
            context = cast(str, req.context)
            continuation = req.continuation
            if context == "":
                # BOS or EOS as context
                context_enc, continuation_enc = (
                    [self.prefix_token_id],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
            new_reqs.append(((context, continuation), context_enc, continuation_enc))
        return self._loglikelihood_tokens_verbose(
            new_reqs, disable_tqdm=disable_tqdm, override_bs=override_bs
        )

    def _loglikelihood_tokens_verbose(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
        verbose_token_logits: bool = False,
    ) -> List[dict]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        verbose_res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by=(
                "contexts"
                if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM and self.logits_cache
                else None
            ),
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs if override_bs is not None else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and n_reordered_requests > 0 and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = 0
            padding_len_cont = 0
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # NOTE(BOLMO): add newlines to the end to ensure one-byte-lookahead acts as expected (not clear if necessary)
                continuation_enc = continuation_enc + [self.newline_id, self.newline_id]

                # when too long to fit in context, truncate from the left
                if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape
                elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                    inp = torch.tensor(
                        (context_enc)[-self.max_length :],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape

                    # build encoder attn masks
                    encoder_attns.append(torch.ones_like(inp))

                    cont = torch.tensor(
                        (continuation_enc)[-self.max_length :],
                        # TODO: left-shift these?
                        # TODO: our code assumes we never end up truncating conts for either model type
                        dtype=torch.long,
                        device=self.device,
                    )
                    (contlen,) = cont.shape

                    conts.append(cont)

                    padding_len_cont = (
                        max(padding_len_cont, contlen) if padding_len_cont is not None else contlen
                    )

                padding_len_inp = (
                    max(padding_len_inp, inplen) if padding_len_inp is not None else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                batched_inps = pad_and_concat(
                    padding_len_inp, inps, padding_side="right"
                )  # [batch, padding_len_inp]
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                # TODO: left-pad encoder inps and mask?
                batched_inps = pad_and_concat(padding_len_inp, inps)  # [batch, padding_len_inp]
                batched_conts = pad_and_concat(padding_len_cont, conts)  # [batch, padding_len_cont]
                batched_encoder_mask = pad_and_concat(
                    padding_len_inp, encoder_attns
                )  # [batch, padding_len_inp]
                call_kwargs = {
                    "attn_mask": batched_encoder_mask,
                    "labels": batched_conts,
                }

            # This is a hack to get the model to work with the masked context spans
            # since masked tokens will have negative input ids
            # it does not change the behavior in the default case
            batched_inps, _ = self._unmask_tokens(batched_inps)

            # NOTE(BOLMO): forced boundaries, built after `_unmask_tokens` so the ids are
            # real bytes rather than the negative placeholders used for masked spans.
            if self.boundary_source is not None:
                boundary_mask = self._build_boundary_mask(batched_inps, inplens)
                call_kwargs["boundary_mask"] = boundary_mask
                if self.boundary_dump is not None:
                    self._dump_boundaries(chunk, boundary_mask, inplens)

            # Show the first direct input into the model
            if not hasattr(self, "_shown_model_input"):
                print("First batch of loglikelihood requests:")
                print(batched_inps)
                self._shown_model_input = True

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, **call_kwargs), dim=-1
            )  # [batch, padding_length (inp or cont), vocab]

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                # NOTE(BOLMO): remove the newline padding
                cont_toks = cont_toks[: len(cont_toks) - 2]
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                    if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
                    else None
                )
                # NOTE(BOLMO): remove \n padding
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len - 2)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # NOTE(BOLMO): replace plain (single byte) logits with byte + boundary logits
                probs = F.softmax(logits.float(), dim=-1)
                probs[..., :self.vocab_size_bolmo] += probs[..., self.vocab_size_bolmo:self.vocab_size_bolmo*2]
                logits = torch.log(probs)
                logits[..., self.vocab_size_bolmo:self.vocab_size_bolmo*2] = -100_000

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(
                        0
                    )  # [1, seq]
                    cont_toks, target_mask = self._unmask_tokens(cont_toks)
                    max_equal = (greedy_tokens == cont_toks)[target_mask].all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits[target_mask].sum()), bool(max_equal))

                    verbose_answer: Dict[str, Any] = {
                        "sum_logits": answer[0],
                        "num_tokens": len(cont_toks[target_mask]),
                        "num_tokens_all": len(ctx_tokens) + len(cont_toks[0]),
                        "is_greedy": answer[1],
                    }
                    if verbose_token_logits:
                        instance_tokens = [
                            self.tok_decode(x, skip_special_tokens=False) for x in cont_toks[0]
                        ]
                        verbose_answer["tokens"] = instance_tokens
                        verbose_answer["logits"] = logits[0].tolist()
                    verbose_res.append(verbose_answer)

                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(verbose_res)

    # arguments = (ctx, deepcopy(self.config.generation_kwargs))
    def generate_until_verbose(
        self, requests: List[GenerateUntilRequest], disable_tqdm: bool = False
    ) -> List[dict]:
        # Convert to request_args used in original lm_eval generate_until for minimal diff
        request_args: List[Tuple[str, dict]] = []
        for request in requests:
            kwargs = request.generation_kwargs
            if kwargs is None:
                kwargs = {}
            else:
                kwargs = kwargs.copy()
            kwargs["until"] = request.stop_sequences
            request_args.append((cast(str, request.context), kwargs))
        res = []
        assert isinstance(request_args[0][0], str), request_args[0]

        def _collate(req: Tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            assert isinstance(req[0], str), req
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(request_args),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
            # Adds a new line after the progress bar so that logs look better.
            bar_format="{l_bar}{bar}{r_bar}\n",
        )
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size if adaptive_batch_size is not None else 0
        )
        batch_fn = (
            self._batch_scheduler if self.batch_size == "auto" and not adaptive_batch_size else None
        )

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        re_ords = Collator(
            request_args,
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        context_length_warning = False
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            # add EOS token to stop sequences
            eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks
            truncate_context = kwargs.pop("truncate_context", True)
            min_acceptable_gen_toks = kwargs.pop("min_acceptable_gen_toks", 0)

            # set the max length in tokens of inputs ("context_enc")
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                if truncate_context:
                    # max len for inputs = max length, minus room to generate the max new tokens
                    max_ctx_len = self.max_length - max_gen_toks
                    if max_ctx_len <= 0:
                        raise ValueError(
                            f"max_gen_toks ({max_gen_toks}) is greater than max_length ({self.max_length}), "
                            + '"consider using "truncate_context": False'
                        )
                else:
                    # If truncate_context is False we modify max_gen_toks instead, to keep full context
                    max_ctx_len = self.max_length
                    # The first context is the longest because of ordering above
                    max_batch_toks = len(self.tok_encode(contexts[0]))
                    new_max_gen_toks = min(max_gen_toks, self.max_length - max_batch_toks)
                    if new_max_gen_toks < max_gen_toks and not context_length_warning:
                        eval_logger.warning(
                            f"Running with effective max_gen_toks of {new_max_gen_toks} instead of {max_gen_toks}!"
                        )
                        context_length_warning = True
                    max_gen_toks = max(new_max_gen_toks, 0)
                    if max_gen_toks == 0:
                        if min_acceptable_gen_toks > 0:
                            # We allow desperation context truncation here, after all
                            max_ctx_len -= min_acceptable_gen_toks
                            max_gen_toks = min_acceptable_gen_toks
                        else:
                            raise ValueError(
                                f"Effective max_gen_toks is 0 (max_batch_toks = {max_batch_toks}, max_ctx_len = {max_ctx_len}) "
                                + '"consider using "truncate_context": True'
                            )
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                # max len for inputs = encoder's whole max_length
                max_ctx_len = self.max_length

            # encode, pad, and truncate contexts for this batch
            context_enc, attn_masks = self.tok_batch_encode(
                list(contexts),  # oe-eval: https://github.com/huggingface/tokenizers/issues/1672
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            context_enc = context_enc.to(self.device)
            attn_masks = attn_masks.to(self.device)

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            # perform batched generation, with args for logit scores
            kwargs["output_scores"] = True
            kwargs["return_dict_in_generate"] = True
            output = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                stop=until,
                **kwargs,
            )
            # Extract generated sequences
            cont_toks_list = []
            for out, context_enc in zip(output.tolist(), context_enc.tolist()):
                out_nopad = [x for x in out if x != self.tokenizer.pad_token_id]
                context_enc_nopad = [x for x in context_enc if x != self.tokenizer.pad_token_id]
                cont_toks = out_nopad[len(context_enc_nopad) :]
                cont_toks_list.append(cont_toks)

            for cont_toks, context in zip(
                cont_toks_list, contexts
            ):
                s_raw = self.tok_decode(cont_toks)
                s = cut_at_stop_sequence(s_raw, until)
                no_pad_len = len(cont_toks)
                for idx, tok in enumerate(cont_toks):
                    # Keep the first eot_token encountered
                    if tok == self.eot_token_id:
                        no_pad_len = idx + 1
                        break
                    if tok == self.tokenizer.pad_token_id:
                        no_pad_len = idx
                        break
                cont_toks_no_pad = cont_toks[:no_pad_len]

                res1 = {
                    "continuation": s,
                    "sum_logits": 0,  # not supported
                    "num_tokens": len(cont_toks_no_pad),
                    # "tokens": cont_toks_no_pad,
                    # "logits": logits.tolist(),
                }
                if s_raw != s:
                    res1["continuation_raw"] = s_raw

                res.append(res1)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        return res

    def generate_until_and_loglikelihood_verbose(
        self, instances: List[RequestInstance], requests: List[GenerateUntilAndLoglikelihoodRequest]
    ):
        """
        seperate and copy arguments into both request types. run independently, then combine results here.
        """
        results_for_requests: List[dict] = []

        # Copy request context for perplexity
        requests_ll: List[GenerateUntilAndLoglikelihoodRequest] = copy.deepcopy(requests)
        for request in requests_ll:
            if hasattr(request, "perplexity_context"):
                request.context = request.perplexity_context

        loglikelihood_requests = [
            LoglikelihoodRequest(context=req.context, continuation=req.continuation)
            for req in requests_ll
        ]
        output = self.loglikelihood_verbose(requests=loglikelihood_requests)
        results_ll = collate_results(instances, output)

        generate_until_requests = [
            GenerateUntilRequest(
                context=req.context,
                stop_sequences=req.stop_sequences,
                generation_kwargs=req.generation_kwargs,
            )
            for req in requests
        ]
        output = self.generate_until_verbose(requests=generate_until_requests)
        results_generate = collate_results(instances, output)

        results_for_requests = []
        for ll_result, generate_result in zip(results_ll, results_generate):
            # These two lists will have different "request" and "model_resps" fields
            result = ll_result

            # Copy the unique request types
            result["model_resps_loglikelihood"] = ll_result["model_resps"]
            result["model_resps_generate"] = generate_result["model_resps"]
            result["request_loglikelihood"] = ll_result["request"]
            result["request_generate"] = generate_result["request"]

            # The final result is a mix of both results
            result["model_resps"] = {
                # Generated continuation
                "continuation": result["model_resps_generate"]["continuation"],
                # Logits under the provided correct continuation
                "sum_logits": result["model_resps_loglikelihood"]["sum_logits"],
                "num_tokens": result["model_resps_loglikelihood"]["num_tokens"],
                "num_tokens_all": result["model_resps_loglikelihood"]["num_tokens_all"],
            }

            results_for_requests += [result]

        return results_for_requests

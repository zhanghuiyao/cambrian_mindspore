import os
import re
import re
import copy
import json
import logging
import pathlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from PIL import Image

import mindspore as ms
from mindspore import Tensor

from transformers import PreTrainedTokenizer

from cambrian import conversation as conversation_lib
from cambrian.mm_utils import tokenizer_image_token, tokenizer_image_token_llama3
from cambrian.constants import (
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_TOKEN_INDEX
)


def make_supervised_data_module(tokenizer: PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator_kwargs = {
        'tokenizer': tokenizer,
    }

    if hasattr(data_args, 'image_token_len'):
        data_collator_kwargs['image_token_len'] = data_args.image_token_len

    if hasattr(data_args, 'vision_tower_aux_token_len_list'):
        data_collator_kwargs['image_aux_token_len_list'] = data_args.vision_tower_aux_token_len_list
    else:
        data_collator_kwargs['image_aux_token_len_list'] = [data_args.image_token_len]

    if hasattr(data_args, 'image_position'):
        data_collator_kwargs['image_position'] = data_args.image_position

    data_collator = DataCollatorForSupervisedDataset(**data_collator_kwargs)

    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)



# 1. Sampler
class LengthGroupedSampler:
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    if generator is not None:
        raise NotImplementedError
    megabatch_indices = np.random.permutation(len(megabatches))
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    if generator is not None:
        raise NotImplementedError
    indices = np.random.permutation(len(lengths))
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]





# 2. Dataset

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    image_position: int = 35  # depends on v1 conv


class LazySupervisedDataset:
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            data_path: str,
            tokenizer: PreTrainedTokenizer,
            data_args: DataArguments
    ):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.data_path = data_path
        self.data_args = data_args
        self.length = self._get_length()

        print(f"Default conversation version: {conversation_lib.default_conversation.version}")

    def _get_length(self):
        """Calculates the number of samples in the .jsonl file."""
        with open(self.data_path, 'r') as file:
            for i, _ in enumerate(file):
                pass
        return i + 1

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.length

    def _compute_lengths(self):
        """Compute and cache lengths of conversations in the dataset."""
        if hasattr(self, 'length_list') and hasattr(self, 'modality_length_list'):
            # Return cached values if already computed
            return self.length_list, self.modality_length_list

        self.length_list = []
        self.modality_length_list = []
        with open(self.data_path, 'r') as file:
            for line in file:
                sample = json.loads(line.strip())
                img_tokens = self.data_args.image_token_len if self._has_image(sample) else 0
                cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
                self.length_list.append(cur_len + img_tokens)
                modality_len = cur_len if 'image' in sample else -cur_len
                self.modality_length_list.append(modality_len)
        return self.length_list, self.modality_length_list

    @property
    def lengths(self):
        length_list, _ = self._compute_lengths()
        return length_list

    @property
    def modality_lengths(self):
        _, modality_length_list = self._compute_lengths()
        return modality_length_list

    def _has_image(self, sample: dict) -> bool:
        return "image" in sample and not str(sample['image']) in ['', 'None', 'none', 'nan']

    def __getitem__(self, i) -> Dict[str, Tensor]:
        # sources = self.list_data_dict[i]

        with open(self.data_path, 'r') as file:
            for idx, line in enumerate(file):
                if idx == i:
                    sources = json.loads(line.strip())
                    break
        dat = sources
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        has_image = self._has_image(dat)
        if has_image:
            image_file = dat['image']
            image_folder = self.data_args.image_folder
            processor_aux_list = self.data_args.image_processor_aux_list
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            except Exception as e:
                print("WARNING: LazySupervisedDataset, got invalid image, skip...")
                print(f"\tException msg: {e}")
                return self.__getitem__(0)
            image_size = image.size

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    # result.paste(pil_img, (0, 0))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    # result.paste(pil_img, (0, 0))
                    return result

            if self.data_args.image_aspect_ratio != 'pad':
                raise NotImplementedError("Only pad is supported for now.")

            image_aux_list = []
            for processor_aux in processor_aux_list:
                image_aux = image
                target_resolution = processor_aux.crop_size['height']
                image_aux = expand2square(image_aux, tuple(int(x * 255) for x in processor_aux.image_mean)).resize(
                    (target_resolution, target_resolution))
                image_aux = processor_aux.preprocess(image_aux)['pixel_values'][0]
                image_aux_list.append(image_aux)

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=has_image)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        if (data_dict['labels'] != IGNORE_INDEX).sum() == 0:
            print("WARNING: LazySupervisedDataset, got non labels, skip...")
            return self.__getitem__(0)
        # image exist in the data
        if has_image:
            data_dict['image_aux_list'] = image_aux_list
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = 336
            processor_aux_list = self.data_args.image_processor_aux_list
            data_dict['image_aux_list'] = [
                Tensor(np.zeros((3, processor_aux.crop_size['height'], processor_aux.crop_size['width'])))
                for processor_aux in processor_aux_list
            ]
            image_size = (crop_size, crop_size)
        data_dict['image_size'] = image_size
        return data_dict


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="np",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        np.not_equal(tokenized.input_ids, tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def preprocess_multimodal(
    sources,
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_3(
        sources,
        tokenizer: PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        prompt = conv.get_prompt()
        if prompt.endswith("<|start_header_id|>assistant<|end_header_id|>"):
            prompt = prompt[:-len("<|start_header_id|>assistant<|end_header_id|>")]
        conversations.append(prompt)

    # Tokenize conversations

    if has_image:
        input_ids = np.stack(
            [tokenizer_image_token_llama3(prompt, tokenizer, return_tensors="np") for prompt in conversations], axis=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="np",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids[:]

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets
    sep = "<|eot_id|>"
    for conversation, target in zip(conversations, targets):

        total_len = int(np.not_equal(target, tokenizer.pad_token_id).sum())

        rounds = conversation.split("<|eot_id|>")

        cur_len = 0

        for i, rou in enumerate(rounds):
            if rou == "":
                break

            rou += sep

            # System Prompt
            if i == 0:
                round_len = len(tokenizer(rou).input_ids)
                # Don't predict system prompt
                target[cur_len: cur_len + round_len] = IGNORE_INDEX
                cur_len += round_len
            # User Prompt
            elif i % 2 == 1:
                if i == 1 and has_image:
                    round_len = len(tokenizer_image_token_llama3(rou, tokenizer))
                else:
                    round_len = len(tokenizer(rou).input_ids)
                # Don't predict system prompt
                target[cur_len: cur_len + round_len] = IGNORE_INDEX
                cur_len += round_len
            # Model Reponse
            elif i % 2 == 0:
                round_len = len(tokenizer(rou).input_ids)
                # Don't predict system prompt
                target[cur_len: cur_len + 3] = IGNORE_INDEX
                cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
        sources: Sequence[str],
        tokenizer: PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        raise NotImplementedError
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        raise NotImplementedError
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
        return preprocess_llama_3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        raise NotImplementedError
    if conversation_lib.default_conversation.version == "mpt":
        raise NotImplementedError
    if conversation_lib.default_conversation.version == "phi3":
        raise NotImplementedError

    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer, return_tensors="np")) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="np") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)





# 3. Collactor

@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer
    image_token_len: int
    image_aux_token_len_list: list
    image_position: int
    return_tensor: bool = True

    def __call__(self, instances: Sequence[Dict], batch_info) -> Dict[str, np.ndarray]:

        image_token_len = self.image_token_len
        image_aux_token_len_list = self.image_aux_token_len_list
        image_position = self.image_position

        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        max_length = self.tokenizer.model_max_length

        padding_side = self.tokenizer.padding_side

        # print_rank0("Pad token id is", self.tokenizer.pad_token_id)

        if padding_side == "left":
            input_ids = [t[:max_length] if t.shape[0] >= max_length else np.pad(t, (max_length - t.shape[0], 0), 'constant', constant_values=self.tokenizer.pad_token_id) for t in input_ids]
            labels = [t[:max_length] if t.shape[0] >= max_length else np.pad(t, ( max_length - t.shape[0], 0), 'constant', constant_values=IGNORE_INDEX) for t in labels]
        else:
            input_ids = [t[:max_length] if t.shape[0] >= max_length else np.pad(t, (0, max_length - t.shape[0]), 'constant', constant_values=self.tokenizer.pad_token_id) for t in input_ids]
            labels = [t[:max_length] if t.shape[0] >= max_length else np.pad(t, (0, max_length - t.shape[0]), 'constant', constant_values=IGNORE_INDEX) for t in labels]

        input_ids = np.stack(input_ids)
        labels = np.stack(labels)
        attention_mask = np.not_equal(input_ids, self.tokenizer.pad_token_id)
        # insert dummy image
        for i in range(len(input_ids)):
            if (input_ids[i] == IMAGE_TOKEN_INDEX).sum() == 0:
                cur_input_ids_tmp = input_ids[i][:]
                cur_input_ids_tmp[image_position+1:] = input_ids[i, image_position:-1]
                cur_input_ids_tmp[image_position] = IMAGE_TOKEN_INDEX
                input_ids[i] = cur_input_ids_tmp

                cur_labels_tmp = labels[i][:]
                cur_labels_tmp[image_position+1:] = labels[i, image_position:-1]
                cur_labels_tmp[image_position] = IGNORE_INDEX
                labels[i] = cur_labels_tmp

                cur_attention_mask_tmp = attention_mask[i][:]
                cur_attention_mask_tmp[image_position+1:] = attention_mask[i, image_position:-1]
                cur_attention_mask_tmp[image_position] = False
                attention_mask[i] = cur_attention_mask_tmp
        image_sizes = [instance['image_size'] for instance in instances]
        new_input_ids, new_labels, new_attention_mask, new_position_ids, im_aux_attention_masks_list = \
            prepare_multimodal_data(input_ids, labels, attention_mask, image_sizes, image_token_len, image_aux_token_len_list, max_length)
        batch = dict(
            input_ids=new_input_ids,
            labels=new_labels,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            image_aux_attention_masks_list=im_aux_attention_masks_list
        )

        if 'image_aux_list' in instances[0]:
            image_aux_list = [instance['image_aux_list'] for instance in instances]
            image_aux_list = [list(batch_image_aux) for batch_image_aux in zip(*image_aux_list)]
            if all(x is not None and x.shape == image_aux_list[0][0].shape for x in image_aux_list[0]):
                batch['images'] = [np.stack(image_aux) for image_aux in image_aux_list]
            else:
                batch['images'] = image_aux_list

        if self.return_tensor:
            batch = {k: Tensor(v) if v is not None else v for k, v in batch.items() }

        return batch


def get_padding_offset(cur_size, original_size):
    cur_w, cur_h = cur_size
    original_w, original_h = original_size

    original_aspect_ratio = original_w / original_h
    current_aspect_ratio = cur_w / cur_h

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = cur_w / original_w
        new_height = int(original_h * scale_factor)
        padding = (cur_h - new_height) // 2
        return 0, 0, padding, padding
    else:
        scale_factor = cur_h / original_h
        new_width = int(original_w * scale_factor)
        padding = (cur_w - new_width) // 2
        return padding, padding, 0, 0


def prepare_image_info(image_size, image_token_len, newline=False):
    num_tokens_per_side = int(image_token_len**0.5)
    if newline:
        # for the newline embedding
        attention_mask = np.ones(num_tokens_per_side, num_tokens_per_side+1, dtype=np.bool)
    else:
        attention_mask = np.ones(num_tokens_per_side, num_tokens_per_side, dtype=np.bool)
    left_offset, right_offset, top_offset, bottom_offset = get_padding_offset((num_tokens_per_side, num_tokens_per_side), image_size)
    if newline:
        if left_offset > 0:
            attention_mask[:, :left_offset] = 0
        if right_offset > 0:
            attention_mask[:, -right_offset-1:-1] = 0
        if top_offset > 0:
            attention_mask[:top_offset, :]=0
        if bottom_offset > 0:
            attention_mask[-bottom_offset:, :] = 0
    else:
        if left_offset > 0:
            attention_mask[:, :left_offset] = 0
        if right_offset > 0:
            attention_mask[:, -right_offset:] = 0
        if top_offset > 0:
            attention_mask[:top_offset, :]=0
        if bottom_offset > 0:
            attention_mask[-bottom_offset:, :] = 0
    attention_mask = attention_mask.flatten()
    position_ids = attention_mask.cumsum(0) - 1
    return attention_mask, position_ids


def prepare_multimodal_data(input_ids, labels, attention_mask, image_sizes, image_token_len=576,
                            image_aux_token_len_list=[192 * 192], max_length=2048):
    input_ids_im_replaced = []
    labels_im_replaced = []
    attention_mask_im_replaced = []
    position_ids_im_replaced = []
    im_aux_attention_masks_list = [[] for _ in range(len(image_aux_token_len_list))]
    base_image_token_len_per_side = int(image_token_len ** 0.5)
    image_aux_token_len_per_side_list = [int(image_aux_token_len_per_side ** 0.5) for image_aux_token_len_per_side in
                                         image_aux_token_len_list]
    # insert the padding tokens to the places of image so we can embed them together
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        assert num_images == 1, f"num_images: {num_images}"
        image_size = image_sizes[batch_idx]

        image_token_indices = [-1] + np.nonzero(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
            cur_input_ids.shape[0]]

        cur_input_ids_im_replaced = []
        cur_labels_im_replaced = []
        cur_attention_mask_im_replaced = []
        cur_position_ids_im_replaced = []

        cur_labels = labels[batch_idx]
        cur_attention_mask = attention_mask[batch_idx]
        index = 0
        for i in range(len(image_token_indices) - 1):
            # still keep the first image token in input_ids for further use
            cur_input_ids_im_replaced.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1] + 1])
            cur_labels_im_replaced.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            cur_attention_mask_im_replaced.append(
                cur_attention_mask[image_token_indices[i] + 1:image_token_indices[i + 1]])
            cur_position_ids_im_replaced.append(
                np.arange(index, index + image_token_indices[i + 1] - (image_token_indices[i] + 1), dtype=np.int32))
            index += image_token_indices[i + 1] - (image_token_indices[i] + 1)

            if i < len(image_token_indices) - 2:
                num_tokens_per_side = int(image_token_len ** 0.5)
                image_token_len_with_newline = image_token_len + num_tokens_per_side
                cur_input_ids_im_replaced.append(
                    np.full((image_token_len_with_newline - 1,), 0, dtype=cur_input_ids.dtype))
                cur_labels_im_replaced.append(
                    np.full((image_token_len_with_newline,), IGNORE_INDEX, dtype=cur_labels.dtype))

                cur_im_attention_mask, cur_im_position_ids = prepare_image_info(image_size, image_token_len,
                                                                                newline=True)

                for aux_i, image_aux_token_len_per_side in enumerate(image_aux_token_len_per_side_list):
                    assert image_aux_token_len_per_side >= base_image_token_len_per_side
                    num_base_crops_per_aux_side = image_aux_token_len_per_side // base_image_token_len_per_side

                    cur_im_aux_attention_mask, _ = prepare_image_info(image_size, image_aux_token_len_per_side ** 2)
                    cur_im_aux_attention_mask = cur_im_aux_attention_mask.reshape((base_image_token_len_per_side,
                                                                                   num_base_crops_per_aux_side,
                                                                                   base_image_token_len_per_side,
                                                                                   num_base_crops_per_aux_side))
                    _new_shape = (base_image_token_len_per_side * base_image_token_len_per_side,
                                  num_base_crops_per_aux_side * num_base_crops_per_aux_side)
                    cur_im_aux_attention_mask = cur_im_aux_attention_mask.transpose(0, 2, 1, 3).reshape(_new_shape)
                    cur_im_aux_attention_mask[cur_im_aux_attention_mask.sum(axis=1) == 0] = True
                    im_aux_attention_masks_list[aux_i].append(cur_im_aux_attention_mask)
                cur_im_position_ids += index

                if cur_attention_mask[image_token_indices[i + 1]]:
                    cur_attention_mask_im_replaced.append(cur_im_attention_mask)
                    cur_position_ids_im_replaced.append(cur_im_position_ids.astype(np.int32))
                    index = cur_im_position_ids.max() + 1
                else:
                    num_tokens_per_side = int(image_token_len ** 0.5)
                    image_token_len_with_newline = image_token_len + num_tokens_per_side
                    cur_attention_mask_im_replaced.append(
                        np.full((image_token_len_with_newline,), 0, dtype=cur_attention_mask.dtype))
                    cur_position_ids_im_replaced.append(
                        np.full((image_token_len_with_newline,), 0, dtype=np.int32))

        input_ids_im_replaced.append(np.concatenate(cur_input_ids_im_replaced, axis=0))
        labels_im_replaced.append(np.concatenate(cur_labels_im_replaced, axis=0))
        attention_mask_im_replaced.append(np.concatenate(cur_attention_mask_im_replaced, axis=0))
        position_ids_im_replaced.append(np.concatenate(cur_position_ids_im_replaced, axis=0))

    # Truncate sequences to max length as image embeddings can make the sequence longer
    new_input_ids = [x[0:max_length] for x in input_ids_im_replaced]
    new_labels = [x[0:max_length] for x in labels_im_replaced]
    new_attention_mask = [x[0:max_length] for x in attention_mask_im_replaced]
    new_position_ids = [x[0:max_length] for x in position_ids_im_replaced]
    new_input_ids = np.stack(new_input_ids)
    new_labels = np.stack(new_labels)
    new_attention_mask = np.stack(new_attention_mask)
    new_position_ids = np.stack(new_position_ids)
    im_aux_attention_masks_list = [np.stack(im_aux_attention_masks) for im_aux_attention_masks in
                                   im_aux_attention_masks_list]
    return new_input_ids, new_labels, new_attention_mask, new_position_ids, im_aux_attention_masks_list


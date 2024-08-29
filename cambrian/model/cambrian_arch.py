import math
import numpy as np
from abc import abstractmethod

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

from ezcolorlog import root_logger as logger

from cambrian.model.multimodal_encoder.builder import build_vision_tower_aux_list
from cambrian.model.multimodal_projector.builder import build_vision_projector
from cambrian.model.vision_sampler import VisionTokenSampler
from cambrian.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN




class CambrianMetaModel:

    def __init__(self, config):
        super(CambrianMetaModel, self).__init__(config)

        if not hasattr(self, "dtype"):
            self.dtype = ms.float32

        if hasattr(config, "mm_vision_tower_aux_list"):

            projector_type = getattr(config, 'mm_projector_type', 'linear')
            if projector_type == 'sva':

                vision_hidden_size = config.vision_hidden_size
                num_query_group = config.num_query_group
                query_num_list = config.query_num_list
                connector_only = config.connector_only
                connector_depth = config.connector_depth
                vision_tower_aux_list = build_vision_tower_aux_list(config)
                self.vision_tower_aux_list = nn.CellList(vision_tower_aux_list)
                self.mm_projector = nn.SequentialCell([
                    nn.Dense(vision_hidden_size * num_query_group, config.hidden_size),
                    nn.GELU(approximate=False),
                    nn.Dense(config.hidden_size, config.hidden_size)
                ])

                image_token_len = config.image_token_len
                vision_tower_aux_token_len_list = self.config.mm_vision_tower_aux_token_len_list
                cross_att_token_len_list = [int(vision_tower_aux_token_len ** 0.5) // int(image_token_len ** 0.5) for
                                            vision_tower_aux_token_len in vision_tower_aux_token_len_list]

                mm_projector_auxes = []
                for aux_i, vision_tower_aux in enumerate(self.vision_tower_aux_list):
                    mm_projector_auxes.append(
                        nn.SequentialCell([
                            nn.Dense(vision_tower_aux.hidden_size, vision_hidden_size),
                            nn.GELU(approximate=False),
                            nn.Dense(vision_hidden_size, vision_hidden_size),
                            nn.LayerNorm([vision_hidden_size])
                        ])
                    )
                self.mm_projector_auxes = nn.CellList(mm_projector_auxes)

                vision_samplers = []
                for query_group_i in range(num_query_group):
                    cross_att_token_len_list = [
                        int(vision_tower_aux_token_len ** 0.5) // int(query_num_list[query_group_i] ** 0.5) for
                        vision_tower_aux_token_len in vision_tower_aux_token_len_list]
                    vision_samplers.append(
                        VisionTokenSampler(
                            vision_hidden_size,
                            vision_hidden_size,
                            [vision_hidden_size] * len(self.vision_tower_aux_list),
                            cross_att_token_len_list,
                            vision_hidden_size,
                            connector_depth
                        )
                    )
                self.vision_samplers = nn.CellList(vision_samplers)

                if not connector_only:
                    num_of_vision_sampler_layers = config.num_of_vision_sampler_layers = config.num_of_vision_sampler_layers
                    config.start_of_vision_sampler_layers = config.start_of_vision_sampler_layers
                    config.stride_of_vision_sampler_layers = config.stride_of_vision_sampler_layers
                    cross_att_token_len_list = [int(vision_tower_aux_token_len ** 0.5) // int(image_token_len ** 0.5)
                                                for vision_tower_aux_token_len in vision_tower_aux_token_len_list]
                    self.vision_sampler_layers = nn.CellList([
                        VisionTokenSampler(
                            config.hidden_size,
                            vision_hidden_size,
                            [vision_hidden_size] * len(self.vision_tower_aux_list),
                            cross_att_token_len_list,
                            vision_hidden_size,
                            1
                        ) for _ in range(0, num_of_vision_sampler_layers)
                    ])

                # vision_embed_std = 1 / math.sqrt(vision_hidden_size)
                self.vision_query = Parameter(
                    Tensor(np.random.randn(num_query_group, vision_hidden_size), dtype=self.dtype),
                    name="vision_query"
                )

                # embed_std = 1 / math.sqrt(config.hidden_size)
                self.image_newline = Parameter(
                    Tensor(np.zeros(config.hidden_size), dtype=self.dtype),
                    name="image_newline"
                )

            else:
                vision_tower_aux_list = build_vision_tower_aux_list(config)
                self.vision_tower_aux_list = nn.CellList(vision_tower_aux_list)
                config.mm_hidden_size = sum(
                    [vision_tower_aux.hidden_size for vision_tower_aux in self.vision_tower_aux_list])
                self.mm_projector = build_vision_projector(config)

                # embed_std = 1 / math.sqrt(config.hidden_size)
                # self.image_newline = Parameter(
                #     Tensor(np.random.randn(config.hidden_size) * embed_std, dtype=self.dtype),
                #     name="image_newline"
                # )
                self.image_newline = Parameter(
                    Tensor(np.zeros(config.hidden_size), dtype=self.dtype),
                    name="image_newline"
                )

    def get_vision_tower_aux_list(self):
        return self.vision_tower_aux_list

    def initialize_vision_modules(self, model_args):
        # vision_tower = model_args.vision_tower
        num_query_group = model_args.num_query_group
        query_num_list = model_args.query_num_list
        vision_hidden_size = model_args.vision_hidden_size
        vision_tower_aux_list = model_args.vision_tower_aux_list
        vision_tower_aux_token_len_list = model_args.vision_tower_aux_token_len_list
        image_token_len = model_args.image_token_len
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        connector_only = model_args.connector_only
        connector_depth = model_args.connector_depth

        # self.config.mm_vision_tower = vision_tower
        self.config.image_token_len = image_token_len
        self.config.num_query_group = num_query_group
        self.config.query_num_list = query_num_list
        assert num_query_group == len(query_num_list)
        self.config.connector_depth = connector_depth
        self.config.mm_vision_tower_aux_list = vision_tower_aux_list
        self.config.mm_vision_tower_aux_token_len_list = vision_tower_aux_token_len_list
        self.config.connector_only = connector_only

        if self.get_vision_tower_aux_list() is None:
            raise AttributeError("please build module on init function..")
        else:
            vision_tower_aux_list = self.vision_tower_aux_list
            for vision_tower_aux in vision_tower_aux_list:
                vision_tower_aux.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.vision_hidden_size = vision_hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            raise AttributeError("please build module on init function..")
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.get_parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_param_dict = ms.load_checkpoint(pretrain_mm_mlp_adapter)

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword + '.' in k}

            ms.load_param_into_net(self.mm_projector, get_w(mm_projector_param_dict, 'mm_projector'), strict_load=True)

            if self.config.mm_projector_type == 'sva':
                for aux_i in range(len(vision_tower_aux_list)):
                    # getattr(self, 'mm_projector_aux_{}'.format(aux_i)).load_state_dict(
                    #     get_w(mm_projector_weights, 'mm_projector_aux_{}'.format(aux_i)), strict=True)
                    ms.load_param_into_net(
                        self.mm_projector_auxes[aux_i], get_w(mm_projector_param_dict, f'mm_projector_aux_{aux_i}'),
                        strict_load=True
                    )

                for query_group_i in range(num_query_group):
                    # getattr(self, "vision_sampler_{}".format(query_group_i)).load_state_dict(
                    #     get_w(mm_projector_weights, "vision_sampler_{}".format(query_group_i)), strict=True)
                    ms.load_param_into_net(
                        self.vision_samplers[query_group_i], get_w(mm_projector_param_dict, f"vision_sampler_{query_group_i}"),
                        strict_load=True
                    )

                if not connector_only:
                    # self.vision_sampler_layers.load_state_dict(get_w(mm_projector_weights, 'vision_sampler_layers'),
                    #                                            strict=True)
                    ms.load_param_into_net(
                        self.vision_sampler_layers, get_w(mm_projector_param_dict, 'vision_sampler_layers'),
                        strict_load=True
                    )

                self.vision_query.data = mm_projector_param_dict['model.vision_query']

            self.image_newline.data = mm_projector_param_dict['model.image_newline']


def unmask_attention_mask(mask, original_size):
    original_w, original_h = original_size
    cur_h, cur_w = mask.shape[1:3]

    original_aspect_ratio = original_w / original_h
    current_aspect_ratio = cur_w / cur_h

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = cur_w / original_w
        new_height = int(original_h * scale_factor)
        padding = (cur_h - new_height) // 2
        if padding > 0:
            mask[:, :padding, :] = 0
            mask[:, -padding:, :] = 0
        return mask
    else:
        scale_factor = cur_h / original_h
        new_width = int(original_w * scale_factor)
        padding = (cur_w - new_width) // 2
        if padding > 0:
            mask[:, :, :padding] = 0
            mask[:, :, -padding:] = 0
        return mask


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:3]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class CambrianMetaForCausalLM:

    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    # def get_vision_tower(self):
    #     return self.get_model().get_vision_tower()

    def get_vision_tower_aux_list(self):
        return self.model.get_vision_tower_aux_list()

    def rearrange_vision_tower_features_train(self, vision_tower_aux_feature_list, vision_tower_aux_attention_masks_list, query_side_len):
        vision_tower_aux_feature_rearranged_list = ()
        vision_tower_aux_attention_masks_rearranged_list = ()
        bs = vision_tower_aux_feature_list[0].shape[0]
        for vision_tower_aux_feature, vision_tower_aux_attention_masks in zip(vision_tower_aux_feature_list, vision_tower_aux_attention_masks_list):

            vision_tower_aux_feature = ops.depend(vision_tower_aux_feature, vision_tower_aux_feature_rearranged_list) # FIXME: zhy_test depend

            aux_height = aux_width = int(vision_tower_aux_feature.shape[1]**0.5)
            # assert aux_height * aux_width == vision_tower_aux_feature.shape[1]
            # assert (aux_height//query_side_len) * query_side_len == aux_height

            reduce_factor = (aux_height//query_side_len)
            vision_tower_aux_feature_rearranged = vision_tower_aux_feature.view(bs, query_side_len, reduce_factor, query_side_len, reduce_factor, -1)
            vision_tower_aux_feature_rearranged = vision_tower_aux_feature_rearranged.permute(0, 1, 3, 2, 4, 5).flatten(start_dim=0, end_dim=2).flatten(start_dim=1, end_dim=2)

            vision_tower_aux_attention_masks_rearranged = vision_tower_aux_attention_masks.view(bs * query_side_len * query_side_len, reduce_factor * reduce_factor)

            vision_tower_aux_feature_rearranged_list += (vision_tower_aux_feature_rearranged,)
            vision_tower_aux_attention_masks_rearranged_list += (vision_tower_aux_attention_masks_rearranged,)
        return vision_tower_aux_feature_rearranged_list, vision_tower_aux_attention_masks_rearranged_list

    def rearrange_vision_tower_features_inference(self, vision_tower_aux_feature_list, query_side_len, image_sizes, unpad=False):
        vision_tower_aux_feature_rearranged_list = ()
        vision_tower_aux_attention_masks_rearranged_list = ()
        bs = vision_tower_aux_feature_list[0].shape[0]
        for vision_tower_aux_feature in vision_tower_aux_feature_list:
            vision_tower_aux_feature = ops.depend(vision_tower_aux_feature, vision_tower_aux_feature_rearranged_list)  # FIXME: zhy_test depend

            aux_height = aux_width = int(vision_tower_aux_feature.shape[1]**0.5)
            # assert (aux_height//query_side_len) * query_side_len == aux_height

            reduce_factor = (aux_height//query_side_len)

            vision_tower_aux_feature_rearranged = ()
            vision_tower_aux_attention_masks_rearranged = ()
            for batch_i in range(bs):
                image_size = image_sizes[batch_i]
                cur_vision_tower_aux_feature = vision_tower_aux_feature[batch_i]

                cur_vision_tower_aux_attention_masks_rearranged = ops.ones((1, aux_height, aux_width), dtype=ms.bool_)
                cur_vision_tower_aux_feature_rearranged = cur_vision_tower_aux_feature.view(1, query_side_len, reduce_factor, query_side_len, reduce_factor, -1)
                cur_vision_tower_aux_feature_rearranged = cur_vision_tower_aux_feature_rearranged.permute(0, 1, 3, 2, 4, 5)
                if unpad:
                    cur_vision_tower_aux_feature_rearranged = unpad_image(cur_vision_tower_aux_feature_rearranged, image_size)
                cur_vision_tower_aux_feature_rearranged = cur_vision_tower_aux_feature_rearranged.flatten(start_dim=0, end_dim=2).flatten(start_dim=1, end_dim=2) # query_side_len*query_side_len X reduce_factor*reduce_factor X C

                cur_vision_tower_aux_attention_masks_rearranged = unmask_attention_mask(cur_vision_tower_aux_attention_masks_rearranged, image_size)
                cur_vision_tower_aux_attention_masks_rearranged = cur_vision_tower_aux_attention_masks_rearranged.view(1, query_side_len, reduce_factor, query_side_len, reduce_factor).permute(0, 1, 3, 2, 4)
                if unpad:
                    cur_vision_tower_aux_attention_masks_rearranged = unpad_image(cur_vision_tower_aux_attention_masks_rearranged, image_size)
                cur_vision_tower_aux_attention_masks_rearranged = cur_vision_tower_aux_attention_masks_rearranged.flatten(start_dim=0, end_dim=2).flatten(start_dim=1, end_dim=2)

                _mask = cur_vision_tower_aux_attention_masks_rearranged.sum(-1) == 0
                cur_vision_tower_aux_attention_masks_rearranged = ops.masked_fill(
                    cur_vision_tower_aux_attention_masks_rearranged,
                    _mask[:, None],
                    True
                )

                vision_tower_aux_feature_rearranged += (cur_vision_tower_aux_feature_rearranged,)
                vision_tower_aux_attention_masks_rearranged += (cur_vision_tower_aux_attention_masks_rearranged,)

            vision_tower_aux_feature_rearranged = ops.cat(vision_tower_aux_feature_rearranged, 0)
            vision_tower_aux_attention_masks_rearranged = ops.cat(vision_tower_aux_attention_masks_rearranged, 0)


            vision_tower_aux_feature_rearranged_list += (vision_tower_aux_feature_rearranged,)
            vision_tower_aux_attention_masks_rearranged_list += (vision_tower_aux_attention_masks_rearranged,)

        return vision_tower_aux_feature_rearranged_list, vision_tower_aux_attention_masks_rearranged_list

    def encode_images(self, image_aux_list):
        vision_tower_aux_list = self.model.vision_tower_aux_list  # .get_vision_tower_aux_list()
        image_aux_features_list = ()
        for image_aux, vision_tower_aux in zip(image_aux_list, vision_tower_aux_list):
            image_aux = ops.depend(image_aux, image_aux_features_list)  # FIXME: zhy_test depend
            image_aux_features = vision_tower_aux(image_aux)
            image_aux_features_list += (image_aux_features,)
        return image_aux_features_list

    # @ms.jit  # zhy_test infer
    def prepare_inputs_labels_for_multimodal_(
            self, input_ids, position_ids, attention_mask, labels, *images_and_sizes
    ):
        # visual_num = len(self.model.vision_tower_aux_list)
        # assert len(images_and_sizes) == visual_num + 1
        images, image_aux_attention_masks_list, image_sizes = \
            images_and_sizes[:-1], None, images_and_sizes[-1:]

        return self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, labels,
                                                         images, image_aux_attention_masks_list, image_sizes)

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, labels,
        images, image_aux_attention_masks_list=None, image_sizes=None
    ):
        vision_tower_aux_list = self.model.vision_tower_aux_list  #.get_vision_tower_aux_list()

        # assert vision_tower_aux_list is not None
        # assert images is not None
        # assert input_ids.shape[1] != 1
        # if vision_tower_aux_list is None or images is None or input_ids.shape[1] == 1:
        #     return input_ids, position_ids, attention_mask, None, labels, None, None, None, None

        image_aux_list = images

        bs = image_aux_list[0].shape[0]
        dtype = image_aux_list[0].dtype

        image_token_len = self.model.image_token_len
        query_num_list = self.model.query_num_list

        final_height = final_width = int(image_token_len**0.5)
        # assert final_width * final_height == image_token_len

        final_image_features_list = []

        # only needed for sva
        vision_tower_aux_feature_list_final = None
        vision_tower_aux_attention_masks_list_final = None
        global_context_feature_final = None

        # breakpoint() # point 1, before encoder images

        # zhy_test infer, breakpoint()
        image_aux_features_list = self.encode_images(image_aux_list)
        np.save("image_aux_features_list_0.npy", image_aux_features_list[0].asnumpy())
        np.save("image_aux_features_list_1.npy", image_aux_features_list[1].asnumpy())
        np.save("image_aux_features_list_2.npy", image_aux_features_list[2].asnumpy())
        np.save("image_aux_features_list_3.npy", image_aux_features_list[3].asnumpy())

        _dtype = image_aux_features_list[0].dtype
        image_aux_features_list = [
            Tensor(np.load("./pt_tensors/image_aux_features_list_0.npy"), _dtype),
            Tensor(np.load("./pt_tensors/image_aux_features_list_1.npy"), _dtype),
            Tensor(np.load("./pt_tensors/image_aux_features_list_2.npy"), _dtype),
            Tensor(np.load("./pt_tensors/image_aux_features_list_3.npy"), _dtype),
        ]

        vision_tower_aux_feature_list = ()
        vision_tower_aux_attention_masks_list = ()
        global_context_feature = None
        if self.model.mm_projector_type == 'sva':
            # get vision tokens from each vision tower
            for aux_i in range(len(vision_tower_aux_list)):
                image_aux_features = image_aux_features_list[aux_i]
                image_aux_features = ops.depend(image_aux_features, vision_tower_aux_feature_list) # FIXME: zhy_test depend

                # image_aux_features = getattr(self.model, 'mm_projector_aux_{}'.format(aux_i))(image_aux_features).to(dtype)
                image_aux_features = self.model.mm_projector_auxes[aux_i](image_aux_features).to(dtype)
                if aux_i == 0:
                    global_context_feature = image_aux_features.mean(1).view(bs, 1, 1, -1)

                vision_tower_aux_feature_list += (image_aux_features,)

            # breakpoint()  # point 2, after mm_projector_aux

            # perform vision sampling for each query group
            for query_group_i, query_num in enumerate(query_num_list):
                query_features_i = self.model.vision_query[query_group_i, :].view(1, 1, 1, -1).broadcast_to((bs, query_num, -1, -1))
                query_features_i = ops.depend(query_features_i, final_image_features_list)  # FIXME: zhy_test depend
                global_context_feature_i = global_context_feature.broadcast_to((-1, query_num, 1, -1)).flatten(start_dim=0, end_dim=1)
                query_side_len = int(query_num**0.5)

                if self.training:
                    vision_tower_aux_feature_list_i, vision_tower_aux_attention_masks_list_i = \
                        self.rearrange_vision_tower_features_train(vision_tower_aux_feature_list, image_aux_attention_masks_list, query_side_len)
                else:
                    vision_tower_aux_feature_list_i, vision_tower_aux_attention_masks_list_i = \
                        self.rearrange_vision_tower_features_inference(vision_tower_aux_feature_list, query_side_len, image_sizes)

                # query_features_i = getattr(self.model, "vision_sampler_{}".format(query_group_i))(query_features_i.flatten(start_dim=0, end_dim=1), global_context_feature_i, *vision_tower_aux_feature_list_i, *vision_tower_aux_attention_masks_list_i)

                # breakpoint()  # point 3.1, before vision_sampler

                query_features_i = self.model.vision_samplers[query_group_i](
                    query_features_i.flatten(start_dim=0, end_dim=1),
                    global_context_feature_i,
                    *vision_tower_aux_feature_list_i,
                    *vision_tower_aux_attention_masks_list_i
                )

                query_features_i = query_features_i.view(bs, query_num, -1)
                # interpolate to the final target size
                if query_side_len != final_height:

                    # breakpoint()  # point 3.2, before interpolate

                    query_features_i = query_features_i.permute(0, 2, 1).view(bs, -1, query_side_len, query_side_len)
                    query_features_i = ops.interpolate(
                        query_features_i.to(ms.float32),
                        size=(final_height, final_width),
                        mode='bilinear',
                        align_corners=False
                    ).to(dtype=query_features_i.dtype)
                    query_features_i = query_features_i.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
                final_image_features_list.append(query_features_i)

            # breakpoint()  # point 3, after vision_sampler

            if self.training:
                vision_tower_aux_feature_list_final, vision_tower_aux_attention_masks_list_final = \
                    self.rearrange_vision_tower_features_train(vision_tower_aux_feature_list, image_aux_attention_masks_list, final_height)
                global_context_feature_final = global_context_feature.broadcast_to((-1, final_height*final_width, 1, -1)).flatten(start_dim=0, end_dim=1)
        else:
            final_image_features_list = image_aux_features_list

        image_features = ops.cat(final_image_features_list, -1)
        image_features = self.model.mm_projector(image_features).to(dtype)

        # breakpoint()  # point 4, after mm_projector

        if self.training:
            image_features = image_features.view(image_features.shape[0], final_height, final_width, -1)
            image_features = ops.cat((
                image_features,
                self.model.image_newline[None, None, None, :].broadcast_to((image_features.shape[0], final_height, 1, -1)).to(image_features.dtype)
            ), axis=2)
            image_features = image_features.flatten(start_dim=1, end_dim=2)
            final_size = ((final_height, final_width),) * bs
        else:
            image_features = image_features.view(bs, final_height, final_width, -1)
            image_features_unpadded = []
            final_size = ()
            if self.model.mm_projector_type == 'sva':
                vision_tower_aux_feature_list_final, vision_tower_aux_attention_masks_list_final = \
                    self.rearrange_vision_tower_features_inference(vision_tower_aux_feature_list, final_height, image_sizes, unpad=True)
                global_context_feature_final = []
            for batch_i in range(bs):
                cur_image_feature = image_features[batch_i]
                image_size = image_sizes[batch_i]

                cur_image_feature = unpad_image(cur_image_feature.unsqueeze(0), image_size)

                cur_h, cur_w = cur_image_feature.shape[1:3]
                final_size += ((cur_h, cur_w),)
                cur_image_feature = cur_image_feature.view(1, cur_h, cur_w, -1)
                cur_image_feature = ops.cat(
                    (cur_image_feature,
                     self.model.image_newline.view(1, 1, 1, -1).broadcast_to((1, cur_h, 1, -1)).to(cur_image_feature.dtype)),
                    axis=2
                )
                cur_image_feature = cur_image_feature.flatten(start_dim=1, end_dim=2)
                image_features_unpadded.append(cur_image_feature.squeeze(0))

                if self.model.mm_projector_type == 'sva':
                    cur_global_context_feature = global_context_feature[batch_i].broadcast_to((cur_h*cur_w, 1, -1))
                    global_context_feature_final.append(cur_global_context_feature)
            if self.model.mm_projector_type == 'sva':
                global_context_feature_final = ops.cat(global_context_feature_final, 0)

            image_features = image_features_unpadded

        # breakpoint()  # point 5, after concat

        # TODO: image start / end is not implemented here to support pretraining.
        # if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
        #     raise NotImplementedError

        if self.training:

            # embed the input_ids
            new_input_ids_padded_for_emb = ops.where(input_ids == IMAGE_TOKEN_INDEX, 0, input_ids)

            # FIXME: level 0, avoid gather_op out of bounds
            new_input_ids_padded_for_emb = ops.clip(
                new_input_ids_padded_for_emb,
                min=0,
                max=self.model.embed_tokens.embedding_table.shape[0] - 1
            )

            input_embeds = self.model.embed_tokens(new_input_ids_padded_for_emb)

            new_input_embeds = []
            cur_image_idx = 0

            # insert the image embeddings
            for batch_idx in range(input_embeds.shape[0]):
                cur_input_embeds = input_embeds[batch_idx]
                cur_input_ids = input_ids[batch_idx]
                ori_input_embeds = cur_input_embeds[:]

                num_images = ops.equal(cur_input_ids, IMAGE_TOKEN_INDEX).to(ms.int32).sum()
                has_images = ops.greater(num_images, 0)
                # if num_images == 0:
                #     cur_image_idx += 1
                #     new_input_embeds.append(cur_input_embeds)
                #     continue

                # 1 image
                cur_image_idx = batch_idx
                cur_image_features = image_features[cur_image_idx]
                _index_table = ops.range(0, len(cur_input_ids), 1)
                _index_table = ops.masked_fill(_index_table,
                                               cur_input_ids != IMAGE_TOKEN_INDEX,
                                               ops.full((), -1, dtype=_index_table.dtype))
                _img_indexes_topk = ops.topk(_index_table, 1)[0]
                _img_indexes = ops.range(0, len(cur_image_features), 1) + _img_indexes_topk[0]

                cur_input_embeds[_img_indexes, :] = cur_image_features
                new_input_embed = cur_input_embeds

                new_input_embed = ops.select(has_images, new_input_embed, ori_input_embeds)

                new_input_embeds.append(new_input_embed)

                # # n images
                # image_token_indices = [-1] + torch.nonzero(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
                # cur_input_embeds_im_replaced = []
                # prev_image_length = 0
                # for i in range(len(image_token_indices) - 1):
                #     # skip the image tokens (1 indicator + (image_length-1) paddings)
                #
                #     # FIXME: prev_image_length = 0 or len(input_ids == IMAGE_TOKEN_INDEX)-1 ?
                #     cur_input_embeds_im_replaced.append(cur_input_embeds[image_token_indices[i]+1+prev_image_length:image_token_indices[i+1]])
                #     if i < len(image_token_indices) - 2:
                #         cur_image_features = image_features[cur_image_idx]
                #         prev_image_length = len(cur_image_features)-1
                #         cur_image_idx += 1
                #         cur_input_embeds_im_replaced.append(cur_image_features)
                #
                # cur_input_embeds_im_replaced = [x.to(self.device) for x in cur_input_embeds_im_replaced]
                # new_input_embeds.append(torch.cat(cur_input_embeds_im_replaced))

            new_input_embeds = ops.stack(new_input_embeds)
            return None, position_ids, attention_mask, new_input_embeds, labels, \
                   vision_tower_aux_feature_list_final, vision_tower_aux_attention_masks_list_final, \
                   final_size, global_context_feature_final

        else:
            # Let's just add dummy tensors if they do not exist,
            # it is a headache to deal with None all the time.
            # But it is not ideal, and if you have a better idea,
            # please open an issue / submit a PR, thanks.
            _labels = labels
            _position_ids = position_ids
            if attention_mask is None:
                attention_mask = ops.ones(input_ids, dtype=ms.bool_)
            else:
                attention_mask = attention_mask.to(ms.bool_)
            if labels is None:
                labels = ops.full_like(input_ids, IGNORE_INDEX)

            new_input_embeds = []
            new_attention_masks = []
            new_labels = []
            new_position_ids = []
            # assert len(image_features) == len(input_ids)
            for batch_idx, cur_input_ids in enumerate(input_ids):

                cur_attention_mask = attention_mask[batch_idx]

                num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

                if num_images == 0:
                    # FIXME: level 0, avoid gather_op out of bounds
                    cur_input_ids = ops.clip(
                        cur_input_ids,
                        min=0,
                        max=self.model.embed_tokens.embedding_table.shape[0] - 1
                    )

                    cur_input_embeds = self.model.embed_tokens(cur_input_ids)
                    new_input_embed = cur_input_embeds
                    new_label = labels[batch_idx]
                    new_position_id = ops.arange(0, cur_input_ids.shape[0], 1, dtype=cur_input_ids.dtype)

                    neg_cur_attention_mask = ops.logical_not(cur_attention_mask)
                    new_input_embed = ops.masked_fill(new_input_embed, neg_cur_attention_mask[:, None], ops.full((), 0, dtype=new_input_embed.dtype))
                    new_label = ops.masked_fill(new_label, neg_cur_attention_mask, ops.full((), IGNORE_INDEX, dtype=new_label.dtype))
                    new_position_id = ops.masked_fill(new_position_id, neg_cur_attention_mask, ops.full((), 0, dtype=new_position_id.dtype))

                    new_input_embeds.append(new_input_embed)
                    new_attention_masks.append(cur_attention_mask)
                    new_labels.append(new_label)
                    new_position_ids.append(new_position_id)

                    continue

                # 1 img
                cur_image_features = image_features[batch_idx]
                _index_table = ops.arange(0, cur_input_ids.shape[0], 1, dtype=ms.int32)
                _im_positions = ops.masked_fill(_index_table, cur_input_ids != IMAGE_TOKEN_INDEX, ops.full((), -1, dtype=ms.int32))
                _im_positions = ops.topk(_im_positions, 1)[0]
                _im_token_len = cur_image_features.shape[0]

                # when tokenizer_padding_side == "right"
                gather_index = ops.select(
                    _index_table < _im_positions, _index_table, 0)
                gather_index = ops.select(
                    _index_table >= (_im_positions + _im_token_len + 1), _index_table - _im_token_len + 1, gather_index)
                gather_index = ops.clip(gather_index, min=0, max=cur_input_ids.shape[0] - 1)
                cur_input_ids = ops.gather(cur_input_ids, gather_index, axis=0)
                cur_labels = ops.gather(labels[batch_idx], gather_index, axis=0)
                cur_attention_mask = ops.gather(cur_attention_mask.to(ms.int32), gather_index, axis=0).to(ms.bool_)

                # FIXME: level 0, avoid gather_op out of bounds
                cur_input_ids = ops.clip(
                    cur_input_ids,
                    min=0,
                    max=self.model.embed_tokens.embedding_table.shape[0] - 1
                )
                cur_input_embeds = self.model.embed_tokens(cur_input_ids)
                _img_indexes = ops.arange(0, _im_token_len, 1, dtype=ms.int32) + _im_positions
                # _img_indexes = ops.broadcast_to(_img_indexes[:, None], (-1, cur_image_features.shape[-1]))
                # new_input_embed = ops.scatter(cur_input_embeds, 0, _img_indexes, cur_image_features.to(cur_input_embeds.dtype))
                cur_input_embeds[_img_indexes] = cur_image_features.to(cur_input_embeds.dtype)
                new_input_embed = cur_input_embeds
                new_label = ops.scatter(cur_labels, 0, _img_indexes, ops.full((_im_token_len,), IGNORE_INDEX, dtype=cur_labels.dtype))
                new_position_id = ops.arange(0, cur_input_ids.shape[0], 1, dtype=ms.int32)

                neg_cur_attention_mask = ops.logical_not(cur_attention_mask)
                new_input_embed = ops.masked_fill(new_input_embed, neg_cur_attention_mask[:, None], ops.full((), 0, dtype=new_input_embed.dtype))
                new_label = ops.masked_fill(new_label, neg_cur_attention_mask, ops.full((), IGNORE_INDEX, dtype=new_label.dtype))
                new_position_id = ops.masked_fill(new_position_id, neg_cur_attention_mask, ops.full((), 0, dtype=new_position_id.dtype))

                new_input_embeds.append(new_input_embed)
                new_attention_masks.append(cur_attention_mask)
                new_labels.append(new_label)
                new_position_ids.append(new_position_id.to(ms.int32))

            new_input_embeds = ops.stack(new_input_embeds, axis=0)
            new_attention_masks = ops.stack(new_attention_masks, axis=0)
            new_labels = ops.stack(new_labels, axis=0)
            new_position_ids = ops.stack(new_position_ids, axis=0)

            input_embeds = new_input_embeds
            attention_mask = new_attention_masks
            labels = new_labels if _labels is not None else None
            position_ids = new_position_ids if _position_ids is not None else None

            # breakpoint()  # point 7, before return

            return None, position_ids, attention_mask, input_embeds, labels, \
                   vision_tower_aux_feature_list_final, vision_tower_aux_attention_masks_list_final, \
                   final_size, global_context_feature_final

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings_p = self.get_input_embeddings().embedding_table
                output_embeddings_p = self.get_output_embeddings().embedding_table

                input_embeddings = input_embeddings_p.data.asnumpy()
                output_embeddings = output_embeddings_p.data.asnumpy()

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    axis=0, keep_dims=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    axis=0, keep_dims=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

                input_embeddings_p.set_data(Tensor(input_embeddings, input_embeddings_p.dtype))
                output_embeddings_p.set_data(Tensor(output_embeddings, output_embeddings_p.dtype))

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().get_parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().get_parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = ms.load_checkpoint(model_args.pretrain_mm_mlp_adapter)
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

                input_embeddings_p.set_data(Tensor(input_embeddings, input_embeddings_p.dtype))
                output_embeddings_p.set_data(Tensor(output_embeddings, output_embeddings_p.dtype))

        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().get_parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().get_parameters():
                    p.requires_grad = False

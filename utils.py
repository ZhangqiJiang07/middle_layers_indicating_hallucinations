import cv2
import random
import inflect
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from typing import List

engine = inflect.engine()


def setup_seeds():
    seed = 927

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    Copied from llava.utils
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def set_act_get_hooks(model, attn_out=False):
    '''
    Modified from:
    https://github.com/google-research/google-research/blob/master/dissecting_factual_predictions/factual_associations_dissection.ipynb
    '''
    
    for attr in ["activations_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})
        else:
            model.activations_ = {}

    def get_activation(name):
        def hook(module, input, output):
            if "attn_out" in name:
                model.activations_[name] = output[0].squeeze(0).detach() if name not in model.activations_ else torch.cat(
                    [model.activations_[name], output[0].squeeze(0).detach()], dim=0
                )
                # print(f"Hooked {name} with shape {model.activations_[name].shape}")

        return hook
    
    hooks = []
    for i in range(model.config.num_hidden_layers):
        if attn_out is True:
            hooks.append(model.layers[i].self_attn.register_forward_hook(get_activation(f"attn_out_{i}")))

    return hooks


# Always remove your hooks, otherwise things will get messy.
def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def get_only_attn_out_contribution(
        model, tokenizer, outputs, text: str, output_start_idx: int
):
    ''' Get the Attn. Sublayer contribution of the selected object token to the final prediction.
    Params:
    -------
    model: the language component of the LVLMs
    tokenizer: the tokenizer of the model
    outputs: dict
        the outputs of the model containing the 'output sequence' and 'scores' attributes
    text: str
        real or hallucinated text
    output_start_idx: int (= input length - 1)

    Return:
    -------
    records_attn: list
        the attention contribution over layers
    '''
    selected_token_id = tokenizer(text, add_special_tokens=False)["input_ids"][0]
    # the first index is adoptted if there are multiple occurrences
    token_in_generation_idx = torch.nonzero(outputs['sequences'][0][1:] == selected_token_id)[0].item()
    final_probs = F.softmax(outputs['scores'][token_in_generation_idx], dim=-1)
    _, topk_token_ids = final_probs.topk(1)
    topk_token_ids = topk_token_ids[0]

    linear_projector = model.lm_head # llava1.5

    records_attn = []
    for layer_i in range(model.model.config.num_hidden_layers):
        # ATTN
        attn_out = (
            model.model.activations_[f"attn_out_{layer_i}"][output_start_idx + token_in_generation_idx, :]
            ).clone().detach()
        proj = linear_projector(attn_out)
        attn_logit = proj.cpu().detach().numpy()
        records_attn.append(attn_logit[topk_token_ids])

    return records_attn


def attnw_over_vision_layer_head_selected_text(
        text: str, outputs, tokenizer, vision_token_start, vision_token_end,
        sort_heads=False
):
    ''' Get the attention weights over the image tokens for the selected object text.
    Params:
    -------
    text: str
        the selected object text
    outputs: dict
        the outputs of the model containing the 'attentions' and 'sequences' attributes
    tokenizer: the tokenizer of the model
    vision_token_start/_end: int
        the start/end index of the image tokens

    Return:
    -------
    text_attnw_matrix: np.ndarray (num_layers, num_heads)
        the attention weights over the image tokens for the selected object text over layers and heads
        (bottom row is the 0-th layer, leftmost column is the first head)
    '''
    try:
        selected_token_id = tokenizer(text, add_special_tokens=False)["input_ids"][0]
        # the first index is adoptted if there are multiple occurrences
        token_in_generation_idx = torch.nonzero(outputs['sequences'][0][1:] == selected_token_id)[0].item()
    except:
        text = engine.plural(text)
        selected_token_id = tokenizer(text, add_special_tokens=False)["input_ids"][0]
        token_in_generation_idx = torch.nonzero(outputs['sequences'][0][1:] == selected_token_id)[0].item()

    text_attnw_layers_heads = outputs['attentions'][token_in_generation_idx]
    text_attnw_matrix = torch.zeros((len(text_attnw_layers_heads), text_attnw_layers_heads[0].shape[1]))
    for i, layer_attnw in enumerate(text_attnw_layers_heads):
        for j, head_attnw in enumerate(layer_attnw.squeeze(0)):
            text_attnw_matrix[len(text_attnw_layers_heads) - 1 - i, j] = \
                head_attnw[-1][vision_token_start:vision_token_end].sum().item()

    if sort_heads:
        text_attnw_matrix, _ = torch.sort(text_attnw_matrix, dim=1, descending=True)

    text_attnw_matrix = text_attnw_matrix.numpy()

    return text_attnw_matrix, token_in_generation_idx


def logitLens_of_vision_tokens(
        model, tokenizer, input_ids, outputs, token_range: List[int], layer_range: List[int],
        logits_warper, logits_processor
):
    ''' Retrieve the text token in the vocabulary with the highest prob
        for each image token in selected token index range.
    Params:
    -------
    model: the language component of the LVLMs
    tokenizer: the tokenizer of the model
    input_ids: tensor
        the input sequence of the model
    outputs: dict
        the outputs of the model containing the 'hidden_states' attribute
    token_range: list
        [start_image_token_idx, end_image_token_idx]
    layer_range: list
        the range of layers to be considered

    Returns:
    --------
    layer_max_prob: tensor (len(layer_range), image_token_num)
        the max prob predicted by the linear projector before softmax on the hidden states of each image token
        over the selected layers
    layer_words: list of list
        the retrieved text token for each image token over the selected layers
    '''
    layer_max_prob = torch.zeros((1, token_range[1] - token_range[0]))
    layer_words = []
    for i in layer_range:
        hidden_state = outputs['hidden_states'][0][i + 1].squeeze(0)
        hidden_state = hidden_state[token_range[0]:token_range[1]].clone().detach()
        logits = model.lm_head(hidden_state).cpu().float() # llava1.5
        logits = F.log_softmax(logits, dim=-1)
        logits_processed = logits_processor(input_ids, logits)
        logits = logits_warper(input_ids, logits_processed)

        probs = F.softmax(logits, dim=-1)
        vals, ids = probs.max(dim=-1)
        layer_max_prob = torch.cat([vals.unsqueeze(0).cpu().detach(), layer_max_prob], dim=0)
        layer_words.append([tokenizer.decode(id, skip_special_tokens=True) for id in ids])

    layer_max_prob = layer_max_prob[:-1] # drop the all zero row

    return layer_max_prob, layer_words


def logitLens_of_vision_tokens_with_discrete_range(
        model, tokenizer, input_ids, outputs, vision_token_start: int,
        discrete_range: List[List[int]], layer_range: List[int],
        logits_warper, logits_processor, fig_name: str = None
):
    '''
    Refer to the function `logitLens_of_vision_tokens` for the detailed description.
    '''
    assert(hasattr(outputs, 'hidden_states'))

    vision_discrete_range = [
        [vision_token_start + range_i[0], vision_token_start + range_i[1] + 1] for range_i in discrete_range
    ]

    each_range_layer_prob_list = []
    each_range_layer_words_list = []
    x_ticks = []
    y_ticks = [f'{i} h_out' for i in layer_range]

    for i, token_range in enumerate(vision_discrete_range):
        x_ticks += np.arange(discrete_range[i][0], discrete_range[i][1] + 1).tolist()
        range_layer_max_prob, layer_words = logitLens_of_vision_tokens(
            model, tokenizer, input_ids, outputs,
            token_range, layer_range,
            logits_warper, logits_processor
        )
        each_range_layer_prob_list.append(range_layer_max_prob)
        each_range_layer_words_list.append(layer_words)

    whole_ranges_layer_prob = np.concatenate(each_range_layer_prob_list, axis=1)

    # plot heatmap
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    im = ax.imshow(
        whole_ranges_layer_prob,
        alpha=0.8,
        )

    # annotate text
    range_flag = 0
    for each_range_layer_words in each_range_layer_words_list:
        for layer_i, each_layer_words in enumerate(each_range_layer_words):
            for col_j, word in enumerate(each_layer_words):
                ax.text(
                    range_flag + col_j, len(layer_range) - 1 - layer_i,
                    word, ha='center', va='center', color='w',
                    fontsize=13, rotation=30,
                )
        range_flag += len(each_layer_words)

    ax.set_xlim(0-0.5, len(x_ticks)-0.5)
    ax.set_xticks([i for i in range(len(x_ticks))])
    ax.set_yticks([i for i in range(len(layer_range))])
    ax.set_xticklabels(x_ticks, fontsize=16)
    ax.set_yticklabels(y_ticks, fontsize=16)
    ax.set_xlabel('Image Tokens Index', fontsize=16)
    ax.set_ylabel('Layers', fontsize=16)
    ax.set_title('Logit Lens of Vision Tokens with Discrete Range', fontsize=16)

    if fig_name is not None:
        plt.savefig(f'./{fig_name}.pdf')
    plt.show()


def plot_VAR_heatmap(avg_data, filename=None):
    # sort heads
    sorted_idx = np.argsort(-avg_data, axis=-1)
    avg_data = np.take_along_axis(avg_data, sorted_idx, axis=-1)

    # plot heatmap
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    im = axes.imshow(
        avg_data, vmin=avg_data.min(),
        vmax=avg_data.max(), cmap='Blues'
    )
    n_layer, n_head = avg_data.shape
    y_label_list = [str(i) for i in range(n_layer)]
    axes.set_yticks(np.arange(0, n_layer, 2))
    axes.set_yticklabels(y_label_list[::-1][::2])
    axes.set_xlabel("Sorted Heads")
    axes.set_ylabel("Layers")
    fig.colorbar(im, ax=axes, shrink=0.4, location='bottom')
    plt.xticks([])
    if filename is not None:
        plt.savefig(filename, dpi=400)
    plt.show()


def show_heatmap_over_image_with_interpolation(
    text, layer_id, head_id, outputs, tokenizer, image,
    vision_token_start, vision_token_end, savefig=True
):
    ''' Show heatmap over the image in i-th head at j-th layer
    Params:
    -------
    text: real or hallucinated object text
    layer_id: int
    head_id: int
    outputs: dict
        additionaly need 'attentions' attribute
    tokenizer: the tokenizer of the model
    image: PIL image
    vision_token_start: int
    vision_token_end: int
    '''
    selected_token_id = tokenizer(text, add_special_tokens=False)["input_ids"][0]
    # the first index is adoptted if there are multiple occurrences
    token_in_generation_idx = torch.nonzero(outputs['sequences'][0][1:] == selected_token_id)[0].item()

    # get row attention matrix
    row_attn = outputs['attentions'][token_in_generation_idx][layer_id].squeeze(0)[head_id].cpu().detach()
    visual_row_attn = row_attn[-1, vision_token_start:vision_token_end].to(torch.float32)
    visual_row_attn = visual_row_attn / visual_row_attn.max()

    # resize
    visual_row_attn = visual_row_attn.reshape(24, 24) # for llava-1.5
    # bilinear interpolation
    attn_over_image = cv2.resize(visual_row_attn.numpy(), (image.size[0], image.size[1]))

    def show_mask_on_image(img, mask):
        img = np.float32(img) / 255
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)

    np_img = np.array(image)
    # plt.imshow(np_img)
    img_with_attn = show_mask_on_image(np_img, attn_over_image)
    plt.imshow(img_with_attn)
    # turn off axis
    plt.axis('off')
    if savefig:
        plt.savefig(
            f"{text}_heatmap_layer{layer_id}_head{head_id}.png",
            bbox_inches='tight',
            pad_inches=0
        )
    plt.show()
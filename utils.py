import numpy as np
import random

from typing import List

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt

import inflect
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

        return hook
    
    hooks = []
    for i in range(model.config.num_hidden_layers):
        if attn_out is True:
            hooks.append(model.layers[i].self_attn.register_forward_hook(get_activation(f"attn_out_{i}")))

    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def get_only_attn_out_contribution(
        model, tokenizer, outputs,
        text: str, input_len: int
):
    selected_token_id = tokenizer(text, add_special_tokens=False)["input_ids"][0]
    # the first index is adoptted if there are multiple occurrences
    token_in_generation_idx = torch.nonzero(outputs['sequences'][0][1:] == selected_token_id)[0].item()
    final_hidden_state = outputs['hidden_states'][token_in_generation_idx][-1].squeeze(0).squeeze(0).cpu().detach()
    final_probs = F.softmax(outputs['scores'][token_in_generation_idx], dim=-1)
    topk_probs, topk_token_ids = final_probs.topk(1)
    topk_token_ids = topk_token_ids[0]

    linear_projector = model.lm_head # llava1.5

    records_attn = []

    for layer_i in range(model.model.config.num_hidden_layers):
        # ATTN
        attn_out = model.model.activations_[f"attn_out_{layer_i}"][input_len + token_in_generation_idx, :].clone().cpu().detach()
        proj = linear_projector(attn_out)
        attn_logit = proj.cpu().detach().numpy()
        records_attn.append(attn_logit[topk_token_ids])

    return records_attn


def attnw_over_vision_layer_head_selected_text(
    text: str, outputs, tokenizer, vision_token_start, vision_token_end,
    sort_heads=False
):
    try:
        selected_token_id = tokenizer(text, add_special_tokens=False)["input_ids"][0]
        # the first index is adoptted if there are multiple occurrences
        token_in_generation_idx = torch.nonzero(outputs['sequences'][0][1:] == selected_token_id)[0].item()
    except:
        text = engine.plural(text)
        selected_token_id = tokenizer(text, add_special_tokens=False)["input_ids"][0]
        # the first index is adoptted if there are multiple occurrences
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
        model, tokenizer, input_ids, outputs,
        token_range: List[int], layer_range: List[int],
        logits_warper, logits_processor
):
    layer_max_prob = torch.zeros((1, token_range[1] - token_range[0]))
    layer_words = []
    y_ticks = []
    for i in layer_range:
        hidden_state = outputs['hidden_states'][0][i + 1].squeeze(0)
        hidden_state = hidden_state[token_range[0]:token_range[1]].clone().detach()
        logits = model.lm_head(hidden_state).cpu().float()
        logits = F.log_softmax(logits, dim=-1)
        logits_processed = logits_processor(input_ids, logits)
        logits = logits_warper(input_ids, logits_processed)

        probs = F.softmax(logits, dim=-1)
        vals, ids = probs.max(dim=-1)
        layer_max_prob = torch.cat([vals.unsqueeze(0).cpu().detach(), layer_max_prob], dim=0)
        layer_words.append([tokenizer.decode(id, skip_special_tokens=True) for id in ids])
        y_ticks.append(f'{i} h_out')

    layer_max_prob = layer_max_prob[:-1] # drop the all zero row

    return layer_max_prob, layer_words


def logitLens_of_vision_tokens_with_discrete_range(
        model, tokenizer, input_ids, outputs, vision_token_start: int,
        discrete_range: List[List[int]], layer_range: List[int],
        logits_warper, logits_processor, savefig: bool = False
):
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

    if savefig:
        plt.savefig(f'./logit_lens_of_vision_tokens_with_discrete_range.pdf')
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
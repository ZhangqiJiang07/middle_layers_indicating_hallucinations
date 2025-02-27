'''
This script extracts the VAR features for the real and hallucinated object tokens.
'''

import argparse
import json
import os
from PIL import Image
import pickle

import numpy as np
from scipy import stats
import tqdm

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from llava.mm_utils import process_images

import warnings 
warnings.filterwarnings('ignore')

from model_manager import ModelManager
from utils import setup_seeds, disable_torch_init
from utils import attnw_over_vision_layer_head_selected_text

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='llava-1.5', help="model")
parser.add_argument(
    "--data-path",
    type=str,
    default="/path/to/COCO/val2014/",
    help="data path"
)
parser.add_argument(
    "--annotations-path",
    type=str,
    default="/path/to/COCO/annotations"
)
parser.add_argument(
    "--instruction-path",
    type=str,
    default="./examples/toy_img_query_list.jsonl"
)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--beam", type=int, default=1) # 1 for Greedy Decoding
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--output-file", type=str, default="llava_7b_var_features.pt")
args = parser.parse_known_args()[0]

setup_seeds()
disable_torch_init()

# Load model
model_manager = ModelManager(args.model)

# initialize CHAIR evaluator
from chair import CHAIR
from chair import chair_eval
cache = 'chair.pkl'
if cache and os.path.exists(cache):
    evaluator = pickle.load(open(cache, 'rb'))
    print(f"loaded evaluator from cache: {cache}")
else:
    print(f"cache not setted or not exist yet, building from scratch...")
    evaluator = CHAIR(args.annotations_path)
    pickle.dump(evaluator, open(cache, 'wb'))
    print(f"cached evaluator to: {cache}")

img_query_lists = [
    json.loads(line) for line in open(args.instruction_path)
]
attn_score = []

for img_i, img_query in enumerate(img_query_lists):
    print(f"{img_i+1}/{len(img_query_lists)}")
    # prepare inputs
    img_id = f"COCO_val2014_{str(img_query['image_id']).zfill(12)}.jpg"
    img_path = os.path.join(args.data_path, img_id)
    img = Image.open(img_path).convert('RGB')
    images_tensor = process_images(
                            [img],
                            model_manager.image_processor,
                            model_manager.llm_model.config
                    ).to(model_manager.llm_model.device, dtype=torch.float16)

    query = [img_query['instruction']]
    questions, input_ids, kwargs = model_manager.prepare_inputs_for_model(query, images_tensor, use_dataloader=False)

    # use hooks to get the attention sublayers' output
    with torch.inference_mode():
        outputs = model_manager.llm_model.generate(
            input_ids,
            do_sample=False,
            num_beams=args.beam,
            max_new_tokens=args.max_tokens,
            use_cache=True,
            output_attentions=True,
            return_dict_in_generate=True,
            **kwargs,
        )

    answer = model_manager.tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)[0].strip()
    img_info = chair_eval(evaluator, img_id, answer)
    vision_token_start = model_manager.img_start_idx
    vision_token_end = model_manager.img_end_idx
    gt_words = img_info['mscoco_gt_words']
    generated_words = img_info['mscoco_generated_words']

    # Real words Calculation
    for ri, real_word in enumerate(set(generated_words) & set(gt_words)):
        # calculate attn sublayer contribution for each real word
        try:
            # get visual attention weights
            real_word_attnw_matrix, _ = attnw_over_vision_layer_head_selected_text(
                    real_word, outputs, model_manager.tokenizer,
                    vision_token_start, vision_token_end
            )
            data = {'atten':real_word_attnw_matrix.flatten(), 'label': 1}
            attn_score.append(data)
        except:
            print(f"'{real_word}' not found in the generated text.")

    if len(img_info['mscoco_hallucinated_words']) == 0:
        continue

    # Hallucinated words Calculation
    hallucination_words = [
        item for sublist in img_info['mscoco_hallucinated_words'] for item in sublist
    ]
    for hi, hallu_word in enumerate(set(hallucination_words)):
        # calculate attn sublayer contribution for each hallu word
        try:
            # get visual attention weights
            hallu_word_attnw_matrix, _ = attnw_over_vision_layer_head_selected_text(
                    hallu_word, outputs, model_manager.tokenizer,
                    vision_token_start, vision_token_end
            )
            data = {'atten':hallu_word_attnw_matrix.flatten(), 'label': 0}
            attn_score.append(data)
        except:
            print(f"'{hallu_word}' not found in the generated text.")

# save the attn_score
base_dir = './var_data'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
torch.save(attn_score, os.path.join(base_dir, args.output_file))
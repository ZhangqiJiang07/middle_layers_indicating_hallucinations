import os
from collections import namedtuple

import torch
import yaml
from constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    IMAGE_TOKEN_INDEX,
    IMAGE_TOKEN_LENGTH,
    INSTRUCTION_TEMPLATE,
    SYSTEM_MESSAGE,
    IDEFICS2_IMAGE_TOKEN_LENGTH_WO_SPLIT,
    IDEFICS2_IMAGE_TOKEN_LENGTH_W_SPLIT,
    IDEFICS2_IMAGE_TOKEN_INDEX,
    IDEFICS_FAKE_TOKEN_AROUND_IMAGE,
)


IDEFICS2_IMAGE_SPLITTING = False



def load_llava_model(model_path):
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    # load the model
    load_8bit = False
    load_4bit = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_name = get_model_name_from_path(model_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name,
        load_8bit, load_4bit, device=device
    )

    return tokenizer, model, image_processor, model



def prepare_llava_inputs(template, query, image_tensor, tokenizer):
    qu = [template.replace("<question>", q) for q in query]
    batch_size = len(query)

    chunks = [q.split("<ImageHere>") for q in qu]
    chunk_before = [chunk[0] for chunk in chunks]
    chunk_after = [chunk[1] for chunk in chunks]

    token_before = (
        tokenizer(
            chunk_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    token_after = (
        tokenizer(
            chunk_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    bos = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * tokenizer.bos_token_id
    )

    img_start_idx = len(token_before[0]) + 1
    img_end_idx = img_start_idx + IMAGE_TOKEN_LENGTH
    image_token = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * IMAGE_TOKEN_INDEX
    )

    input_ids = torch.cat([bos, token_before, image_token, token_after], dim=1)
    kwargs = {}

    kwargs["images"] = image_tensor.half()

    return qu, input_ids, img_start_idx, img_end_idx, kwargs





class ModelManager:
    def __init__(self, model_name):
        self.model_name = model_name.lower()
        self.tokenizer = None
        self.vlm_model = None
        self.llm_model = None
        self.image_processor = None
        self.load_model()

    def load_model(self):
        if self.model_name == "llava-1.5":
            model_path = os.path.expanduser("/path/to/LLaVA-1.5")
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_llava_model(model_path)
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
    def construct_template(self):
        if self.model_name == "llava-1.5":
            template = SYSTEM_MESSAGE + " " + INSTRUCTION_TEMPLATE[self.model_name]
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        return template
        
    def prepare_inputs_for_model(self, query, image):
        template = self.construct_template()

        if self.model_name == "llava-1.5":
            from llava.mm_utils import process_images
            images_tensor = process_images(
                                [image],
                                self.image_processor,
                                self.llm_model.config
                            ).to(self.llm_model.device, dtype=torch.float16)
            questions, input_ids, img_start_idx, img_end_idx, kwargs = prepare_llava_inputs(
                template, query, images_tensor, self.tokenizer
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        self.img_start_idx = img_start_idx
        self.img_end_idx = img_end_idx

        return questions, input_ids, kwargs

    def decode(self, output_ids):
        # get outputs
        if self.model_name == "llava-1.5":
            # replace image token by pad token
            output_ids = output_ids.clone()
            output_ids[output_ids == IMAGE_TOKEN_INDEX] = torch.tensor(
                0, dtype=output_ids.dtype, device=output_ids.device
            )

            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            output_text = [text.split("ASSISTANT:")[-1].strip() for text in output_text]
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        return output_text

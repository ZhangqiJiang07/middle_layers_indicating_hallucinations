
# LLaVA-1.5
IMAGE_TOKEN_INDEX = -200
IMAGE_TOKEN_LENGTH = 576
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"

SYSTEM_MESSAGE = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere>\n<question> ASSISTANT:",
    "internvl": "USER: <ImageHere> <question> ASSISTANT:",
    "idefics2": "User:<ImageHere><question><end_of_utterance>\nAssistant:",
}
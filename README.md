# Devils in Middle Layers of Large Vision-Language Models

## Requirements
Take LLaVA-1.5-7B as an example, we conduct our experiments based on:  
&emsp;Python version: 3.10.14  
&emsp;CUDA toolkit version: 12.1  

To install the package in an existing environment, please run
``` bash
pip install -r requirements.txt
```

## Content
- `middle_layers_indicating_hallucination.ipynb`: Code for our main findings.
- `modify_attention.py`: Code for _Heads Guided Attention Intervention method_.
- `model_manager.py`: Load model, prepare inputs and decode output.
- `chair_utils.py`: contains the CHAIR evaluator for sampling real and hallucinated object tokens.

## Usage
To use our attention intervention method, please insert the following code before model inference:
```python
llama_head_guide(
    model=model_manager.llm_model,
    guided_layer_range=guided_layer_range,
    aggregation='mean',
    alpha=alpha,
    img_start_idx=model_manager.img_start_idx,
    img_end_idx=model_manager.img_end_idx
)
```
where the parameter `guided_layer_range` represents the range of the visual information enrichment stage, e.g., 5-18 layers for LLaVA-1.5-7B.

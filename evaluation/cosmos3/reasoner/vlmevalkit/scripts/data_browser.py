"""
pip install gradio    # proxy_on first
python vis_geochat_data.py
# browse data in http://127.0.0.1:10064
"""

import argparse
import base64
import copy
import io
import json
import os
import os.path as osp
import string
import time
from io import BytesIO
from typing import Sequence

import gradio as gr
import pandas as pd
from PIL import Image

from vlmeval.api import OpenAIWrapper
from vlmeval.dataset import SUPPORTED_DATASETS, build_dataset
from vlmeval.smp import LMUDataRoot, encode_image_file_to_base64, load

SYS = "You are a helpful assistant. Your job is to faithfully translate all provided text into Chinese faithfully. "

# Translator = SiliconFlowAPI(model='Qwen/Qwen2.5-7B-Instruct', system_prompt=SYS)
Translator = OpenAIWrapper(model='gpt-4o-mini', system_prompt=SYS)


def image_to_mdstring(image):
    return f"![image](data:image/jpeg;base64,{image})"


def images_to_md(images):
    return '\n\n'.join([image_to_mdstring(image) for image in images])


def mmqa_display(question, target_size=2048):
    question = {k.lower() if len(k) > 1 else k: v for k, v in question.items()}
    keys = list(question.keys())
    keys = [k for k in keys if k not in ['index', 'image']]

    idx = question.pop('index', 'XXX')
    text = f'\n- INDEX: {idx}\n'

    if 'image' in question:
        images = question.pop('image')
        if images[0] == '[' and images[-1] == ']':
            images = eval(images)
        else:
            images = [images]
    else:
        images = question.pop('image_path')
        if images[0] == '[' and images[-1] == ']':
            images = eval(images)
        else:
            images = [images]
        images = [encode_image_file_to_base64(x) for x in images]
    
    qtext = question.pop('question', None)
    if qtext is not None:
        text += f'- QUESTION: {qtext}\n'

    if 'A' in question:
        text += f'- Choices: \n'
        for k in string.ascii_uppercase:
            if k in question:
                text += f'\t-{k}: {question.pop(k)}\n'
    answer = question.pop('answer', None)
    
    for k in question:
        if not pd.isna(question[k]):
            text += f'- {k.upper()}. {question[k]}\n'
    
    if answer is not None:
        text += f'- ANSWER: {answer}\n'

    image_md = images_to_md(images)

    return text, image_md


def parse_args():
    parser = argparse.ArgumentParser()
    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    return args


def gradio_app_vis_dataset(port=7860):
    data, loaded_obj = None, {}

    def btn_submit_click(filename, ann_id):
        if filename not in loaded_obj:
            return filename_change(filename, ann_id)
        nonlocal data
        data_desc = gr.Markdown(f'Visualizing {filename}, {len(data)} samples in total. ')
        if ann_id < 0 or ann_id >= len(data):
            return filename, ann_id, data_desc, gr.Markdown('Invalid Index'), gr.Markdown(f'Index out of range [0, {len(data) - 1}]')
        item = data.iloc[ann_id]
        text, image_md = mmqa_display(item)
        return filename, ann_id, data_desc, image_md, text

    def btn_next_click(filename, ann_id):
        return btn_submit_click(filename, ann_id + 1)

    # def translate_click(anno_en):
    #     return gr.Markdown(Translator.generate(anno_en))

    def filename_change(filename, ann_id):
        nonlocal data, loaded_obj

        def legal_filename(filename):
            LMURoot = LMUDataRoot()
            if filename in SUPPORTED_DATASETS:
                return build_dataset(filename).data
            elif osp.exists(filename):
                data = load(filename)
                assert 'index' in data and 'image' in data
                image_map = {i: image for i, image in zip(data['index'], data['image'])}
                for k, v in image_map.items():
                    if (not isinstance(v, str) or len(v) < 64) and v in image_map:
                        image_map[k] = image_map[v]
                data['image'] = [image_map[k] for k in data['index']]
                return data
            elif osp.exists(osp.join(LMURoot, filename)):
                filename = osp.join(LMURoot, filename)
                return legal_filename(filename)
            else:
                return None

        data = legal_filename(filename)
        if data is None:
            return filename, 0, gr.Markdown(''), gr.Markdown("File not found"), gr.Markdown("File not found")
        
        loaded_obj[filename] = data
        return btn_submit_click(filename, 0)

    with gr.Blocks() as app:
        
        filename = gr.Textbox(
            value='Dataset Name (supported by VLMEvalKit) or TSV FileName (Relative under `LMURoot` or Real Path)', 
            label='Dataset', 
            interactive=True,
            visible=True)
            
        with gr.Row():
            ann_id = gr.Number(0, label='Sample Index (Press Enter)', interactive=True, visible=True)
            btn_next = gr.Button("Next")
            # btn_translate = gr.Button('CN Translate')

        with gr.Row():
            data_desc = gr.Markdown('Dataset Description', label='Dataset Description')
        
        with gr.Row():
            image_output = gr.Markdown('Image PlaceHolder', label='Image Visualization')
            anno_en = gr.Markdown('Image Annotation', label='Image Annotation')
            # anno_cn = gr.Markdown('Image Annotation (Chinese)', label='Image Annotation (Chinese)')

        input_components = [filename, ann_id]
        all_components = [filename, ann_id, data_desc, image_output, anno_en]

        filename.submit(filename_change, input_components, all_components)
        ann_id.submit(btn_submit_click, input_components, all_components)
        btn_next.click(btn_next_click, input_components, all_components)
        # btn_translate.click(translate_click, anno_en, anno_cn)

    # app.launch()
    app.launch(server_name='0.0.0.0', debug=True, show_error=True, server_port=port)


if __name__ == "__main__":
    args = parse_args()
    gradio_app_vis_dataset(port=args.port)


import os
import random
import numpy as np
import torch
import string
import cv2
from PIL import Image
import base64
import io

from flask import Flask, request
from flask import send_file
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS, cross_origin
from collections import OrderedDict
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from daam import trace
from ecco.output import NMF
import ecco
import re
import shutil
import base64
from io import BytesIO
from gevent import pywsgi

import argparse

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# app = Flask(__name__)
# CORS(app, supports_credentials=True)
def parse_layer_name(layer_name, base_module):
    """
    Return pytorch model layer based on layer_name and the model.
    One can register forward hook easily by using the returned module.
    ---
    Input:
        layer_name: string. Use "." to indicates the sub_module level.
            For example, features.denseblock1.denselayer1.conv in densenet.
        base_module: torch.nn.modules. DNN model. If the model is in DataParallel,
            pass model.module.
    Return:
        target_module: torch.nn.modules or None(when not found).
    """
    target_name_list = layer_name.split(".")
    target_name = target_name_list[0]
    for name, module in base_module._modules.items():
        if name == target_name:
            if len(target_name_list) == 1:
                return module
            else:
                next_level_layer = target_name_list[1:]
                next_level_layer = ".".join(next_level_layer)
                return parse_layer_name(next_level_layer, module)
    return None


class HookRecorder:
    """This is the hook for pytorch model.
    It is used to record the value of hidden states.
    """

    def __init__(self, layer_names: list, model, record_mode="aggregate", position='output'):
        '''
        record_mode:
            aggregate: for all the record input, use torch.cat() to aggregate them.
            separate: for all the record input, only append the final result.
        '''
        self.recorder = dict()
        self.layer_names = layer_names
        if isinstance(model, torch.nn.DataParallel):
            self.model = model.module
        else:
            self.model = model
        self.handlers = list()
        self.record_mode = record_mode
        self.position = position

    def _register_hooker(self, name):
        self.recorder[name] = list()
        if self.position == 'output':
            def named_hooker(module, input, output):
                self.recorder[name].append(output)
        else:
            def named_hooker(module, input, output):
                self.recorder[name].append(input)
        return named_hooker

    def register_hookers(self):
        for l_name in self.layer_names:
            module = parse_layer_name(l_name, self.model)
            if module is None:
                raise Exception("Layer not found")
            handler = module.register_forward_hook(self._register_hooker(l_name))
            self.handlers.append(handler)

    def get_result(self):
        result = dict()
        for key in self.recorder:
            if self.record_mode == "aggregate":
                result[key] = torch.cat(self.recorder[key])
            else:
                result[key] = self.recorder[key]
        return result

    def clear_cache(self):
        for key in self.recorder:
            self.recorder[key] = list()

    def remove_handlers(self):
        for i in self.handlers:
            i.remove()
        self.handlers.clear()

    def __del__(self):
        self.remove_handlers()


class DiffusionModelWrapper:
    def __init__(self, device="cuda:0", random_seed=2023):
        self.device = device
        self.random_seed = random_seed

        # control random seed
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.use_deterministic_algorithms(True)

        # initialize diffusion model
        model_id = "/model/stable-diffusion-2-1"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)

        # hook text encoder
        layers_to_hook = []
        for i in range(23):
            layers_to_hook.append('text_model.encoder.layers.' + str(i) + '.mlp.activation_fn')

        self.hooker = HookRecorder(layers_to_hook, self.pipe.text_encoder, 'raw')

    def generate_image(self, prompts, image_id, index,highlight_keywords, attention_multiplier):
        print(prompts)
        # with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        # with trace(self.pipe, prompt=prompts, highlight_key_words=highlight_keywords, highlight_amp_mags=attention_multiplier) as tc:
        with trace(self.pipe) as tc:
            self.hooker.register_hookers()
            torch.manual_seed(self.random_seed)
            # out = self.pipe(prompts)
            with torch.no_grad():
                out = self.pipe(prompts,num_inference_steps=50)
            heat_map_global, attention = tc.compute_global_heat_map()
            #test
            means = attention.mean(dim=(1, 2))  # 在第二和第三维度上计算平均值

            all_attention_list = means.tolist()
            activations = {'encoder': []}
            for k, v in self.hooker.get_result().items():
                activations['encoder'].append(v[0].detach().cpu().numpy())
            activations['encoder'] = np.array(activations['encoder'])
            activations['encoder'] = np.moveaxis(activations['encoder'], 0, 1)
            activations['encoder'] = np.moveaxis(activations['encoder'], 2, 3)
            self.hooker.remove_handlers()
            tokenizer_output = self.pipe.tokenizer(prompts,
                                                padding="max_length",
                                                max_length=self.pipe.tokenizer.model_max_length,
                                                return_tensors="pt")
            n_input_tokens = tokenizer_output['attention_mask'].sum().item()
            token_ids = tokenizer_output['input_ids'][0][:n_input_tokens]
            tokens = [[v.replace('</w>', '') for v in self.pipe.tokenizer.convert_ids_to_tokens(token_ids)]]
            old_tokens = [[v for v in self.pipe.tokenizer.convert_ids_to_tokens(token_ids)]]
            old_token_ids = token_ids.unsqueeze(0)
            old_activations = activations['encoder'].transpose(3, 0, 1, 2)
            tokens = [[]]
            token_ids = [[]]
            activations = {'encoder': []}
            token_index = 0
            while token_index < n_input_tokens:
                sub_token = old_tokens[0][token_index]
                if sub_token == '<|startoftext|>' or sub_token == '<|endoftext|>' or '</w>' in sub_token:
                    tokens[0].append(sub_token.replace('</w>', ''))
                    token_ids[0].append(old_token_ids[0][token_index])
                    activations['encoder'].append(old_activations[token_index])
                    token_index += 1
                else:
                    sub_token_index = token_index
                    tmp_token = ""
                    tmp_attention = np.zeros(old_activations[token_index].shape)
                    while '</w>' not in old_tokens[0][sub_token_index]:
                        tmp_attention += old_activations[sub_token_index]
                        tmp_token += old_tokens[0][sub_token_index]
                        sub_token_index += 1
                    tmp_attention += old_activations[sub_token_index]
                    tmp_token += old_tokens[0][sub_token_index]
                    sub_token_index += 1
                    tokens[0].append(tmp_token.replace('</w>', ''))
                    token_ids[0].append(old_token_ids[0][token_index])
                    activations['encoder'].append(tmp_attention / (sub_token_index - token_index))
                    token_index = sub_token_index
            activations['encoder'] = np.array(activations['encoder']).transpose(1, 2, 3, 0)
            n_input_tokens = len(tokens[0])
            config = {'tokenizer_config': {'token_prefix': '', 'partial_token_prefix': ''}}
            nmf = NMF(activations=activations, n_components=10, n_input_tokens=n_input_tokens, token_ids=token_ids,
                    tokens=tokens, _path=os.path.dirname(ecco.__file__), config=config)
            explanations = nmf.explore(filter_token=True, top_k=5, printJson=True)
            highlighted_images = []
            os.makedirs('./tmp/', exist_ok=True)
            token_importance = {-1: 0, -2: float('inf'), -3: 0}
            new_token_list = []
            for i in range(len(tokens[0])):
                token = tokens[0][i]
                if token in string.punctuation or token == '<|startoftext|>' or token == '<|endoftext|>':
                    continue
                heat_map = heat_map_global.compute_word_heat_map(token, i-1)
                heat_map_mean = heat_map.value.mean().detach().cpu().item()
                heat_map_array = heat_map.value.cpu().numpy()

                token_importance[i] = float(np.sum(heat_map_array))
                new_token_list.append({'token': token, 'attention_score': float(np.sum(heat_map_array))})
                # new_token_dic[token] = float(np.sum(heat_map_array))
                token_importance[-1] += float(np.sum(heat_map_array))
                if float(np.sum(heat_map_array)) > token_importance[-3]:
                    token_importance[-3] = float(np.sum(heat_map_array))
                if float(np.sum(heat_map_array)) < token_importance[-2]:
                    token_importance[-2] = float(np.sum(heat_map_array))
                heat_map_array = cv2.resize(heat_map_array, (768, 768))
                image = np.array(out.images[0])
                heat_map_array[heat_map_array >= heat_map_mean] = 1
                heat_map_array[heat_map_array < heat_map_mean] = 0.2
                heat_map_array *= 255
                heat_map_array = heat_map_array.astype(np.uint8)
                heat_map_array = np.expand_dims(heat_map_array, -1)
                rgba_image = np.concatenate((image, heat_map_array), axis=2)
                highlight_image = Image.fromarray(rgba_image, mode='RGBA')
                highlight_image.save('./tmp/token%d.png' % i)
                highlighted_images.append('./tmp/token%d.png' % i)
            image = out.images[0]
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = buffered.getvalue()
            img_base64 = base64.b64encode(img_str)
            img_base64_string = img_base64.decode('utf-8')
            # out.images[0].save('./tmp/img.png')
            return {'token_explanations': explanations,
                    'token_image_highlight': highlighted_images,
                    'token_importance': token_importance,
                    "token_attention_linking": new_token_list,
                    "all_attention_list" :all_attention_list,
                    'generated_image': "XXX/" + str(index) + ".png",
                    "img_base64": img_base64_string}



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='PromptCharm Backend', description='This is the backend program of PromptCharm')
    parser.add_argument("--seed", default=2024, help="random seed", type=int)
    args = parser.parse_args()

    custom_diffusion_model = DiffusionModelWrapper(random_seed=args.seed)
    app = Flask("Penguin")
    @app.route("/api/v1/predict", methods=['GET', 'POST'])
    @cross_origin()
    def get_prompts():
        data = request.get_json()
        prompts = data['input']
        version = data['version']
        attention = data['attention']
        image_id = data['image_id']
        index = data['index']
        highlight_keywords = data['highlight_keywords']
        attention_multiplier = data['attention_multiplier']
        explanations = custom_diffusion_model.generate_image(prompts,image_id,index,highlight_keywords,attention_multiplier)
        return explanations, 200
    cors = CORS(app, resorces={r'/d/*': {"origins": '*'}})
    api = Api(app)
    print("ok")
    app.run(host='0.0.0.0', port=3000)


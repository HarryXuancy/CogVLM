# -*- encoding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pdb
import torch
from sat.model.mixins import CachedAutoregressiveMixin
from sat.model import AutoModel
import argparse

from utils.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
from utils.models import CogAgentModel, CogVLMModel
class cogvlm_model():

    def __init__(self):
        local_model_path = os.environ['COGVLM_MODEL_PATH']
        self.parse_args = argparse.Namespace(
            max_length=2048,
            top_p=0.4,
            topk=1,
            temperature=0.8,
            chinese=True,
            version='chat_old',
            quant=None,
            from_pretrained='cogagent-vqa',
            local_tokenizer='lmsys/vicuna-7b-v1.5',
            fp16=False,
            bf16=True,
            stream_chat=True)
        
        model, model_args = AutoModel.from_pretrained(
            'cogagent-vqa',
            args=argparse.Namespace(
            deepspeed=None,
            local_rank=0,
            world_size=1,
            model_parralel_size=1,
            mode='inference',
            skip_init=True,
            rank=0,
            use_gpu_initialization=True,
            device='cuda',
            **vars(self.parse_args)
            ),
            home_path = local_model_path,)
        self.model = model.eval()
        language_processor_version = model_args.text_processor_version if 'text_processor_version' in model_args else self.parse_args.version
        tokenizer = llama2_tokenizer("lmsys/vicuna-7b-v1.5", signal_type=language_processor_version)
        self.image_processor = get_image_processor(model_args.eva_args["image_size"][0])
        self.cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None
        self.model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
        self.text_processor_infer = llama2_text_processor_inference(tokenizer, 2048, self.model.image_length)
        


    def CogVLM_inference(self,img_path,question):
        with torch.no_grad():
                history = None
                cache_image = None
                assert img_path is not None
                query = question
                assert query is not None 
                
                try:
                    response, history, cache_image = chat(
                        img_path,
                        self.model,
                        self.text_processor_infer,
                        self.image_processor,
                        query,
                        history=history,
                        cross_img_processor=self.cross_image_processor,
                        image=cache_image,
                        max_length=2048,
                        top_p=0.4,
                        temperature=.8,
                        top_k=1,
                        invalid_slices=self.text_processor_infer.invalid_slices,
                        args=self.parse_args
                        )
                except Exception as e:
                    print(e)

                
                pdb.set_trace()
                return response


hus = cogvlm_model()
pdb.set_trace()
result = hus.CogVLM_inference("/mnt/nfs/xcy/DataSet/Combination_pictures/collection_5.png","Describe the picture.")
print(result)
print(11111111111111111111)
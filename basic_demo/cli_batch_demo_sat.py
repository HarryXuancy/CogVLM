# -*- encoding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize
from sat.model import AutoModel


from utils.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
from utils.models import CogAgentModel, CogVLMModel

def read_png_files_recursively(directory):
    """
    递归地从给定目录及其子目录中读取 .png 文件路径。

    参数:
    directory (str): 根目录路径。

    返回:
    Generator[str]: .png 文件路径的生成器。
    """
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.png'):
                file_path = os.path.join(root, filename)
                yield file_path

def read_png_files_from_directory(directory):
    """
    从给定目录中逐个读取文件路径。

    参数:
    directory (str): 目录路径。

    返回:
    Generator[str]: 文件路径的生成器。
    """
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            if filename.lower().endswith('.png'):
                yield os.path.join(directory, filename)

def write_answer_to_file(answer, file_path):
    """
    将回答写入到指定的文本文件中。

    参数:
    answer (str): 要写入的回答文本。
    file_path (str): 文件路径。
    """
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(answer + '\n')  # 追加回答并在末尾添加换行符

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--chinese", action='store_true', help='Chinese interface')
    parser.add_argument("--version", type=str, default="chat", choices=['chat', 'vqa', 'chat_old', 'base'], help='version of language process. if there is \"text_processor_version\" in model_config.json, this option will be overwritten')
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')

    parser.add_argument("--from_pretrained", type=str, default="cogagent-chat", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--stream_chat", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    args = parser.parse_args()

    # path of cogagent-vqa model
    local_model_path = "/mnt/nfs/hushuai.p/cogvlm"
    # load model
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=rank,
        rank=rank,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
        device='cpu' if args.quant else 'cuda',
        **vars(args)
    ), 
        home_path = local_model_path,
        overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
    model = model.eval()
    from sat.mpu import get_model_parallel_world_size
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    language_processor_version = model_args.text_processor_version if 'text_processor_version' in model_args else args.version
    print("[Language processor version]:", language_processor_version)
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=language_processor_version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None
    
    if args.quant:
        quantize(model, args.quant)
        if torch.cuda.is_available():
            model = model.cuda()


    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

    if args.chinese:
        if rank == 0:
            print('欢迎使用 CogAgent-CLI ，输入图像URL或本地路径读图，继续输入内容对话，clear 重新开始，stop 终止程序')
    else:
        if rank == 0:
            print('Welcome to CogAgent-CLI. Enter an image URL or local file path to load an image. Continue inputting text to engage in a conversation. Type "clear" to start over, or "stop" to end the program.')
    
    directory_path = '/mnt/nfs/xcy/DataSet/Combination_pictures'
    chat_log_path = '/mnt/nfs/xcy/lxqw_short_log.jsonl'
    with torch.no_grad():
        for png_file_path in read_png_files_recursively(directory_path):
            history = None
            cache_image = None
            image_path = [png_file_path]
            query = ["Please describe the picture for me in short."]
            if world_size > 1:
                torch.distributed.broadcast_object_list(image_path, 0)
            if world_size > 1:
                torch.distributed.broadcast_object_list(query, 0)
            image_path = image_path[0]
            # write_answer_to_file(image_path, chat_log_path)
            query = query[0]
            
            while True:
                if query == "clear":
                    break
                if query == "stop":
                    sys.exit(0)
                try:
                    response, history, cache_image = chat(
                        image_path,
                        model,
                        text_processor_infer,
                        image_processor,
                        query,
                        history=history,
                        cross_img_processor=cross_image_processor,
                        image=cache_image,
                        max_length=args.max_length,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        invalid_slices=text_processor_infer.invalid_slices,
                        args=args
                        )
                except Exception as e:
                    print(e)
                    break

                filename = os.path.basename(image_path)
                log = "{" + f"\"file_name\": \"{filename}\", \"text\": \"lxqw style, {response}\"" + "}"
                write_answer_to_file(log, chat_log_path)

                if rank == 0 and not args.stream_chat:
                    if args.chinese:
                        print("模型："+response)
                    else:
                        print("Model: "+response)
                
                query = ["clear"]
                if world_size > 1:
                    torch.distributed.broadcast_object_list(query, 0)
                query = query[0]


if __name__ == "__main__":
    main()

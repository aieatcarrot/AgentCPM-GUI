import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import json
import cv2
import numpy as np
import io
import uuid
import os

class AgentCPM:
    def __init__(self, thought=True):
        model_path = "openbmb/AgentCPM-GUI"  # model path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.model = model.to("cuda:0")
        self.system_prompt = self.init_system_prompt(thought)
        self.cache_dir = "cache"

    def init_system_prompt(self, thought=True):
        ACTION_SCHEMA = json.load(open('eval/utils/schema/schema.json', encoding="utf-8"))
        items = list(ACTION_SCHEMA.items())
        insert_index = 3
        if thought:
            items.insert(insert_index, ("required", ["thought"])) # enable/disable thought by setting it to "required"/"optional"
        else:
            items.insert(insert_index, ("optional", ["thought"]))
        ACTION_SCHEMA = dict(items)
        SYSTEM_PROMPT = f'''# Role
        你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的问题，分析当前界面的GUI元素和布局，生成相应的操作。

        # Task
        针对用户问题，根据输入的当前屏幕截图，输出下一步的操作。

        # Rule
        - 以紧凑JSON格式输出
        - 输出操作必须遵循Schema约束

        # Schema
        {json.dumps(ACTION_SCHEMA, indent=None, ensure_ascii=False, separators=(',', ':'))}'''
        return SYSTEM_PROMPT

    def __call__(self, org_img, instruction, multi_turn=False, cache_id=None):
        histroy = []
        if multi_turn:
            histroy = self.recovery_from_cache(cache_id)
        img, scale = self.preprocess(org_img)
        result = self.inference(instruction, img, histroy)
        format_info = self.postprocess(img, scale, result)
        if multi_turn:
            if format_info["status"] == "finish":
                os.removedirs(f"{self.cache_dir}/{cache_id}")
                cache_id = None
            else:
                cache_id = self.cache_result(img, instruction, result, cache_id)
        format_info["cache_id"] = cache_id
        return format_info
    
    def recovery_from_cache(self, cache_id=None):
        if cache_id is None:return []
        histroy_info = json.load(open(f"{self.cache_dir}/{cache_id}/histroy.json", "r"))
        image_files = histroy_info["images"]
        images = []
        for file in image_files:
            image = Image.open(file).convert("RGB")
            images.append(image)
        conversations = histroy_info["conversations"]
        assert len(images) * 2 == len(conversations), "error"
        for i, conversation in enumerate(conversations):
            if conversation["role"] == "user":
                conversations[i]["content"].append(images[i//2])
        return conversations
    
    def cache_result(self, image, instruction, llm_result, cache_id):
        if cache_id is None:
            cache_id = str(uuid.uuid1())
            os.makedirs(f"{self.cache_dir}/{cache_id}")
        if os.path.exists(f"{self.cache_dir}/{cache_id}/histroy.json"):
            histroy_info = json.load(open(f"{self.cache_dir}/{cache_id}/histroy.json", "r"))
            image_num = len(histroy_info["images"]) + 1
            image_path = f"{self.cache_dir}/{cache_id}/screenshot_{image_num}.png"
            image.save(image_path)
            histroy_info["images"].append(image_path)
            histroy_info["conversations"].append({"role": "user",
                                  "content": [f"<Question>{instruction}</Question>\n当前屏幕截图："]})
            histroy_info["conversations"].append({"role": "assistant",
                                   "content": [llm_result]})
        else:
            image_path = f"{self.cache_dir}/{cache_id}/screenshot_1.png"
            image.save(image_path)
            histroy_info = {
                "images":[image_path],
                "conversations":[{"role": "user",
                                  "content": [f"<Question>{instruction}</Question>\n当前屏幕截图："]},
                                  {"role": "assistant",
                                   "content": [llm_result]}
                                ]
            }
        json.dump(histroy_info, open(f"{self.cache_dir}/{cache_id}/histroy.json", "w"))
        return cache_id
    
    def preprocess(self, image):
        if isinstance(image, str):
            image_pil = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)
        elif isinstance(image, bytes):
            image_pil = Image.open(io.BytesIO(image))
        elif isinstance(image, Image.Image):
            image_pil = image.copy()
        else:
            assert False, "unknown image data type!"
        resolution = image_pil.size
        w, h = resolution
        max_v = max(w, h)
        max_line_res = 1120
        scale = max_line_res / max_v
        rew = int(w * scale)
        reh = int(h * scale)
        img = image_pil.resize((rew, reh),resample=Image.Resampling.LANCZOS)
        return img, scale
    
    def inference(self, instruction, image, histroy):
        messages = [{
            "role": "user",
            "content": [
                f"<Question>{instruction}</Question>\n当前屏幕截图：",
                image
            ]
        }]
        histroy.extend(messages)
        outputs = self.model.chat(
            image=None,
            msgs=histroy,
            system_prompt=self.system_prompt,
            tokenizer=self.tokenizer,
            temperature=0.1,
            top_p=0.3,
            n=1,
        )
        return outputs

    def postprocess(self, org_img, scale, result):
        info = json.loads(result)
        width, height  = org_img.size
        json_info = {
            "status": "continue",
            "action":"",
            "parameters":{
                "point":[0, 0],
                "text":""
            }
        }

        if "POINT" in info:
            point = info["POINT"]
            new_point = [int(point[0]/1000*width/scale), int(point[1]/1000*height/scale)]
            json_info["action"] = "click"
            json_info["parameters"]["point"] = new_point
        
        if "TYPE" in info:
            json_info["action"] = "input"
            json_info["parameters"]["text"] = info["TYPE"]

        if "STATUS" in info:
            json_info["status"] = info["STATUS"]

        if "thought" in info:
            json_info["thought"] = info["thought"]
        return json_info
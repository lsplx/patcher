import json
import requests
import base64
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.multi_modal import OfaPreprocessor
from collections import deque
# model = 'damo/ofa_visual-question-answering_pretrain_huge_en'
# preprocessor = OfaPreprocessor(model_dir=model)
# ofa_pipe = pipeline(
#     Tasks.visual_question_answering,
#     model=model,
#     preprocessor=preprocessor)
# 初始化一个空字典来存放提取的数据
captions_and_ids = {}
# 加载CLIP模型和处理器
model_id = "/newdata/czy/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)


def calculate_average_attention(token_importance, object_tokens):
    total_score = 0
    count = 0
    for token in object_tokens:
        for item in token_importance:
            if item['token'] == token:
                total_score += item['attention_score']
                count += 1
                break
    return total_score / count if count > 0 else 0

def call_t2iapi(text):
    datas = {"input":text, "version":"1", "attention": [["1","2"],["3","1"]], "image_id": index, "index": index, "highlight_keywords": "none","attention_multiplier":"none"}
    data = json.dumps(datas)
    response = requests.post(url, data=data, headers=headers)
    result_api = response.json()
    return result_api

def save_img(base64_str, index):
    img_bytes = base64.b64decode(base64_str)
    # 使用BytesIO处理解码后的字节数据
    img_io = io.BytesIO(img_bytes)
    # 使用Pillow加载图像
    image = Image.open(img_io)
    # 保存图像到本地文件系统
    save_path = "/newdata/czy/Text_to_image_repair/stable_diffusion_14/ATE_feature/" + str(index) + ".png"
    image.save(save_path)
    return save_path

def compute_clip_score(image, text):
    # 处理图像和文本输入
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    
    # 使用CLIP模型进行前向传递
    outputs = model(**inputs)
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
     # 计算余弦相似度
    similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
    
    return similarity.item()

def VQAvalidate( image_path, object1, object2):
    result_dic = {}
    image = Image.open(image_path)
    object1_clip_score = compute_clip_score(image, object1)
    object2_clip_score = compute_clip_score(image, object2)
    result_dic[object1] = object1_clip_score
    result_dic[object2] = object2_clip_score
    if object1_clip_score > 0.17:
        judgeobject1 = True
    else:
        judgeobject1 = False
    
    if object2_clip_score > 0.17:
        judgeobject2 = True
    else:
        judgeobject2 = False
    if (judgeobject1 and judgeobject2):
        # return "correct"
        result_dic["is_correct"] = "correct"
        result_dic["wrongobject"] = []
        result_dic["correctobject"] = []
        # data["is_correct"] = "correct"
        # correct_num += 1
        # continue
    else:
        result_dic["is_correct"] = "wrong"
        if judgeobject1 == False and judgeobject2 == True:
            result_dic["wrongobject"] = [object1]
            result_dic["correctobject"] = [object2]
        elif judgeobject2 == False and judgeobject1 == True:
            result_dic["wrongobject"] = [object2]
            result_dic["correctobject"] = [object1]
        elif judgeobject2 == False and judgeobject1 == False: 
            result_dic["wrongobject"] = [object2,object1]
            result_dic["correctobject"] = []
    return result_dic

# read file
with open('/data/czy/stablediffusion-main/wordnet_process/repair_method/stable14/ATE_objects_judge.json', 'r', encoding='utf-8') as file:
    datajson = json.load(file)

#/data/czy/stablediffusion-main/wordnet_process/repair_method/MSCOCO_wordstree.json
with open('/data/czy/stablediffusion-main/wordnet_process/repair_method/ATE_wordstree.json', 'r', encoding='utf-8') as file:
    treejson = json.load(file)

#/data/czy/stablediffusion-main/wordnet_process/repair_method/MSCOCO_shape_GPT.json
with open('/data/czy/stablediffusion-main/wordnet_process/repair_method/ATE_shape_GPT.json', 'r', encoding='utf-8') as file:
    featurejson = json.load(file)

final_result = []
#call t2i model
url = "http://172.16.20.73:3000/api/v1/predict"
headers = {'Content-Type': 'application/json'}
correct_num = 0
wrong_num = 0
index = 0
for index, each in enumerate(datajson[:500]):
    # index = index + 1
    
    #extract object in sentence...
    #call api
    if 'is_correct' in each and each['is_correct'] == 'wrong':
        print(index)
        t2iresult = call_t2iapi(each["sentence"])
        image = save_img(t2iresult["img_base64"], index)
        object1 = each["pair"][0]
        object2 = each["pair"][1]
        VQA_result = VQAvalidate( image, object1, object2)
        if len(VQA_result['wrongobject']) == 1:
            wrong_tokens = VQA_result['wrongobject'][0].split()
            correct_tokens = VQA_result['correctobject'][0].split()
            wrong_score = calculate_average_attention(t2iresult['token_attention_linking'], wrong_tokens)
            correct_score = calculate_average_attention(t2iresult['token_attention_linking'], correct_tokens)
            original_attention_diff = wrong_score - correct_score
            each["original_attention_diff"] = original_attention_diff
            # 获取motorcycle字典
            shape_feature = featurejson.get(VQA_result['wrongobject'][0], {})
            # 处理shape键
            shapes = shape_feature.get("shape", "")
            shape_list = shapes.split("; ")
            # print("Shapes:")
            max_shape_attention_diff = float('-inf')
            for shape in shape_list[:4]:
                #replace word
                newtext = each["sentence"].replace(VQA_result['wrongobject'][0], shape)
                # apply first condition
                t2iresult_shape = call_t2iapi(newtext)
                image = save_img(t2iresult_shape["img_base64"], index)
                # object1 = each["pair"][0]
                # object2 = each["pair"][1]
                VQA_result_shape = VQAvalidate( image, VQA_result['wrongobject'][0], VQA_result['correctobject'][0])
                if VQA_result_shape["is_correct"] == "correct":
                    each['is_correct'] = "correct"  # 标记当前节点
                    each["shape"] = shape
                    each["shapetext"] = newtext
                    break
                if len(VQA_result_shape['wrongobject']) == 1:
                    # wrong_tokens = VQA_result['wrongobject'][0].split()
                    # correct_tokens = VQA_result_shape['correctobject'][0].split()
                    wrong_score = calculate_average_attention(t2iresult_shape['token_attention_linking'], wrong_tokens)
                    correct_score = calculate_average_attention(t2iresult_shape['token_attention_linking'], correct_tokens)
                    attention_diff = wrong_score - correct_score
                    # each['attention_diff'] = attention_diff
                    if attention_diff > original_attention_diff and attention_diff > max_shape_attention_diff:
                        max_shape_attention_diff = attention_diff
                         # 更新记录最大attention_diff的键
                        each['attention_shape_diff'] = max_shape_attention_diff
                        each["shape"] = shape

        if len(VQA_result['wrongobject']) == 2:
            if VQA_result[object1] < VQA_result[object2]:
                wrong_tokens = object1.split()
                correct_tokens = object2.split()
                wrongobject = object1
                correctobject = object2
            else:
                wrong_tokens = object2.split()
                correct_tokens = object1.split()
                wrongobject = object2
                correctobject = object1
            wrong_tokens = wrongobject.split()
            correct_tokens = correctobject.split()
            wrong_score = calculate_average_attention(t2iresult['token_attention_linking'], wrong_tokens)
            correct_score = calculate_average_attention(t2iresult['token_attention_linking'], correct_tokens)
            original_attention_diff = wrong_score - correct_score
            each["original_attention_diff"] = original_attention_diff
            # 获取motorcycle字典
            shape_feature = featurejson.get(wrongobject, {})
            # 处理shape键
            shapes = shape_feature.get("shape", "")
            shape_list = shapes.split("; ")
            # print("Shapes:")
            max_shape_attention_diff = float('-inf')
            for shape in shape_list[:4]:
                #replace word
                newtext = each["sentence"].replace(wrongobject, shape)
                # apply first condition
                t2iresult_shape = call_t2iapi(newtext)
                image = save_img(t2iresult_shape["img_base64"], index)
                # object1 = each["pair"][0]
                # object2 = each["pair"][1]
                VQA_result_shape = VQAvalidate( image, wrongobject, correctobject)
                if VQA_result_shape["is_correct"] == "correct":
                    each['is_correct'] = "correct"  # 标记当前节点
                    each["shape"] = shape
                    each["shapetext"] = newtext
                    break
                if len(VQA_result_shape['wrongobject']) == 1:
                    # wrong_tokens = VQA_result['wrongobject'][0].split()
                    # correct_tokens = VQA_result_shape['correctobject'][0].split()
                    wrong_score = calculate_average_attention(t2iresult_shape['token_attention_linking'], wrong_tokens)
                    correct_score = calculate_average_attention(t2iresult_shape['token_attention_linking'], correct_tokens)
                    attention_diff = wrong_score - correct_score
                    # each['attention_diff'] = attention_diff
                    if attention_diff > original_attention_diff and attention_diff > max_shape_attention_diff:
                        max_shape_attention_diff = attention_diff
                         # 更新记录最大attention_diff的键
                        each['attention_shape_diff'] = max_shape_attention_diff
                        each["shape"] = shape
            # if each['is_correct'] == "correct":
            #     continue

            # color_feature = featurejson.get(VQA_result['wrongobject'][0], {})
            # colors = color_feature.get("color", "")
            # color_list = colors.split(";")
            # # print("Shapes:")
            # max_color_attention_diff = float('-inf')
            # for color in color_list[:3]:
            #     #replace word
            #     newtext = each["sentence"].replace(VQA_result['wrongobject'][0], color + " " + VQA_result['wrongobject'][0])
            #     # apply first condition
            #     t2iresult_color = call_t2iapi(newtext)
            #     image = save_img(t2iresult["img_base64"], index)
            #     # object1 = each["pair"][0]
            #     # object2 = each["pair"][1]
            #     VQA_result_color = VQAvalidate( image, VQA_result['wrongobject'][0], VQA_result['correctobject'][0])
            #     if VQA_result_color["is_correct"] == "correct":
            #         each['is_correct'] = "correct"  # 标记当前节点
            #         break
            #     if len(VQA_result['wrongobject']) == 1:
            #         # wrong_tokens = VQA_result['wrongobject'][0].split()
            #         # correct_tokens = VQA_result['correctobject'][0].split()
            #         wrong_score = calculate_average_attention(t2iresult_color['token_attention_linking'], wrong_tokens)
            #         correct_score = calculate_average_attention(t2iresult_color['token_attention_linking'], correct_tokens)
            #         attention_diff = wrong_score - correct_score
            #         # each['attention_diff'] = attention_diff
            #         if attention_diff > original_attention_diff and attention_diff > max_color_attention_diff:
            #             max_color_attention_diff = attention_diff
            #              # 更新记录最大attention_diff的键
            #             each['attention_color_diff'] = max_color_attention_diff
            #             each["color"] = color
            
   

# 指定要保存的JSON文件名
output_filename = '/data/czy/stablediffusion-main/wordnet_process/repair_method/stable14/ATE_feature_V14.json'

# 打开文件用于写入
with open(output_filename, 'w', encoding='utf-8') as output_file:
    # 将数据写入JSON文件，使用indent参数让输出的文件格式化更易于阅读
    json.dump(datajson, output_file, ensure_ascii=False, indent=4)

print(f"Data has been saved to {output_filename}")
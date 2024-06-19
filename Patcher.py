import json
import requests
import base64
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import io
from collections import deque
import spacy
import json
import nltk
from nltk.corpus import wordnet as wn
from openai import OpenAI
import numpy as np
import json
import json
from nltk.corpus import wordnet as wn
from openai import OpenAI
import numpy as np
client = OpenAI(api_key="your api key")
nlp = spacy.load("en_core_web_trf")
def get_embedding(text):
    response = client.embeddings.create(
    input=text,
    model="text-embedding-3-large"
)
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_hyponyms(synset, depth, original_word):
    hyponyms = {}
    original_word_embedding = get_embedding(original_word)  # 
    for hyponym in synset.hyponyms():
        hyponym_name = hyponym.lemmas()[0].name().replace('_', ' ')  # 
        hyponym_embedding = get_embedding(hyponym_name)
        similarity = cosine_similarity(original_word_embedding, hyponym_embedding)
        if similarity > 0.5:
            child_hyponyms, _ = get_hyponyms(hyponym, depth + 1, original_word)
            hyponym_name = hyponym.lemma_names('eng')[0]
            hyponyms[hyponym_name] = {"data": child_hyponyms, "depth": depth + 1, "sim":similarity}
    return hyponyms, depth

def explore_word_hyponym(word):
    results = {}
    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:  
        return {"Hyponyms": "No data available"}
    hyponyms, _ = get_hyponyms(synsets[0], 0, word)
    return {"Hyponyms": hyponyms}


def extract_objects(sentence):
    doc = nlp(sentence)
    def is_subject_or_object(token):
        return token.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj")

    def remove_determiners(chunk):
        tokens = [token for token in chunk if token.pos_ != "DET"]
        return " ".join(token.text for token in tokens)

    def contains_verb(chunk):
        return any(token.pos_ == "VERB" for token in chunk)

    subject_objects = []

    for chunk in doc.noun_chunks:
        if contains_verb(chunk):
            continue
        
        if is_subject_or_object(chunk.root):
            cleaned_chunk = remove_determiners(chunk)
            subject_objects.append(cleaned_chunk)

    return subject_objects

def calculate_att_diff(correct_set, wrong_set):
    N = len(wrong_set)
    C = len(correct_set) 
    total_diff = 0
    count = 0  
    if C == 0:
        for i in range(N):
            for j in range(i + 1, N):
                total_diff += abs(wrong_set[i] - wrong_set[j])
                count += 1
    else:
        for Oi in wrong_set:
            for Oj in correct_set:
                total_diff += abs(Oi - Oj)
                count += 1
    if count == 0:
        return 0  
    att_diff = total_diff / count
    return att_diff

captions_and_ids = {}

# upload clip
model_id = "XXX/clip-vit-large-patch14"
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
    img_io = io.BytesIO(img_bytes)
    image = Image.open(img_io)
    save_path = "XXX/" + str(index) + ".png"
    image.save(save_path)
    return save_path

def compute_clip_score(image, text):
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    
    outputs = model(**inputs)
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
    similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
    
    return similarity.item()

def Neglected_Identification( image_path, object_list):
    result_dic = {}
    image = Image.open(image_path)
    neglected_list = []
    correct_list = []
    for each in object_list:
        each_clip_score = compute_clip_score(image, each)
        if each_clip_score > 0.17:
            correct_list.append(each)
        else:
            neglected_list.append(each)
    if len(neglected_list) == 0:
        result_dic["is_correct"] = "correct"
        result_dic["wrongobject"] = correct_list
        result_dic["correctobject"] = neglected_list
    else:
        result_dic["is_correct"] = "wrong"
        result_dic["wrongobject"] = correct_list
        result_dic["correctobject"] = neglected_list
    return result_dic

def find_hyponyms(tree, current_node, direction):
    queue = deque([(key, val) for key, val in tree.items()])
    while queue:
        key, value = queue.popleft()
        #first time
        if current_node == "":
            return key
        if current_node == key:
            if direction == "child":
                if "data" in value and value["data"]:
                    subkey, subval = value["data"].items()[0]
                    return subkey
                else:
                    if queue:
                        siblekey, siblevalue = queue.popleft()
                        return siblekey
                    else:
                        return ""
            else:
                if queue:
                        siblekey, siblevalue = queue.popleft()
                        return siblekey
                else:
                    return ""
        else:
            if "data" in value and value["data"]:
                for subkey, subval in value["data"].items():
                    queue.append((subkey, subval))
def GPT_modifier(key):
    prompt = "What are the most common physical characteristics of the " + key + "? " + ''' 
            Please output only the the physical characteristics without explanation and add physical characteristics on the object to contruct a fluent phrase and separating each phrase with a semicolon.
        Example:
        Question: What are the most common physical characteristics of bicycle?
        Output: two-wheeled bicycle; bicycle with Pedals; bicycle with Chain and Gears
    ''' 
    completion = client.chat.completions.create(
    model="gpt-3.5",
    messages=[
        {"role": "user", "content": prompt}
    ]
    )
    result = completion.choices[0].message.content.lower()
    return result

def GPT_color(key):
    prompt = "What are the most common color of the " + key + "? " + ''' 
        Please output only the the colors without explanation and add colors on the object to contruct a fluent phrase and separating each phrase with a semicolon.
    Example:
    Question: What are the most common color of apple?
    Output: red apple; green apple'''
    completion = client.chat.completions.create(
    model="gpt-3.5",
    messages=[
        {"role": "user", "content": prompt}
    ]
    )
    result = completion.choices[0].message.content.lower()
    return result

        
# read file
with open('your_multipleobject_file.json', 'r', encoding='utf-8') as file:
    datajson = json.load(file)


final_result = []
#call t2i model
url = "http://XXX:8000/api/v1/predict"
headers = {'Content-Type': 'application/json'}
for index, each in enumerate(datajson):
    #extract object
    object_list = extract_objects(each["sentence"])
    #call t2i_and_attention_tool
    t2iresult = call_t2iapi(each["sentence"])
    image = save_img(t2iresult["img_base64"], index)
    Identification_result = Neglected_Identification(image, object_list)
    if Identification_result["is_correct"] == "correct":
        each["is_correct"] = "correct"
        each["repaired_prompt"] = each["sentence"]
        continue
    for neglect_object in Identification_result["wrongobject"]:
        #cal attention for each object
        atten_neglects = []
        for each_object in Identification_result["wrongobject"]:
            object_tokens = each_object.split()
            attention_score = calculate_average_attention(t2iresult['token_attention_linking'], object_tokens)
            atten_neglects.append(attention_score)
        atten_corrects = []
        if len(Identification_result["correctobject"]) != 0:
            for each_object in Identification_result["correctobject"]:
                object_tokens = each_object.split()
                attention_score = calculate_average_attention(t2iresult['token_attention_linking'], object_tokens)
                atten_corrects.append(attention_score)
        original_attention_diff = calculate_att_diff(atten_corrects, atten_neglects)
        #implicit feature enhancing
        each["hyponym_repair"] = True
        hyponym_tree = explore_word_hyponym(neglect_object)  
        neglect_hyponym_tree = hyponym_tree["Hyponyms"]
        direction = "sible"
        if neglect_hyponym_tree == "No data available" or neglect_hyponym_tree == {}:
            each["hyponym_repair"] = False
        if each["hyponym_repair"] == True:
            hyponym_word = find_hyponyms(neglect_hyponym_tree, neglect_object, direction)
            hyponym_sentence = each["sentence"].replace(neglect_object, hyponym_word)
        #explicit feature enhancing
        shape_list = GPT_modifier(neglect_object).split("; ")
        currect_shape_index = 0
        feature_sentence = each["sentence"].replace(neglect_object, shape_list[currect_shape_index])
        currect_shape_index += 1
        each["feature_repair"] = True
        while True:
            #return to judge
            if each["hyponym_repair"] == False and each["feature_repair"]== False:
                each["repaired_prompt"] = each["sentence"]
                each["is_correct"] = "wrong"
                break
            if each["hyponym_repair"] == True:
                t2iresult_hyponym = call_t2iapi(hyponym_sentence)
                image_hyponym = save_img(t2iresult["img_base64"], index)
                Identification_result_hyponym = Neglected_Identification(image, object_list)
                if Identification_result_hyponym["is_correct"] == "correct":
                    each["is_correct"] = "correct"
                    each["repaired_prompt"] = hyponym_sentence
                    break
                else:
                    atten_neglects = []
                    for each_object in Identification_result_hyponym["wrongobject"]:
                        if each_object != neglect_object:
                            object_tokens = each_object.split()
                            attention_score = calculate_average_attention(t2iresult_hyponym['token_attention_linking'], object_tokens)
                            atten_neglects.append(attention_score)
                        else:
                            object_tokens = hyponym_word.split()
                            attention_score = calculate_average_attention(t2iresult_hyponym['token_attention_linking'], object_tokens)
                            atten_neglects.append(attention_score)
                    atten_corrects = []
                    if len(Identification_result["correctobject"]) != 0:
                        for each_object in Identification_result["correctobject"]:
                            object_tokens = each_object.split()
                            attention_score = calculate_average_attention(t2iresult['token_attention_linking'], object_tokens)
                            atten_corrects.append(attention_score)
                    hyponym_attention_diff = calculate_att_diff(atten_corrects, atten_neglects)
            if each["feature_repair"] == True:
                t2iresult_feature = call_t2iapi(feature_sentence)
                image_feature = save_img(t2iresult["img_base64"], index)
                Identification_result_feature = Neglected_Identification(image, object_list)
                if Identification_result_feature["is_correct"] == "correct":
                    each["is_correct"] = "correct"
                    each["repaired_prompt"] = feature_sentence
                    break
            if hyponym_attention_diff < original_attention_diff:
                direction = "child"
            else:
                direction = "sible"
            if each["hyponym_repair"] == True:
                hyponym_word = find_hyponyms(neglect_hyponym_tree, neglect_object, direction)
                if hyponym_word != "":
                    hyponym_sentence = each["sentence"].replace(neglect_object, hyponym_word)
                else:
                    each["hyponym_repair"] == False          
            if currect_shape_index < len(shape_list):
                feature_sentence = each["sentence"].replace(neglect_object, shape_list[currect_shape_index])
                currect_shape_index += 1
            else:
                each["feature_repair"] == False  

output_filename = 'XXX_repair.json'

with open(output_filename, 'w', encoding='utf-8') as output_file:
    json.dump(datajson, output_file, ensure_ascii=False, indent=4)
   
print(f"Data has been saved to {output_filename}")
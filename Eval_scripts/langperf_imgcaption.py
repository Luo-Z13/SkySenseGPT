import json
import os
import random
from tqdm import tqdm
from openai import OpenAI
import datetime
import time
import argparse
import re
import openai
from functools import partial
import httpx
import math
gpt35_key = "your key"
gpt4_key = "your key"
api_base = "https://oneapi.xty.app/v1"


client = OpenAI(
    base_url=api_base,
    api_key=gpt35_key,
    http_client=httpx.Client(
        base_url=api_base,
        follow_redirects=True,
    ),
)


SYS_VQA="""
Please act as an impartial judge and conduct a comprehensive assessment of a multimodal AI assistant's performance in the field of Visual Question Answering (VQA). Each data sample to be evaluated follows the following format:
[Ground Truth]
{Ground truth}\n
[answer]
{answer}\n
Your task is to evaluate the quality of natural language generation from AI assistant considering factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. 
Please according to the correct answer [ground_truth] and the answer [answer] analyzed by AI, give the accuracy of [answer] compared to [ground_truth] through an overall score ranging from 0 to 10 based on explanation, where a higher score indicates better overall performance. Please output in the following format:\n
[Overall Score] {An integer ranging from 0 to 10 representing the final evaluation score}\n
Please ensure that your evaluation score comprehensively captures the AI assistant's performance avoiding any potential bias. Assuming that the visual information mentioned by the AI assistant is contained in the image, you only need to give the score. Your assessments will contribute to enhancing the assistant's effectiveness in visual question answering.
"""


DATA_Template = """
[Ground Truth]
{}\n
[answer]
{}\n
"""
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_model_name(answer_path):
    js_name = os.path.basename(answer_path)
    match = re.match(r'([^_]+)', js_name)
    return match.group(1)

def generate_query(gt, answer):
    query = DATA_Template.format(gt,answer)
    return query


def parse_score(text):
    #match = re.search(r'\[Overall Score\]: (\d+)', text)
    match1 = [int(x) for x in re.findall(r'\[Overall Score\]\s*\{(\d+)\}', text)]
    match2=[int(x) for x in re.findall(r'\[Overall Score\]\s*(\d+)', text)]
    match3=[int(x) for x in re.findall(r'Overall Score:\s*(\d+)', text)]
    score=match1+match2+match3
    
       
    return score

def GPT_Metric(base_data_path,  response_path):
    questions=[]
    questions = [json.loads(q) for q in open(os.path.expanduser(base_data_path), "r")]
    answers_file = os.path.expanduser(response_path)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    sample_ids = random.sample(range(len(questions)), 200)
    sample_ids.sort()
    v6_list=[0,0,0,0,0,0]
    v6_bool_list = [len(sample_ids) for _ in range(6)]
    for i in tqdm(sample_ids):
        question_id=questions[i]['question_id']
        gt=questions[i]['ground_truth']
        answer=questions[i]['answer']
        query=generate_query(gt,answer)
        messages = [{"role":"system", "content": SYS_VQA}]
        messages.append({"role":"user", "content":query})
        response_list=[]
        while True:
            try:
                response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages,n=6, max_tokens=512)
                response_list.append(response)
                break
            except:
                continue
        cur_text = []
        cur_score = []
        print(response)
        for ans in response.choices:
            cur_text.append(ans.message.content)
            score = parse_score(ans.message.content)
            print(score)
            if len(score)!=0:
                cur_score.append(score[0])
            else :
                cur_score.append(0)
        input_tokens=response.usage.prompt_tokens
        comp_tokens=response.usage.completion_tokens
        cost = input_tokens*0.03/1000+comp_tokens*0.06/1000
        cur_dict={
            'ids':question_id,
            'response':cur_text,
            'scores': cur_score,
            'cost': cost,
        }
        res_list=[]
        res_list.append(cur_dict)
        ans_file.write(json.dumps(cur_dict)+"\n")
        ans_file.flush()
        time.sleep(2)
    
    total_score, total_cost = 0,0
    for res in res_list:
        tmp=0
        non_zero_count = len(list(filter(bool, res['scores'])))
        for idx,i in enumerate(res['scores']):
            tmp+=i
            v6_list[idx]=v6_list[idx]+i
            if i==0:
                v6_bool_list[idx]=v6_bool_list[idx]-1
        tmp/=non_zero_count 
        total_score+=tmp
        total_cost+=res['cost']
    total_score/=len(res_list)
    for i in range(6):
        v6_list[i]=v6_list[i]/v6_bool_list[i]
    print(v6_bool_list)
    overall_res={
        'Avg Score': total_score,
        'Cost':total_cost,
        'v6_score':v6_list
    }
    print(overall_res)
    ans_file.write(json.dumps(overall_res)+"\n")
    
    
def eval_score(response_list):
    total = 0
    cnt = 0
    for item in response_list:
        text = item['choices'][0]['message']['content']
        score = parse_score(text)
        if score!=None and score>=0:
            total+=score
            cnt+=1
    return total/cnt*10.0


def eval_cost(response_list):
    inputs=0
    outputs = 0
    cost=0.0
    for item in response_list:
        ic= item["usage"]["prompt_tokens"]
        oc = item["usage"]["completion_tokens"]
        inputs+=ic
        outputs+=oc
    print(inputs)
    print(outputs)
    cost = inputs/1000*0.0015+outputs/1000*0.002
    return cost

if __name__ == '__main__':
    file_path=[
        'image_caption_output.jsonl'
    ]
    base_path = r'/pythonProject/'
    count=7
    for file_idx in file_path:
        count=count+1
        answer_path = os.path.join(base_path, f'score{count}.jsonl')
        parser = argparse.ArgumentParser()
        parser.add_argument('--base-data-path', default=file_idx)
        parser.add_argument('--response-path', default=answer_path
                        )
        parser.add_argument("--num-chunks", type=int, default=1)
        parser.add_argument("--chunk-idx", type=int, default=0)
        parser.add_argument("--temperature", type=float, default=0.2)
        parser.add_argument("--top_p", type=float, default=None)
        parser.add_argument("--num_beams", type=int, default=1)
        args = parser.parse_args()
        GPT_Metric(args.base_data_path,args.response_path)

    
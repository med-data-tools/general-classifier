import csv
import re
import torch
import json
import torch
import ast
import uuid
import json
import os
import time
import gc
from openai import OpenAI
torch.manual_seed(0)
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from guidance import models, select, gen
global model
global modelType
global inferenceType
global tokenizer
global LLM


global promptModel
global promptModelType
global promptInferenceType
global promptModelTokenizer
global promptLLM
global previous_results
global topics
global selectOptions
global topic_id_counter
global interface


model=""
modelType="Transformers"
inferenceType=""
promptModel=""
promptModelType="Transformers"
promptInferenceType=""




topics = []


interface = False
previous_results = {}
topic_id_counter = 0



class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def setModel(newModel,newModelType="Transformers",api_key="",newInferenceType="transformers"):
    global model
    global modelType
    global inferenceType
    model=newModel
    modelType=newModelType
    inferenceType=newInferenceType
    global ModelGuidance 
    global client
    global LLM
    global tokenizer
    if modelType=="Transformers":
        if inferenceType=="transformers":
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            LLM = AutoModelForCausalLM.from_pretrained(
                model,
                device_map='cuda',
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation = "flash_attention_2"
            )
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif inferenceType=="guidance":
            ModelGuidance = models.Transformers(model, echo=False, trust_remote_code=True)
        else:
            print("Invalid inference Type.")
        
    if modelType=="OpenAI":
        inferenceType="cloud"
        if not api_key=="":
            client = OpenAI(api_key=api_key)
    if modelType=="DeepInfra":
        inferenceType="cloud"
        if not api_key=="":
            client = OpenAI(api_key=api_key,base_url="https://api.deepinfra.com/v1/openai")

            
def load_model():
    """Load model and return it"""
    LLM = AutoModelForCausalLM.from_pretrained(
        model,
        device_map='cuda',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )        
    return LLM
    
def unload_model(model_to_unload):
    """Thoroughly unload model from GPU memory"""
    if model_to_unload is not None:
        # Move model to CPU first
        model_to_unload.cpu()
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Delete the model object
        del model_to_unload
        # Force garbage collection multiple times
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    return None


def setPromptModel(newPromptModel, newPromptModelType, api_key="", newInferenceType="guidance"):
    global Model
    global promptModel
    global promptModelType
    global promptInferenceType
    promptModel=newPromptModel
    promptModelType=newPromptModelType
    promptInferenceType=newInferenceType
    global promptModelGuidance 
    global promptLLM
    global client
    if promptModelType=="Transformers":
        if inferenceType=="transformers":
            promptModelTokenizer = AutoTokenizer.from_pretrained(promptModel, trust_remote_code=True)
            promptLLM = AutoModelForCausalLM.from_pretrained(
                promptModel,
                device_map='cuda',
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation = "flash_attention_2"
            )
            promptModelTokenizer.pad_token_id = promptModelTokenizer.eos_token_id
        elif inferenceType=="guidance":
            if promptModel==Model:
                promptModelGuidance=ModelGuidance
            else:
                promptModelGuidance = models.Transformers(model, device_map='cuda', torch_dtype=torch.bfloat16, echo=False, trust_remote_code=True)
    if modelType=="OpenAI" or promptModelType=="OpenAI":
        if not api_key=="":
            client = OpenAI(api_key=api_key)
    if modelType=="DeepInfra" or promptModelType=="DeepInfra":
        if not api_key=="":
            client = OpenAI(api_key=api_key,base_url="https://api.deepinfra.com/v1/openai")


def calculate_word_probability(prompt, target_word):
    """
    Only available for inferenceType='transformers'
    Calculate the probability of a target word following a prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt string
        target_word: The word to calculate probability for

    Returns:
        tuple: (probability, token_probabilities)
    """

    LLM.eval()

    if not prompt.endswith(' '):
        target_word = ' ' + target_word

    target_tokens = tokenizer.encode(target_word, add_special_tokens=False)
    token_probabilities = []
    current_text = prompt

    for i, token_id in enumerate(target_tokens):
        inputs = tokenizer(current_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = LLM(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        token_prob = next_token_probs[token_id].item()
        token_probabilities.append(token_prob)
        current_text = tokenizer.decode(
            tokenizer.encode(current_text, add_special_tokens=False) + [token_id],
            skip_special_tokens=True
        )
    total_probability = torch.tensor(token_probabilities).prod().item()

    return total_probability, token_probabilities
    
    
def calculate_options_probabilities(prompt, options):
    """   
    Args:
        prompt: Input prompt string
        options: List of possible options to evaluate
        
    Returns:
        tuple: (best_option, best_probability, all_probabilities_dict, relative_probabilities_dict)
    """
    LLM.eval()
    if not prompt.endswith(' '):
        space_prefix = ' '
    else:
        space_prefix = ''
    first_token_groups = {}
    for option in options:
        first_token = tokenizer.encode(space_prefix + option, add_special_tokens=False)[0]
        if first_token not in first_token_groups:
            first_token_groups[first_token] = []
        first_token_groups[first_token].append(option)
    base_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    option_probabilities = {}
    with torch.no_grad():
        base_outputs = LLM(**base_inputs)
        base_logits = base_outputs.logits[0, -1, :]
        base_probs = torch.nn.functional.softmax(base_logits, dim=-1)
        
    for first_token_id, group_options in first_token_groups.items():
        if len(group_options) == 1:
            option = group_options[0]
            option_with_space = space_prefix + option
            token_ids = tokenizer.encode(option_with_space, add_special_tokens=False)
            
            if len(token_ids) == 1:
                option_probabilities[option] = base_probs[first_token_id].item()
            else:
                probability, _ = calculate_word_probability(prompt, option)
                option_probabilities[option] = probability
        else:
            for option in group_options:
                probability, _ = calculate_word_probability(prompt, option)
                option_probabilities[option] = probability
    
    total_probability = sum(option_probabilities.values())
    relative_probabilities = {
        option: (prob / total_probability if total_probability > 0 else 0)
        for option, prob in option_probabilities.items()
    }
    
    best_option = max(option_probabilities.items(), key=lambda x: x[1])
    
    return best_option[0], best_option[1], option_probabilities, relative_probabilities, relative_probabilities[best_option[0]]
     
    
        
def getAnswer(prompt, categories, constrainedOutput, temperature=0.0, thinkStep=0):
    if inferenceType=="cloud":
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=30,
            temperature=temperature,
        )
        generated_answer = completion.choices[0].message.content
        for option in categories:
            escaped_option = re.escape(option)

            if re.search(escaped_option, generated_answer, re.IGNORECASE):
                ret = option  
                break
        else:
            ret = "undefined"
        return ret, "-"
        
    elif inferenceType=="guidance":
        if constrainedOutput==True:
            output=ModelGuidance+f' '+prompt+select(options=categories,name='answer')
            ret=output["answer"]   
        else:
            output=ModelGuidance+f' '+prompt+gen(max_tokens=15,name='answer')
            generated_answer = output["answer"]
            for option in categories:
                escaped_option = re.escape(option)
                if re.search(escaped_option, generated_answer, re.IGNORECASE):
                    ret = option  
                    break
            else:
                ret = "undefined"


        return ret, "-"
    elif inferenceType=="transformers":
        ### Optional additional Think-Step
        if thinkStep>0:
            current_text = prompt
            inputs = tokenizer(current_text, return_tensors="pt").to("cuda")
            period_id = tokenizer.encode(".", add_special_tokens=False)[-1]
            question_id = tokenizer.encode("?", add_special_tokens=False)[-1]
            exclamation_id = tokenizer.encode("!", add_special_tokens=False)[-1]
            stopping_criteria = StoppingCriteriaList([
            StopOnTokens([period_id, question_id, exclamation_id])
            ])
        
            outputs = LLM.generate(inputs.input_ids, temperature=0.01, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95, stopping_criteria=stopping_criteria)
            newPrompt=tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]+" Therefore the correct answer is '"
            print("New Prompt after Think-Step: ", newPrompt)
        else:
            newPrompt=prompt
        if constrainedOutput==True:
            best_option, best_prob, all_probs_abs, all_probs_rel, best_rel_prob = calculate_options_probabilities(newPrompt, categories)
        else:
            inputs = tokenizer(newPrompt, return_tensors="pt").to("cuda")
            outputs = LLM.generate(inputs.input_ids, max_length=100, num_return_sequences=1, temperature=temperature, do_sample=True, top_k=50, top_p=0.95)
            for option in categories:
                escaped_option = re.escape(option)
                if re.search(escaped_option, generated_answer, re.IGNORECASE):
                    ret = option  
                    break
                else:
                    ret = "undefined"
            best_option=ret
            best_rel_prob="-"
            
        return best_option, best_rel_prob
    

def evaluate_condition(condition):
    if not condition: 
        return True

    if "==" not in condition:
        print(f"Invalid condition format: {condition}")
        return False

    left_side, right_side = condition.split("==", 1)
    left_side = left_side.strip()
    right_side = right_side.strip()

    if left_side not in previous_results:
        #print(f"No previous classification result for topic '{left_side}'. Condition: {condition}")
        return False

    chosen_cat_id = previous_results[left_side]

    return chosen_cat_id == right_side


def classify(text, isItASingleClassification=True, constrainedOutput=True, withEvaluation=False, groundTruthRow=None):
    selectOptions = []
    for topic_data in topics:
        tmpSelectOptions = "["
        for category_input, _, _ in topic_data['categories']:
            tmpSelectOptions += "'" + category_input.value + "',"
        tmpSelectOptions = tmpSelectOptions[:-1] + "]"
        selectOptions.append(tmpSelectOptions)

    ret = []
    probs = []

    if withEvaluation and groundTruthRow is not None:
        for i, topic_info in enumerate(topics):
            groundTruthCategoryName = groundTruthRow[i+1] 
            gt_cat_id = None
            for (cat_input, _, cat_id) in topic_info['categories']:
                if cat_input.value == groundTruthCategoryName:
                    gt_cat_id = cat_id
                    break
            previous_results[topic_info['id']] = gt_cat_id

    for l in range(len(selectOptions)):
        condition = topics[l]['condition'].value.strip()
        condition_is_true = evaluate_condition(condition)

        if not condition_is_true:
            ret.append("")
            if interface == True and isItASingleClassification:
                print(f"Skipping {topics[l]['topic_input'].value} due to unmet condition: {condition}")
            continue

        prompt = topics[l]['prompt'].value
        prompt = prompt.replace('[TOPIC]', topics[l]['topic_input'].value)
        prompt = prompt.replace('[CATEGORIES]', selectOptions[l])
        prompt = prompt.replace('[TEXT]', text)

        categories=ast.literal_eval(selectOptions[l])
        answer, bestRelProb = getAnswer(prompt, categories, constrainedOutput, 0.0)
        
        #Output for Single Classification
        if isItASingleClassification==True:
            print(topics[l]['topic_input'].value,":",answer," (Relative Probability:",bestRelProb,")")
        ret.append(answer)
        probs.append(bestRelProb)

        if not withEvaluation:
            chosen_category_id = None
            for category_input, _, category_id in topics[l]['categories']:
                if category_input.value == answer:
                    chosen_category_id = category_id
                    break
            previous_results[topics[l]['id']] = chosen_category_id

        if interface == True and isItASingleClassification:
            print(f"{topics[l]['topic_input'].value}: {answer}")

    return ret,probs

        
def get_current_accuracy(topic_info):
    label_text = topic_info['performance_label'].value
    match = re.match(r"Accuracy:\s+([\d.]+)%", label_text)
    if match:
        return float(match.group(1))
    return 0.0



                

def generate_id():
    return str(uuid.uuid4())[:8] 



def number_to_letters(num, uppercase=True):
    letters = ""
    while num > 0:
        num -= 1
        letters = chr((num % 26) + (65 if uppercase else 97)) + letters
        num //= 26
    return letters



def show_topics_and_categories():
    if not topics:
        print("No topics are currently defined.")
        return

    for i, topic_info in enumerate(topics, start=1):
        topic_name = topic_info['topic_input'].value
        topic_id = topic_info.get('id', '?')
        
        condition_val = topic_info['condition'].value if 'condition' in topic_info else None
        prompt_val    = topic_info['prompt'].value    if 'prompt'    in topic_info else None
        
        print(f"Topic {i} (ID={topic_id}): {topic_name}")

        if condition_val:
            print(f"  Condition: {condition_val}")

        if prompt_val:
            print(f"  Prompt: {prompt_val}")

        categories = topic_info.get('categories', [])
        if not categories:
            print("    [No categories in this topic]")
        else:
            for j, (category_input, _, cat_id) in enumerate(categories, start=1):
                cat_name = category_input.value
                print(f"    {j}. {cat_name} (ID={cat_id})")



   
def add_topic(topic_name, 
              categories=[], 
              condition="", 
              prompt="INSTRUCTION: You are a helpful classifier. You select the correct of the possible categories "
        "for classifying a piece of text. The topic of the classification is '[TOPIC]'. "
        "The allowed categories are '[CATEGORIES]'. QUESTION: The text to be classified is '[TEXT]'. "
        "ANSWER: The correct category for this text is '"):
   
    global topic_id_counter
    topic_id_counter += 1
    
    if prompt is None:
        prompt = (
            "INSTRUCTION: You are a helpful classifier. You select the correct of the possible categories "
            "for classifying a piece of text. The topic of the classification is '[TOPIC]'. "
            "The allowed categories are '[CATEGORIES]'. QUESTION: The text to be classified is '[TEXT]'. "
            "ANSWER: The correct category for this text is '"
        )

    topic_input_mock = MockText(topic_name)
    condition_mock = MockText(condition)
    prompt_mock = MockText(prompt)
    
    topic_id = number_to_letters(topic_id_counter, uppercase=True)
    
    topic_info = {
        'id': topic_id,
        'topic_input': topic_input_mock,
        'condition': condition_mock,
        'categories': [],
        'prompt': prompt_mock,
        'categories_container': None,
        'topic_box': None,
        'performance_label': None,
        'checkPrompt_button': None,
        'num_iterations_input': None,
        'iteratePromptImprovements_button': None,
        'replacePrompt_button': None,
        
        'best_prompt_found': None,
        'best_prompt_accuracy': None,
        
        'category_counter': 0
    }
    
    for cat_str in categories:
        topic_info['category_counter'] += 1
        cat_id = number_to_letters(topic_info['category_counter'], uppercase=False)  # a, b, c ...
        category_tuple = (MockText(cat_str), None, cat_id)
        
        topic_info['categories'].append(category_tuple)
    
    topics.append(topic_info)
    return topic_info


def remove_topic(topic_id_str):
    for i, t in enumerate(topics):
        if t.get('id') == topic_id_str:
            del topics[i]
            print(f"Topic (ID={topic_id_str}) removed.")
            return 

    print(f"No topic found with ID={topic_id_str}.")
    
    
    
def add_category(topicId, categoryName, Condition=""):
    found_topic = None
    for topic_info in topics:
        if topic_info.get('id') == topicId:
            found_topic = topic_info
            break

    if not found_topic:
        print(f"No topic found with ID={topicId}")
        return

    if 'category_counter' not in found_topic:
        found_topic['category_counter'] = 0

    found_topic['category_counter'] += 1
    cat_id = number_to_letters(found_topic['category_counter'], uppercase=False)
    new_category_tuple = (MockText(categoryName), None, cat_id)

    if 'categories' not in found_topic:
        found_topic['categories'] = []
    found_topic['categories'].append(new_category_tuple)

    if Condition:
        if 'condition' not in found_topic or not hasattr(found_topic['condition'], 'value'):
            found_topic['condition'] = MockText("")
        found_topic['condition'].value = Condition

    print(f"Category '{categoryName}' (ID={cat_id}) added to topic '{topicId}'.")
    if Condition:
        print(f"  Updated topic condition to: {Condition}")
        
        
def remove_category(topicId, categoryId):
    for topic_info in topics:
        if topic_info.get('id') == topicId:
            categories = topic_info.get('categories', [])
            for i, (cat_input, cat_box, cat_id) in enumerate(categories):
                if cat_id == categoryId:
                    del categories[i]
                    print(f"Removed category (ID={categoryId}) from topic (ID={topicId}).")
                    return
            
            print(f"Category with ID='{categoryId}' not found in topic (ID={topicId}).")
            return

    print(f"No topic found with ID='{topicId}'.")
    
    
    
def save_topics(filename):
    data = []
    for topic_info in topics:
        topic_data = {
            'id': topic_info.get('id', ''),
            'topic_input': topic_info['topic_input'].value if 'topic_input' in topic_info else '',
            'condition': topic_info['condition'].value if 'condition' in topic_info else '',
            'prompt': topic_info['prompt'].value if 'prompt' in topic_info else '',
            'categories': []
        }

        for (cat_input, _, cat_id) in topic_info.get('categories', []):
            cat_name = cat_input.value
            topic_data['categories'].append({
                'id': cat_id,
                'value': cat_name
            })

        data.append(topic_data)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Topics saved to {filename}")
    
def load_topics(filename):
    global topics
    topics.clear()  

    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for topic_data in data:
        new_topic = {
            'id': topic_data.get('id', ''),
            'topic_input': MockText(topic_data.get('topic_input', '')),
            'condition': MockText(topic_data.get('condition', '')),
            'prompt': MockText(topic_data.get('prompt', '')),
            'categories': [],
            'category_counter': 0
        }

        for cat_dict in topic_data.get('categories', []):
            cat_id = cat_dict.get('id', '')
            cat_value = cat_dict.get('value', '')
            new_topic['category_counter'] += 1
            new_topic['categories'].append(
                (MockText(cat_value), None, cat_id)
            )

        topics.append(new_topic)

    print(f"Loaded {len(topics)} topic(s) from {filename}")
    
    
def add_condition(topicId, categoryId, conditionStr):
    found_topic = None
    for topic in topics:
        if topic.get('id') == topicId:
            found_topic = topic
            break

    if found_topic is None:
        print(f"No topic found with ID={topicId}.")
        return

    categories = found_topic.get('categories', [])
    
    for i, cat_tuple in enumerate(categories):
        if len(cat_tuple) == 3:
            (cat_input, cat_box, cat_id) = cat_tuple
            cat_condition = ""  # no condition yet
        else:
            (cat_input, cat_box, cat_id, cat_condition) = cat_tuple

        if cat_id == categoryId:
            new_cat_tuple = (cat_input, cat_box, cat_id, conditionStr)
            categories[i] = new_cat_tuple
            print(f"Condition '{conditionStr}' added to category (ID={categoryId}) in topic (ID={topicId}).")
            return

    print(f"No category (ID={categoryId}) found in topic (ID={topicId}).")
    
    
def remove_condition(topicId, categoryId):
    found_topic = None
    for topic in topics:
        if topic.get('id') == topicId:
            found_topic = topic
            break

    if found_topic is None:
        print(f"No topic found with ID={topicId}.")
        return

    categories = found_topic.get('categories', [])

    for i, cat_tuple in enumerate(categories):
        if len(cat_tuple) == 3:
            (cat_input, cat_box, cat_id) = cat_tuple
            cat_condition = None  
        else:
            (cat_input, cat_box, cat_id, cat_condition) = cat_tuple

        if cat_id == categoryId:
            if len(cat_tuple) == 3:
                print(f"Category (ID={categoryId}) in topic (ID={topicId}) has no condition.")
                return
            else:
                new_cat_tuple = (cat_input, cat_box, cat_id, "")
                categories[i] = new_cat_tuple
                print(f"Condition removed from category (ID={categoryId}) in topic (ID={topicId}).")
                return

    print(f"No category (ID={categoryId}) found in topic (ID={topicId}).")

    
    
def get_header_list():
    global topics
    headerlist=["Text"]
    for l in range(len(topics)):
        headerlist.append(topics[l]['topic_input'].value)
    return headerlist

    
def classify_table(dataset, withEvaluation=False, constrainedOutput=True, BATCH_SIZE=100):
    global LLM
    start_time = time.time()
    csv_file = dataset + ".csv"
    if not os.path.exists(csv_file):
        print(f"No {csv_file} file found.")
        return

    categoryConfusions = []
    for i, topic_info in enumerate(topics):
        cat_map = {}
        for (cat_input, _, _cat_id) in topic_info['categories']:
            cat_name = cat_input.value
            cat_map[cat_name] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        categoryConfusions.append(cat_map)

    numberOfCorrectResults = []
    numberOfRelevantAttempts = []
    
    
    
    def process_batch(batch_rows, start_index):
        global LLM
        """Process a batch of rows with given model"""
        batch_results = []
        batch_probs = []
        batch_indices = []
        for idx, row in enumerate(batch_rows):
            actual_index = start_index + idx
            if withEvaluation:
                result, probs = classify(
                    row[0],
                    isItASingleClassification=False,
                    constrainedOutput=constrainedOutput,
                    withEvaluation=True,
                    groundTruthRow=row
                )
            else:
                result, probs = classify(
                    row[0],
                    isItASingleClassification=False,
                    constrainedOutput=constrainedOutput
                )
            batch_results.append(result)
            batch_probs.append(probs)
            batch_indices.append(actual_index)
        return batch_results, batch_probs, batch_indices

    startcount = 1
    endcount = -1
    saveName = dataset + "_(result)"

    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        readerlist = list(reader)

        header_row = get_header_list()
        singleResult = [""]
        elementcounter = -1
        for element in header_row:
            elementcounter += 1
            if elementcounter == 0:
                singleResult.append(element)
            else:
                numberOfCorrectResults.append(0)
                numberOfRelevantAttempts.append(0)
                if withEvaluation:
                    singleResult.append(element + "(GroundTruth)")
                singleResult.append(element)
                singleResult.append("Probability")

        with open(saveName + ".csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writerow(singleResult)

        # Process data rows in batches with model reloading
        current_batch = []
        total_rows = len(readerlist[startcount:endcount if endcount != -1 else None])
        processed_rows = 0
        current_index = startcount
                
        
        for count, row in enumerate(readerlist[startcount:endcount if endcount != -1 else None], start=startcount):
            current_batch.append(row)
            
            if len(current_batch) >= BATCH_SIZE or processed_rows + len(current_batch) == total_rows:
                
                
                # Process batch
                batch_results, batch_probs, batch_indices = process_batch(current_batch, current_index)
                
                # Unload model to free GPU memory
                if(modelType=="Transformers"):
                    unload_model(LLM)
                
                    # Load fresh model instance
                    LLM = load_model()
                
                # Write results and update metrics
                for idx, (batch_row, result, prob) in enumerate(zip(current_batch, batch_results, batch_probs)):
                    # Use the correct row index from batch_indices
                    row_index = batch_indices[idx]
                    process_and_write_result(
                        row_index, batch_row, result, prob, 
                        saveName, withEvaluation, 
                        numberOfCorrectResults, 
                        numberOfRelevantAttempts,
                        categoryConfusions
                    )
                
                processed_rows += len(current_batch)
                print(f"Processed {processed_rows}/{total_rows} rows ({(processed_rows/total_rows)*100:.2f}%)")

                current_index += len(current_batch)
                
                # Clear batch
                current_batch = []

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Write evaluation metrics 
    if withEvaluation:
        with open(saveName + ".csv", 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)

            writer.writerow([
                "Topic", "Accuracy", "Correct Attempts", "Relevant Attempts",
                "Micro Acc", "Micro Prec", "Micro Recall", "Micro F1",
                "TP", "FP", "FN", "TN"
            ])

            for i, topic_info in enumerate(topics):
                sumTP = 0
                sumFP = 0
                sumFN = 0
                sumTN = 0

                cat_map = categoryConfusions[i]
                for cat_name, conf_map in cat_map.items():
                    sumTP += conf_map["TP"]
                    sumFP += conf_map["FP"]
                    sumFN += conf_map["FN"]
                    sumTN += conf_map["TN"]

                if numberOfRelevantAttempts[i] > 0:
                    accuracy = (numberOfCorrectResults[i] / numberOfRelevantAttempts[i]) * 100.0
                else:
                    accuracy = -1

                micro_accuracy = (sumTP / (sumTP + sumFN)) if (sumTP + sumFN) > 0 else 0.0
                micro_precision = (sumTP / (sumTP + sumFP)) if (sumTP + sumFP) > 0 else 0.0
                micro_recall = (sumTP / (sumTP + sumFN)) if (sumTP + sumFN) > 0 else 0.0
                micro_f1 = 0.0
                if micro_precision > 0 and micro_recall > 0:
                    micro_f1 = 2.0 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

                topic_name = topic_info['topic_input'].value
                writer.writerow([
                    topic_name,
                    f"{accuracy:.2f}%",
                    numberOfCorrectResults[i],
                    numberOfRelevantAttempts[i],
                    f"{micro_accuracy*100:.2f}%",
                    f"{micro_precision*100:.2f}%",
                    f"{micro_recall*100:.2f}%",
                    f"{micro_f1*100:.2f}%",
                    sumTP,
                    sumFP,
                    sumFN,
                    sumTN
                ])
            #Write elapsed Time
            writer.writerow(["Elapsed Time", f"{elapsed_time:.2f} seconds"])

    
    print(f"Classification of '{dataset}.csv' complete. Output written to '{saveName}.csv'.")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")



def process_and_write_result(count, row, result, prob, saveName, withEvaluation, 
                           numberOfCorrectResults, numberOfRelevantAttempts,
                           categoryConfusions):
    """Helper function to process and write a single result"""
    # Update confusion matrices
    for tIndex, predCategory in enumerate(result):
        if withEvaluation and (tIndex + 1) < len(row):
            groundTruth = row[tIndex + 1].strip()
            if groundTruth:
                for cat_name, conf_map in categoryConfusions[tIndex].items():
                    if cat_name == groundTruth and cat_name == predCategory:
                        conf_map["TP"] += 1
                    elif cat_name != groundTruth and cat_name == predCategory:
                        conf_map["FP"] += 1
                    elif cat_name == groundTruth and cat_name != predCategory:
                        conf_map["FN"] += 1
                    else:
                        conf_map["TN"] += 1

    # Prepare result row
    singleResult = [str(count), row[0]]
    tmpCount = 0
    for ret in result:
        tmpCount += 1
        if withEvaluation and tmpCount < len(row):
            ground_truth = row[tmpCount].strip()
            if ground_truth:
                numberOfRelevantAttempts[tmpCount - 1] += 1
                singleResult.append(ground_truth)
                singleResult.append(ret)
                if(len(prob)>0):
                    singleResult.append(prob.pop(0))
                else:
                    singleResult.append("")
                if ret == ground_truth:
                    numberOfCorrectResults[tmpCount - 1] += 1
            else:
                singleResult.append("")
                singleResult.append("")
                singleResult.append("")
        else:
            if not withEvaluation:
                singleResult.append(ret)
                if(len(prob)>0):
                    if not ret=="":
                        singleResult.append(prob.pop(0))
                    else:
                        singleResult.append("")
                else:
                    singleResult.append("")
            else:
                singleResult.append("UNDEFINED")
                singleResult.append("UNDEFINED")

    # Write to file
    with open(saveName + ".csv", 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(singleResult)
    
    
def check_prompt_performance_for_topic(
    topicId,
    dataset,
    constrainedOutput=True,
    groundTruthCol=None
):
    csv_file = dataset + ".csv"
    if not os.path.exists(csv_file):
        print(f"No {csv_file} file found.")
        return

    found_topic = None
    topic_index = None
    for i, t in enumerate(topics):
        if t.get('id') == topicId:
            found_topic = t
            topic_index = i
            break

    if found_topic is None:
        print(f"No topic found with ID={topicId}.")
        return

    if groundTruthCol is None:
        groundTruthCol = topic_index + 1
        
    print(topic_index)
    print(groundTruthCol)

    local_categories = [
        cat_input.value
        for (cat_input, _, cat_id) in found_topic.get('categories', [])
    ]

    relevant_attempts = 0
    correct_predictions = 0

    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        rows = list(reader)

    for rowIndex, row in enumerate(rows):
        if rowIndex == 0:
            continue

        if len(row) <= groundTruthCol:
            continue

        text_to_classify = row[0].strip()
        groundTruthCategoryName = row[groundTruthCol].strip()
        if not groundTruthCategoryName:
            continue

        prompt_template = found_topic['prompt'].value
        prompt_categories_str = "[" + ",".join(f"'{cat}'" for cat in local_categories) + "]"
        prompt = prompt_template.replace('[TOPIC]', found_topic['topic_input'].value)
        prompt = prompt.replace('[CATEGORIES]', prompt_categories_str)
        prompt = prompt.replace('[TEXT]', text_to_classify)

        answer, bestRelProb = getAnswer(prompt, local_categories, constrainedOutput)

        relevant_attempts += 1
        if answer == groundTruthCategoryName:
            correct_predictions += 1
            
        print("Answer:",answer," GT:",groundTruthCategoryName)

    if relevant_attempts > 0:
        accuracy = (correct_predictions / relevant_attempts) * 100.0
        print(f"Topic (ID={topicId}) => Accuracy: {accuracy:.2f}%  "
              f"({correct_predictions} / {relevant_attempts} attempts)")
    else:
        print(f"Topic (ID={topicId}): No relevant attempts (no rows with non-empty groundTruth).")
        

        
        
def getLLMImprovedPromptWithFeedback(old_prompt, old_accuracy, topic_info):
    global PromptLLM
    topic_name = topic_info['topic_input'].value
    category_list = [cat_input.value for (cat_input, _, _cat_id) in topic_info['categories']]
    category_str = ", ".join(category_list) if category_list else "No categories defined"

    system_content = (
        f"You are an advanced prompt engineer.\n"
        f"The classification topic is '{topic_name}'.\n"
        f"The available categories for this topic are: {category_str}\n"
        "Rewrite the user's prompt to achieve higher accuracy on classification tasks.\n"
        "You MUST keep the placeholder [TEXT].\n"
        "IMPORTANT: Output ONLY the final prompt, wrapped in triple backticks.\n"
        "No commentary, bullet points, or explanations.\n"
        "The new prompt should be in English.\n"
    )

    user_content = (
        f"Previously, the prompt achieved an accuracy of {old_accuracy:.2f}%. \n"
        "Here is the old prompt:\n\n"
        f"{old_prompt}\n\n"
        "Please rewrite/improve this prompt. Keep [TEXT]. "
        "Wrap your entire revised prompt in triple backticks, with no extra lines."
    )

    if promptModelType in ("OpenAI", "DeepInfra"):
        try:
            completion = client.chat.completions.create(
                model=promptModel,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=250,
                temperature=0.7
            )
            improved_prompt = completion.choices[0].message.content.strip()

            match = re.search(r"```(.*?)```", improved_prompt, flags=re.DOTALL)
            if match:
                improved_prompt = match.group(1).strip()
            else:
                print("Warning: The LLM did not provide triple backticks. Using full text.")

            print("Improved Prompt:", improved_prompt)  # Debug

            if not improved_prompt or "[TEXT]" not in improved_prompt:
                print("Warning: The improved prompt is empty or lacks [TEXT]. Reverting to old prompt.")
                return old_prompt

            return improved_prompt

        except Exception as e:
            print(f"Error calling OpenAI/DeepInfra: {e}")
            return old_prompt

    else:
        try:
            base_instruction = system_content
            improvement_request = (
                f"{base_instruction}\n\n"
                f"Original prompt:\n{old_prompt}\n"
            )

            
            if promptInferenceType=="transformers":
                inputs = promptModelTokenizer(improvement_request, return_tensors="pt").to("cuda")
                period_id = promptModelTokenizer.encode(".", add_special_tokens=False)[-1]
                question_id = tokenizer.encode("?", add_special_tokens=False)[-1]
                exclamation_id = tokenizer.encode("!", add_special_tokens=False)[-1]
                stopping_criteria = StoppingCriteriaList([
                    StopOnTokens([period_id, question_id, exclamation_id])
                ])
                outputs = PromptLLM.generate(inputs.input_ids, temperature=0.01, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95, stopping_criteria=stopping_criteria)
                new_prompt=tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

            elif promptInferenceType=="guidance":
                script = promptModelGuidance + f" {improvement_request}" + gen(max_tokens=250, name='improvedPrompt')
                new_prompt = script["improvedPrompt"]

            if not new_prompt or "[TEXT]" not in new_prompt:
                print("Warning: The improved prompt is empty or lacks [TEXT]. Reverting to old prompt.")
                return old_prompt

            return new_prompt

        except Exception as e:
            print(f"Error calling local approach: {e}")
            return old_prompt
        

        
        



        
def evaluate_prompt_accuracy(topic_info, prompt, dataset, constrainedOutput, groundTruthCol):
    csv_file = dataset + ".csv"
    if not os.path.exists(csv_file):
        print(f"No {csv_file} file found.")
        return 0.0

    local_categories = [cat_input.value for (cat_input, _, _) in topic_info.get('categories', [])]
    relevant_attempts = 0
    correct_predictions = 0

    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        rows = list(reader)

    for i, row in enumerate(rows):
        if i == 0: 
            continue
        if len(row) <= groundTruthCol:
            continue

        text_to_classify = row[0].strip()
        groundTruthCategoryName = row[groundTruthCol].strip()
        if not groundTruthCategoryName:
            continue

        prompt_categories_str = "[" + ",".join(f"'{cat}'" for cat in local_categories) + "]"
        final_prompt = prompt.replace('[TOPIC]', topic_info['topic_input'].value)
        final_prompt = final_prompt.replace('[CATEGORIES]', prompt_categories_str)
        final_prompt = final_prompt.replace('[TEXT]', text_to_classify)

        answer, bestRelProb = getAnswer(final_prompt, local_categories, constrainedOutput)
        relevant_attempts += 1
        if answer == groundTruthCategoryName:
            correct_predictions += 1

    if relevant_attempts > 0:
        return (correct_predictions / relevant_attempts) * 100.0
    return 0.0    

        
def improve_prompt(topicId, dataset, constrainedOutput=True, groundTruthCol=None, num_iterations=10):
    found_topic = next((t for t in topics if t.get('id') == topicId), None)
    if not found_topic:
        print(f"No topic found with ID {topicId}.")
        return
    
    topic_index = topics.index(found_topic)
    if groundTruthCol is None:
        groundTruthCol = (topic_index * 2) + 1

    old_prompt = found_topic['prompt'].value
    old_accuracy = evaluate_prompt_accuracy(found_topic, old_prompt, dataset, constrainedOutput, groundTruthCol)

    best_prompt = old_prompt
    best_accuracy = old_accuracy

    print("========================================")
    print(f"Starting iterative prompt improvement for topic '{found_topic['id']}'")
    print(f"Baseline accuracy: {best_accuracy:.2f}%")
    print("========================================")

    for iteration in range(1, num_iterations + 1):
        new_prompt = getLLMImprovedPromptWithFeedback(best_prompt, best_accuracy, found_topic)
        if "[TEXT]" not in new_prompt:
            print("Warning: The improved prompt lost [TEXT]. Skipping iteration.")
            continue

        new_accuracy = evaluate_prompt_accuracy(found_topic, new_prompt, dataset, constrainedOutput, groundTruthCol)
        diff = new_accuracy - best_accuracy

        print(f"Iteration {iteration}:")
        print(f"New prompt accuracy: {new_accuracy:.2f}% (was {best_accuracy:.2f}%)")

        if diff > 0.001:
            print(f"Improvement found (+{diff:.2f}%). Updating best prompt.")
            best_prompt = new_prompt
            best_accuracy = new_accuracy
        else:
            print("No improvement. Keeping current best prompt.")
        print("----------------------------------------")

    print("========================================")
    print(f"Final best accuracy: {best_accuracy:.2f}%")
    print("Best prompt:\n", best_prompt)
    print("========================================\n")

    if best_accuracy > old_accuracy:
        found_topic['best_prompt_found'] = best_prompt
        found_topic['best_prompt_accuracy'] = best_accuracy
    else:
        found_topic['best_prompt_found'] = None
        found_topic['best_prompt_accuracy'] = None
        
        
def setPrompt(topicId, newPrompt):
    for topic in topics:
        if topic.get('id') == topicId:
            if 'prompt' in topic and hasattr(topic['prompt'], 'value'):
                topic['prompt'].value = newPrompt
            else:
                topic['prompt'] = MockText(newPrompt)
            print(f"Prompt for topic ID {topicId} updated.")
            return

    print(f"Topic with ID {topicId} not found.")

def removeAllTopics():
    global topics, topic_id_counter, previous_results, selectOptions
    
    topics.clear()            
    topic_id_counter = 0         
    previous_results.clear()    
    if 'selectOptions' in globals():
        selectOptions.clear()   
    
    print("All topics have been removed, counters reset, and related data cleared.")
    
    
    
    

                
        
class MockText:
    def __init__(self, value: str):
        self.value = value
















        





#### UI
def check_temporary_prompt_performance(topic_info, custom_prompt):
    global classify_CSV_box
    global with_Evaluation_Checkbox
    global constrainedOutputCheckbox
    """
    Evaluate how well a *custom* prompt performs (without overwriting topic_info['prompt']).
    Returns the accuracy as a float (0..100).
    Skips rows with empty groundTruth.
    """
    topic_index = topics.index(topic_info)
    dataset_name = classify_CSV_box.children[0].value
    withEvaluation = with_Evaluation_Checkbox.value
    constrainedOutput = constrainedOutputCheckbox.value

    csv_file = dataset_name + ".csv"
    if not os.path.exists(csv_file):
        return 0.0 

    local_categories = [cat_input.value for (cat_input, _, cat_id) in topic_info['categories']]

    relevant_attempts = 0
    correct_predictions = 0

    with open(csv_file, 'r', encoding='utf-8') as file:
        rows = list(csv.reader(file, delimiter=';'))

    for i, row in enumerate(rows):
        if i == 0:
            continue  # skip header

        if len(row) <= topic_index:
            continue

        text_to_classify = row[0]
        if withEvaluation:
            groundTruthCategoryName = row[topic_index + 1].strip() if len(row) > topic_index else ""
        else:
            groundTruthCategoryName = ""

        if not groundTruthCategoryName:
            # Condition not met or empty groundTruth => skip
            continue

        # Build categories string for the custom prompt
        prompt_categories_str = "[" + ",".join(f"'{cat}'" for cat in local_categories) + "]"

        # Replace placeholders in the custom prompt
        tmp_prompt = custom_prompt.replace('[TOPIC]', topic_info['topic_input'].value)
        tmp_prompt = tmp_prompt.replace('[CATEGORIES]', prompt_categories_str)
        tmp_prompt = tmp_prompt.replace('[TEXT]', text_to_classify)

        # Classify
        answer, bestRelProb = getAnswer(tmp_prompt, local_categories, constrainedOutput)
        

        relevant_attempts += 1
        if answer == groundTruthCategoryName:
            correct_predictions += 1

    if relevant_attempts > 0:
        return (correct_predictions / relevant_attempts) * 100.0
    return 0.0


def openInterface():
    global classify_CSV_box
    global with_Evaluation_Checkbox
    global constrainedOutputCheckbox
    global topics
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import json
    import os
    def on_value_change(change):
        if change['new']:  # Check if the new value is True (checkbox checked)
            freeText_input.layout.visibility = 'hidden'
            classify_CSV_box.layout.visibility = 'visible'
        else:
            freeText_input.layout.visibility = 'visible'
            classify_CSV_box.layout.visibility = 'hidden'
    def add_topic_ui(b=None, topic_data=None):
        global topic_id_counter
        topic_id_counter += 1
        topic_id = number_to_letters(topic_id_counter, uppercase=True)

        topic_input = widgets.Text(
            value=topic_data['topic_input'] if topic_data else '',
            description=f'Topic {topic_id}:',
            disabled=False
        )
        condition_input = widgets.Text(
            value=topic_data['condition'] if topic_data and 'condition' in topic_data else '',
            description='Condition:',
            disabled=False
        )

        add_category_button = widgets.Button(description="Add Category")
        remove_topic_button = widgets.Button(description="Remove Topic")
        categories_container = widgets.VBox()

        # Prompt
        tmpPrompt = topic_data['prompt'] if topic_data and 'prompt' in topic_data else (
            "INSTRUCTION: You are a helpful classifier. You select the correct of the possible categories "
            "for classifying a piece of text. The topic of the classification is '[TOPIC]'. "
            "The allowed categories are '[CATEGORIES]'. QUESTION: The text to be classified is '[TEXT]'. "
            "ANSWER: The correct category for this text is '"
        )
        prompt_input = widgets.Text(value=tmpPrompt, description='Prompt:', disabled=False)

        # Performance label
        performance_label = widgets.Label(value="")

        # Buttons: Check Prompt Performance, iteration input, iteration improvement, and "Replace Prompt"
        checkPrompt_button = widgets.Button(description="Check Prompt Performance")
        checkPrompt_button.layout.display = "none"

        num_iterations_input = widgets.IntText(
            value=1,
            description='Iterations:',
            layout=widgets.Layout(width='120px')
        )
        num_iterations_input.layout.display = "none"

        iteratePromptImprovements_button = widgets.Button(description="Iterate Prompt Improvement")
        iteratePromptImprovements_button.layout.display = "none"

        replacePrompt_button = widgets.Button(description="Replace Prompt (New Accuracy: ?)")
        replacePrompt_button.layout.display = "none"  # hidden unless a better prompt is found



        # Row 1: prompt, performance label, + add/remove category
        prompt_row = widgets.HBox([
            prompt_input,
            performance_label,
            add_category_button,
            remove_topic_button
        ])

        # Row 2: performance/iteration controls + the hidden "Replace Prompt" button
        performance_row = widgets.HBox([
            checkPrompt_button,
            num_iterations_input,
            iteratePromptImprovements_button,
            replacePrompt_button
        ])

        topic_box = widgets.VBox([
            widgets.HBox([topic_input, condition_input]),
            prompt_row,
            performance_row,
            categories_container
        ])

        # Construct topic_info now that we have all widgets
        topic_info = {
            'id': topic_id,
            'topic_input': topic_input,
            'condition': condition_input,
            'categories': [],
            'prompt': prompt_input,
            'categories_container': categories_container,
            'topic_box': topic_box,

            'performance_label': performance_label,
            'checkPrompt_button': checkPrompt_button,
            'num_iterations_input': num_iterations_input,
            'iteratePromptImprovements_button': iteratePromptImprovements_button,
            'replacePrompt_button': replacePrompt_button,

            # We'll store the best prompt found after iteration in here
            'best_prompt_found': None,
            'best_prompt_accuracy': None
        }

        # Callbacks
        checkPrompt_button.on_click(lambda btn: check_prompt_performance_for_topic_ui(topic_info))

        def do_iterative_prompt_improvement_callback(btn):
            replacePrompt_button.layout.display = "none"
            iterations = num_iterations_input.value
            iterative_prompt_improvement_ui(topic_info, iterations)

        iteratePromptImprovements_button.on_click(do_iterative_prompt_improvement_callback)

        # When "Replace Prompt" is clicked, update the prompt input with the new best prompt
        def replace_prompt_callback(btn):
            if topic_info['best_prompt_found'] is not None:
                topic_info['prompt'].value = topic_info['best_prompt_found']
                # Optionally re-check performance immediately
                check_prompt_performance_for_topic_ui(topic_info)
            # Hide the button again or keep it, your choice:
            topic_info['replacePrompt_button'].layout.display = "none"

        replacePrompt_button.on_click(replace_prompt_callback)

        # Category add/remove callbacks
        add_category_button.on_click(lambda btn: add_category_ui(topic_info))
        remove_topic_button.on_click(lambda btn: remove_topic_ui(topic_info))

        topics.append(topic_info)
        update_topics_container()

        # Load existing categories if present
        if topic_data and 'categories' in topic_data:
            for category_data in topic_data['categories']:
                add_category_ui(topic_info, category_data)
    def on_evaluation_toggle(change):
        new_value = change['new']  # True if checked, False if unchecked
        for topic_info in topics:
            # Check Prompt Performance
            if 'checkPrompt_button' in topic_info:
                topic_info['checkPrompt_button'].layout.display = "inline-block" if new_value else "none"

            # show/hide iteration input & button
            if 'num_iterations_input' in topic_info:
                topic_info['num_iterations_input'].layout.display = "inline-block" if new_value else "none"
            if 'iteratePromptImprovements_button' in topic_info:
                topic_info['iteratePromptImprovements_button'].layout.display = "inline-block" if new_value else "none"

            # Possibly clear performance label if disabling
            if not new_value and 'performance_label' in topic_info:
                topic_info['performance_label'].value = ""
                all_num_iterations_input.layout.display = "none"
                improveAllPrompts_button.layout.display = "none"
            else:
                all_num_iterations_input.layout.display = "inline-block"
                improveAllPrompts_button.layout.display = "inline-block"
    # Function to add new category input fields to a specific topic
    def add_category_ui(topic_info, category_data=None):
        topic_info.setdefault('category_counter', 0)  # Initialize the category counter if not present
        topic_info['category_counter'] += 1
        category_id = number_to_letters(topic_info['category_counter'], uppercase=False)  # Generate category ID as lowercase letters

        new_category_input = widgets.Text(
            value=category_data['value'] if category_data else '',
            description=f'Category {category_id}:',
            disabled=False
        )
        remove_button = widgets.Button(description="Remove Category")
        category_box = widgets.HBox([new_category_input, remove_button])

        remove_button.on_click(lambda btn: remove_category_ui(topic_info, category_box))

        topic_info['categories'].append((new_category_input, category_box, category_id))
        update_categories_container(topic_info)

    # Function to remove a specific category from a specific topic
    def remove_category_ui(topic_info, category_box):
        topic_info['categories'] = [(input_field, box, cat_id) 
                                    for (input_field, box, cat_id) in topic_info['categories'] 
                                    if box != category_box]
        update_categories_container(topic_info)

    # Function to update the categories container for a specific topic
    def update_categories_container(topic_info):
        topic_info['categories_container'].children = [box for _, box, _ in topic_info['categories']]
        if interface:
            update_topics_container()

    # Function to remove a specific topic
    def remove_topic_ui(topic_info):
        topics.remove(topic_info)
        update_topics_container()

    # Function to update the topics container
    def update_topics_container():
        topics_container.children = [td['topic_box'] for td in topics]         


    # Save topics to a local file
    def save_topics_ui(b):
        saveFileName = save_container.children[0].value + ".json"
        save_topics(saveFileName)


    # Load topics from a local file
    def load_topics_ui(b):
        global topic_id_counter
        global topics
        topic_id_counter=0
        loadFileName = load_container.children[0].value + ".json"
        if os.path.exists(loadFileName):
            with open(loadFileName, 'r') as f:
                data = json.load(f)
            topics = []
            topics_container.children = []
            for topic_data in data:
                add_topic_ui(topic_data=topic_data)
            if interface:
                print(f"Loaded topics from {loadFileName}")
        else:
            if interface:
                print(f"No {loadFileName} file found")
                
    def check_prompt_performance_for_topic_ui(
        topic_info,
        constrainedOutput=True,
        groundTruthCol=None
    ):
        topic_info['performance_label'].value = f" "
        # 1) Figure out which topic index this is among 'topics'
        topic_index = topics.index(topic_info)
        topic=topics[topic_index]
        topicId=topic.get('id')
        check_prompt_performance_for_topic(topicId,classify_CSV_box.children[0].value)


     

    def do_Classification_Button_Function():
        global selectOptions
        # Record the start time
        start_time = time.time()
        numberOfCorrectResults = []
        if groupClassificationCheckbox.value: #Group Classification
            classify_table(classify_CSV_box.children[0].value, with_Evaluation_Checkbox.value, constrainedOutputCheckbox.value, BATCH_SIZE=100)
            #groupClassification(classify_CSV_box.children[0].value,with_Evaluation_Checkbox.value,numberOfCorrectResults,constrainedOutputCheckbox.value)
        else:            
            classify(freeText_container.children[0].value, True, constrainedOutputCheckbox.value)
        end_time = time.time()
        elapsed_time = end_time - start_time
        saveName=classify_CSV_box.children[0].value+"_(result)"
        
        ##Save for GroupClassification
        if groupClassificationCheckbox.value:
            with open(saveName+"(single).csv",'a',newline='',encoding='utf-8') as f:
                singleResult=[]
                singleResult.append(str(elapsed_time)+"seconds")
                writer = csv.writer(f , delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
                writer.writerow(singleResult)
                singleResult=[]
                singleResult.append("Model:")
                singleResult.append(model)
                writer = csv.writer(f , delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
                writer.writerow(singleResult)
                singleResult=[]
                singleResult.append("ModelType:")
                singleResult.append(modelType)
                writer = csv.writer(f , delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
                writer.writerow(singleResult)
                for l in range(len(selectOptions)):
                    singleResult=["Prompt:"]
                    singleResult.append(topics[l]['prompt'].value)
                    singleResult.append("Topic:")
                    singleResult.append(topics[l]['topic_input'].value)
                    singleResult.append("Categories:")
                    singleResult.append(selectOptions[l])
                    writer = csv.writer(f , delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
                    writer.writerow(singleResult)

                singleResult=["Time:"]
                singleResult.append(str(elapsed_time)+"seconds")
                writer = csv.writer(f , delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
                writer.writerow(singleResult)

        print(f"This classification took {elapsed_time:.2f} seconds")






    def getLLMImprovedPromptWithFeedback_ui(old_prompt, old_accuracy, topic_info):
        """
        Similar to getLLMImprovedPrompt, but also provides the last iteration's accuracy 
        to the LLM so it can attempt to improve further. Additionally, 
        we add the topic name AND the categories in the system content for context.
        """
        topic_name = topic_info['topic_input'].value  # e.g., "Breast Cancer Family History"
        # Build a list of category names for this topic
        category_list = [cat_input.value for (cat_input, _, cat_id) in topic_info['categories']]
        category_str = ", ".join(category_list) if category_list else "No categories defined"

        # We'll reuse your existing model selection logic
        if promptModelType in ("OpenAI", "DeepInfra"):
            try:
                # Include topic name and categories in the system_content
                system_content = (
                    f"You are an advanced prompt engineer.\n"
                    f"The text to be classified is a german mammography report\n"
                    f"The classification topic is '{topic_name}'.\n"
                    f"The available categories for this topic are: {category_str}\n"
                    "Rewrite the user's prompt to achieve higher accuracy on classification tasks.\n"
                    "You MUST keep the placeholder [TEXT].\n"
                    "IMPORTANT: Output ONLY the final prompt, wrapped in triple backticks.\n"
                    "No commentary, bullet points, or explanations."
                    "The new prompt should be in english"
                )

                user_content = (
                    f"Previously, the prompt achieved an accuracy of {old_accuracy:.2f}%. \n"
                    "Here is the old prompt:\n\n"
                    f"{old_prompt}\n\n"
                    "Please rewrite/improve this prompt. Keep [TEXT]. "
                    "Wrap your entire revised prompt in triple backticks, with no extra lines."
                )

                completion = client.chat.completions.create(
                    model=promptModel,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=250,
                    temperature=0.7
                )
                improved_prompt = completion.choices[0].message.content.strip()

                # Now parse out whatever is between triple backticks.
                match = re.search(r"```(.*?)```", improved_prompt, flags=re.DOTALL)
                if match:
                    # Extract the text inside ```...```
                    improved_prompt = match.group(1).strip()
                else:
                    print("Warning: The LLM did not provide triple backticks. Using full text.")

                print("Improved Prompt:", improved_prompt)  # Debug

                # Basic check
                if not improved_prompt or "[TEXT]" not in improved_prompt:
                    print("Warning: The improved prompt is empty or lacks [TEXT]. Reverting to old prompt.")
                    return old_prompt

                return improved_prompt

            except Exception as e:
                print(f"Error calling OpenAI/DeepInfra: {e}")
                return old_prompt

        else:
            # Local approach
            try:
                base_instruction = (
                    f"The previous iteration got {old_accuracy:.2f}% accuracy. \n"
                    f"The classification topic is '{topic_name}'.\n"
                    f"The available categories are: {category_str}\n"
                    "Please refine the prompt below to achieve higher accuracy, keeping [TEXT]."
                )
                improvement_request = f"{base_instruction}\n\nOriginal prompt:\n{old_prompt}\n"

                if constrainedOutputCheckbox.value:
                    script = ModelGuidance + f" {improvement_request}" + select(
                        options=["ImprovementA", "ImprovementB", "ImprovementC"],
                        name='improvedPrompt'
                    )
                    new_prompt = script["improvedPrompt"]
                else:
                    script = ModelGuidance + f" {improvement_request}" + gen(max_tokens=250, name='improvedPrompt')
                    new_prompt = script["improvedPrompt"]

                if not new_prompt or "[TEXT]" not in new_prompt:
                    print("Warning: The improved prompt is empty or lacks [TEXT]. Reverting to old prompt.")
                    return old_prompt

                return new_prompt

            except Exception as e:
                print(f"Error calling local approach: {e}")
                return old_prompt




    def iterative_prompt_improvement_ui(topic_info, num_iterations=10):
        old_prompt = topic_info['prompt'].value
        old_accuracy = get_current_accuracy(topic_info)
        if old_accuracy <= 0:
            old_accuracy = check_temporary_prompt_performance(topic_info, old_prompt)

        best_prompt = old_prompt
        best_accuracy = old_accuracy

        print("========================================")
        print(f"Starting iterative prompt improvement for topic '{topic_info['id']}'")
        print(f"Baseline accuracy: {best_accuracy:.2f}%")
        print("========================================")
        
        if best_accuracy <100:
            for iteration in range(1, num_iterations + 1):
                new_prompt = getLLMImprovedPromptWithFeedback_ui(best_prompt, best_accuracy, topic_info)
                if "[TEXT]" not in new_prompt:
                    print("Warning: The improved prompt lost [TEXT]. Skipping iteration.")
                    continue

                new_accuracy = check_temporary_prompt_performance(topic_info, new_prompt)
                diff = new_accuracy - best_accuracy

                print(f"New prompt accuracy: {new_accuracy:.2f}% (was {best_accuracy:.2f}%)")

                if diff > 0.001:
                    print(f"Improvement found (+{diff:.2f}%). Updating best prompt.")
                    best_prompt = new_prompt
                    best_accuracy = new_accuracy
                else:
                    print(f"No improvement. Keeping old prompt.")
                print("----------------------------------------")

            print("========================================")
            print(f"Final best accuracy: {best_accuracy:.2f}%")
            print("Best prompt:\n", best_prompt)
            print("========================================\n")

            # If the best accuracy is better than the original, store it in topic_info
            if best_accuracy > old_accuracy:
                topic_info['best_prompt_found'] = best_prompt
                topic_info['best_prompt_accuracy'] = best_accuracy

                # Update the label on the replacePrompt_button
                label_text = f"Replace Prompt (New Accuracy: {best_accuracy:.2f}%)"
                topic_info['replacePrompt_button'].description = label_text
                # Show the button
                topic_info['replacePrompt_button'].layout.display = "inline-block"
            else:
                # Otherwise, hide the button (or keep hidden)
                topic_info['replacePrompt_button'].layout.display = "none"
                topic_info['best_prompt_found'] = None
                topic_info['best_prompt_accuracy'] = None


    def do_iterative_prompt_improvement_for_all_ui(btn):
            iterations = all_num_iterations_input.value
            print(f"Starting iterative improvement of all prompts for {iterations} iterations each.")

            # For each topic, call iterative_prompt_improvement
            for i, topic_info in enumerate(topics):
                print(f"\n>>> Improving topic #{i+1} (ID {topic_info['id']})")
                iterative_prompt_improvement_ui(topic_info, iterations)
                
    groupClassificationCheckbox=widgets.Checkbox(value=False, description='Group Classification', disabled=False )
    display(groupClassificationCheckbox)

    groupClassificationCheckbox.observe(on_value_change, names='value')

    #ConstrainedOutput Checkbox
    constrainedOutputCheckbox=widgets.Checkbox(
        value=True,
        description='Constrained Output',
        disabled=False
    )
    display(constrainedOutputCheckbox)



    #FreeText Container
    freeText_container = widgets.VBox()
    display(freeText_container)

    freeText_input = widgets.Text(
        value="",
        description='FreeText:',
        disabled=False
    )

    freeText_container.children = [freeText_input]



    #Classify CSV Container
    classify_CSV_input = widgets.Text(
            value="data",
            description='CSV File:',
            disabled=False
        )

    with_Evaluation_Checkbox=widgets.Checkbox(
        value=False,
        description='Evaluation',
        disabled=False
    )
    with_Evaluation_Checkbox.observe(on_evaluation_toggle, names='value')
    

    classify_CSV_box = widgets.HBox([classify_CSV_input,with_Evaluation_Checkbox])
    classify_CSV_box.layout.visibility = "hidden"

    display(classify_CSV_box)
    
    
    
    all_num_iterations_input = widgets.IntText(
        value=1,
        description='All Iterations:',
        layout=widgets.Layout(width='150px')
    )
    # Hide or show only when with_Evaluation is True, at your preference:
    all_num_iterations_input.layout.display = "none"
    
    # The button to iteratively improve all prompts
    improveAllPrompts_button = widgets.Button(description="Iteratively Improve all Prompts")
    improveAllPrompts_button.layout.display = "none"

    display(widgets.HBox([all_num_iterations_input, improveAllPrompts_button]))
    improveAllPrompts_button.on_click(do_iterative_prompt_improvement_for_all_ui)




    # Container to hold all topics
    topics_container = widgets.VBox()
    display(topics_container)

    # List to keep track of all topics and their categories


    # Function to add a new topic with its own category container


    plus_topic_button = widgets.Button(description="Add Topic")
    plus_topic_button.on_click(add_topic_ui)
    display(plus_topic_button)




    save_container = widgets.HBox()
    display(save_container)



    saveFileText_input = widgets.Text(
        value="topics",
        description='File Name:',
        disabled=False
    )


    save_button = widgets.Button(description="Save Topics")
    save_button.on_click(save_topics_ui)
    save_container.children = [saveFileText_input,save_button]





    load_container = widgets.HBox()
    display(load_container)



    loadFileText_input = widgets.Text(
        value="topics",
        description='File Name:',
        disabled=False
    )


    load_button = widgets.Button(description="Load Topics")
    load_button.on_click(load_topics_ui)
    load_container.children = [loadFileText_input,load_button]





    show_all_button = widgets.Button(description="Show Topics and Categories")
    show_all_button.on_click(lambda b: show_topics_and_categories())
    display(show_all_button)




    classify_button = widgets.Button(description="Do Classification")
    classify_button.on_click(lambda b: do_Classification_Button_Function())
    display(classify_button)
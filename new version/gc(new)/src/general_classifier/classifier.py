import uuid
import ast
import json # Retained for now, can be removed if a linter confirms unused
import os   # Retained for now, can be removed if a linter confirms unused
from .models import ModelManager
from .persistence import save_data_to_json, load_data_from_json

class MockText:
    def __init__(self, value: str):
        self.value = value

class GeneralClassifier:
    def __init__(self):
        self.topics = []
        self.topic_id_counter = 0
        self.previous_results = {}
        self.model_manager = ModelManager()

    def set_model(self, model_name, model_type="Transformers", api_key="", inference_type="transformers"):
        self.model_manager.set_model(model_name, model_type, api_key, inference_type)

    def add_topic(self, topic_name, 
                  categories=[], 
                  condition="", 
                  prompt="INSTRUCTION: You are a helpful classifier. You select the correct of the possible categories "
        "for classifying a piece of text. The topic of the classification is '[TOPIC]'. "
        "The allowed categories are '[CATEGORIES]'. QUESTION: The text to be classified is '[TEXT]'. "
        "ANSWER: The correct category for this text is '"):
        
        self.topic_id_counter += 1
        topic_id = f"T{self.topic_id_counter}"

        new_topic = {
            'id': topic_id,
            'topic_input': MockText(topic_name),
            'categories': [],
            'condition': MockText(condition),
            'prompt': MockText(prompt)
        }

        for category_name in categories:
            category_id = str(uuid.uuid4())
            new_topic['categories'].append((MockText(category_name), MockText(""), category_id))

        self.topics.append(new_topic)
        print(f"Topic '{topic_name}' added with ID {topic_id}.")
        return topic_id

    def remove_topic(self, topic_id_str):
        topic_found = False
        for i, topic in enumerate(self.topics):
            if topic.get('id') == topic_id_str:
                del self.topics[i]
                topic_found = True
                print(f"Topic with ID {topic_id_str} has been removed.")
                break
        if not topic_found:
            print(f"Topic with ID {topic_id_str} not found.")

    def add_category(self, topicId, categoryName, Condition=""):
        topic_found = False
        for topic in self.topics:
            if topic.get('id') == topicId:
                category_id = str(uuid.uuid4())
                topic['categories'].append((MockText(categoryName), MockText(Condition), category_id))
                topic_found = True
                print(f"Category '{categoryName}' added to topic ID {topicId} with ID {category_id}.")
                return category_id
        if not topic_found:
            print(f"Topic with ID {topicId} not found.")

    def remove_category(self, topicId, categoryId):
        for topic in self.topics:
            if topic.get('id') == topicId:
                for i, (_, _, cat_id) in enumerate(topic['categories']):
                    if cat_id == categoryId:
                        del topic['categories'][i]
                        print(f"Category with ID {categoryId} removed from topic ID {topicId}.")
                        return
        print(f"Category or topic not found.")

    def set_prompt(self, topicId, newPrompt):
        for topic in self.topics:
            if topic.get('id') == topicId:
                if 'prompt' in topic and hasattr(topic['prompt'], 'value'):
                    topic['prompt'].value = newPrompt
                else:
                    topic['prompt'] = MockText(newPrompt)
                print(f"Prompt for topic ID {topicId} updated.")
                return
        print(f"Topic with ID {topicId} not found.")

    def add_category_condition(self, topic_id, category_id, condition_str):
        """Adds or updates a condition for a specific category within a topic."""
        for topic in self.topics:
            if topic.get('id') == topic_id:
                for i, category_tuple in enumerate(topic['categories']):
                    # category_tuple is (name_mock, condition_mock, cat_id_in_topic)
                    if len(category_tuple) == 3 and category_tuple[2] == category_id:
                        condition_mock = category_tuple[1]
                        condition_mock.value = condition_str
                        print(f"Condition '{condition_str}' set for category ID {category_id} in topic ID {topic_id}.")
                        return
                print(f"Category ID {category_id} not found in topic ID {topic_id}.")
                return
        print(f"Topic ID {topic_id} not found.")

    def remove_category_condition(self, topic_id, category_id):
        """Removes a condition from a specific category within a topic (sets it to empty string)."""
        for topic in self.topics:
            if topic.get('id') == topic_id:
                for i, category_tuple in enumerate(topic['categories']):
                    if len(category_tuple) == 3 and category_tuple[2] == category_id:
                        condition_mock = category_tuple[1]
                        condition_mock.value = ""
                        print(f"Condition removed from category ID {category_id} in topic ID {topic_id}.")
                        return
                print(f"Category ID {category_id} not found in topic ID {topic_id}.")
                return
        print(f"Topic ID {topic_id} not found.")

    def remove_all_topics(self):
        self.topics.clear()
        self.topic_id_counter = 0
        self.previous_results.clear()
        print("All topics have been removed, counters reset, and related data cleared.")

    def classify(self, text, is_single_classification=True, constrained_output=True, with_evaluation=False, ground_truth_row=None):
        select_options = []
        for topic_data in self.topics:
            tmp_select_options = "["
            for category_input, _, _ in topic_data['categories']:
                tmp_select_options += "'" + category_input.value + "',"
            tmp_select_options = tmp_select_options[:-1] + "]"
            select_options.append(tmp_select_options)

        ret = []
        probs = []

        if with_evaluation and ground_truth_row is not None:
            for i, topic_info in enumerate(self.topics):
                ground_truth_category_name = ground_truth_row[i+1] 
                gt_cat_id = None
                for (cat_input, _, cat_id) in topic_info['categories']:
                    if cat_input.value == ground_truth_category_name:
                        gt_cat_id = cat_id
                        break
                self.previous_results[topic_info['id']] = gt_cat_id

        for l, topic_info in enumerate(self.topics):
            condition = topic_info['condition'].value.strip()
            if not self._evaluate_condition(condition):
                ret.append("")
                if is_single_classification:
                    print(f"Skipping {topic_info['topic_input'].value} due to unmet condition: {condition}")
                continue

            prompt = topic_info['prompt'].value
            prompt = prompt.replace('[TOPIC]', topic_info['topic_input'].value)
            prompt = prompt.replace('[CATEGORIES]', select_options[l])
            prompt = prompt.replace('[TEXT]', text)

            categories = ast.literal_eval(select_options[l])
            answer, best_rel_prob = self.model_manager.get_answer(prompt, categories, constrained_output)
            
            if is_single_classification:
                print(f"{topic_info['topic_input'].value}: {answer} (Relative Probability: {best_rel_prob})")
            
            ret.append(answer)
            probs.append(best_rel_prob)

            if not with_evaluation:
                chosen_category_id = None
                for category_input, _, category_id in topic_info['categories']:
                    if category_input.value == answer:
                        chosen_category_id = category_id
                        break
                self.previous_results[topic_info['id']] = chosen_category_id

        return ret, probs

    def _evaluate_condition(self, condition):
        if not condition:
            return True

        if "==" not in condition:
            print(f"Invalid condition format: {condition}")
            return False

        topic_id_to_check, expected_category_name = [x.strip() for x in condition.split("==", 1)]

        if topic_id_to_check not in self.previous_results:
            return False

        chosen_category_id = self.previous_results[topic_id_to_check]

        source_topic = next((topic for topic in self.topics if topic.get('id') == topic_id_to_check), None)

        if not source_topic:
            return False

        chosen_category_name = next((cat_input.value for cat_input, _, cat_id in source_topic['categories'] if cat_id == chosen_category_id), None)

        return chosen_category_name == expected_category_name

    def get_header_list(self):
        headerlist = ["Text"]
        for topic in self.topics:
            headerlist.append(topic['topic_input'].value)
        return headerlist

    def save_topics(self, filename):
        data = []
        for topic_info in self.topics:
            topic_data = {
                'id': topic_info.get('id', ''),
                'topic_input': topic_info['topic_input'].value if 'topic_input' in topic_info else '',
                'condition': topic_info['condition'].value if 'condition' in topic_info else '', # Topic-level condition
                'prompt': topic_info['prompt'].value if 'prompt' in topic_info else '',
                'categories': []
            }
            # Categories are (MockText(name), MockText(condition), cat_id)
            for (cat_name_mock, cat_condition_mock, cat_id) in topic_info.get('categories', []):
                cat_name = cat_name_mock.value
                cat_condition = cat_condition_mock.value # Category-level condition
                topic_data['categories'].append({
                    'id': cat_id,
                    'name': cat_name,       # Using 'name' key
                    'condition': cat_condition # Storing category condition
                })
            data.append(topic_data)
        
        save_data_to_json(data, filename) # Use the persistence function

    def load_topics(self, filename):
        data = load_data_from_json(filename)
        if data is None:
            # Error/file not found messages are handled by load_data_from_json
            return

        self.topics.clear()
        max_topic_num = 0

        for topic_data in data:
            new_topic = {
                'id': topic_data.get('id', ''),
                'topic_input': MockText(topic_data.get('topic_input', '')),
                'condition': MockText(topic_data.get('condition', '')),
                'prompt': MockText(topic_data.get('prompt', '')),
                'categories': []
            }

            for cat_dict in topic_data.get('categories', []):
                cat_id = cat_dict.get('id', '')
                cat_name = cat_dict.get('name', '')  # Expect 'name' key
                cat_condition = cat_dict.get('condition', '')  # Load category condition
                new_topic['categories'].append(
                    (MockText(cat_name), MockText(cat_condition), cat_id)
                )
            
            self.topics.append(new_topic)
            
            # Update topic counter
            topic_id_str = new_topic['id']
            if topic_id_str.startswith('T'):
                try:
                    topic_num = int(topic_id_str[1:])
                    if topic_num > max_topic_num:
                        max_topic_num = topic_num
                except ValueError:
                    pass # Ignore if the part after 'T' is not a number
        
        self.topic_id_counter = max_topic_num
        print(f"Topics loaded from {filename}")

    def show_topics_and_categories(self):
        if not self.topics:
            print("No topics are currently defined.")
            return

        for i, topic_info in enumerate(self.topics, start=1):
            topic_name = topic_info['topic_input'].value
            topic_id = topic_info.get('id', '?')
            
            condition_val = topic_info['condition'].value if 'condition' in topic_info else None
            prompt_val = topic_info['prompt'].value if 'prompt' in topic_info else None

            print(f"Topic {i} (ID={topic_id}): {topic_name}")

            if condition_val:
                print(f"  Condition: {condition_val}")

            if prompt_val:
                print(f"  Prompt: {prompt_val}")

            categories = topic_info.get('categories', [])
            if not categories:
                print("    [No categories in this topic]")
            else:
                for j, (cat_name_mock, cat_condition_mock, cat_id) in enumerate(categories, start=1):
                    cat_name = cat_name_mock.value
                    cat_condition = cat_condition_mock.value
                    display_str = f"    {j}. {cat_name} (ID={cat_id})"
                    if cat_condition:
                        display_str += f" (Condition: {cat_condition})"
                    print(display_str)

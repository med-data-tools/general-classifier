import sys
import os

# Add the 'src' directory to the Python path
# This allows importing modules from the 'src' directory as if it's a top-level package directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from general_classifier import GeneralClassifier

def run_test():
    """
    A comprehensive test script for the GeneralClassifier.
    This test demonstrates:
    - Basic topic and category creation.
    - Conditional topic execution.
    - Classification of multiple texts.
    """
    print("--- Initializing General Classifier Test ---")
    
    # 1. Initialize the classifier and set the model
    classifier = GeneralClassifier()
    classifier.set_model("openai-community/gpt2", inference_type="transformers", model_type="Transformers")

    # 2. Add a primary topic
    car_brand_topic_id = classifier.add_topic(
        topic_name="Car Brand",
        categories=["BMW", "Audi", "Mercedes", "Toyota"]
    )

    # 3. Add a conditional topic that depends on the first topic's result
    classifier.add_topic(
        topic_name="BMW Model Type",
        categories=["SUV", "Sedan", "Coupe"],
        condition=f"{car_brand_topic_id}==BMW"
    )

    # 4. Display all defined topics and their categories
    print("\n--- Defined Topics and Categories ---")
    classifier.show_topics_and_categories()
    print("-------------------------------------\n")

    # NEW: Test Category Conditions
    print("\n--- Testing Category Conditions ---")
    # First, get the category ID for "BMW" under "Car Brand" topic
    bmw_category_id = None
    # car_brand_topic_id is already available from when the topic was added
    for topic_details in classifier.topics:
        if topic_details['id'] == car_brand_topic_id:
            for cat_tuple in topic_details['categories']:
                # cat_tuple is (MockText(name), MockText(condition), id)
                if cat_tuple[0].value == "BMW": 
                    bmw_category_id = cat_tuple[2]
                    break
            if bmw_category_id:
                break
    
    if bmw_category_id:
        print(f"Targeting Topic ID: {car_brand_topic_id}, Category: BMW (ID: {bmw_category_id}) for condition tests.")

        # 4.1. Add a condition to the "BMW" category
        print("\n--- Adding condition to 'BMW' category ('engine_type==Petrol') ---")
        classifier.add_category_condition(car_brand_topic_id, bmw_category_id, "engine_type==Petrol")
        classifier.show_topics_and_categories()
        print("-------------------------------------\n")

        # 4.2. Remove the condition from the "BMW" category
        print("\n--- Removing condition from 'BMW' category ---")
        classifier.remove_category_condition(car_brand_topic_id, bmw_category_id)
        classifier.show_topics_and_categories()
        print("-------------------------------------\n")
    else:
        print("ERROR: Could not find 'BMW' category ID in 'Car Brand' topic to run condition tests.")
    print("--- End of Category Condition Tests ---\n")

    # 5. Classify a text about a BMW
    print("--- Test Case 1: Classifying a text about a BMW ---")
    text1 = "The new BMW X5 has impressive features and a powerful engine."
    print(f'Text: "{text1}"')
    results1, probabilities1 = classifier.classify(
        text=text1,
        is_single_classification=True,
        constrained_output=True
    )
    print(f"\nClassification results: {results1}")
    print(f"Confidence scores: {probabilities1}")
    print("---------------------------------------------------\n")

    # 6. Classify a text about an Audi
    print("--- Test Case 2: Classifying a text about an Audi ---")
    text2 = "The Audi A4 is a popular sedan known for its luxury and performance."
    print(f'Text: "{text2}"')
    results2, probabilities2 = classifier.classify(
        text=text2,
        is_single_classification=True,
        constrained_output=True
    )
    print(f"\nClassification results: {results2}")
    print(f"Confidence scores: {probabilities2}")
    print("---------------------------------------------------\n")

    # 7. Test saving and loading topics
    print("--- Test Case 3: Saving and Loading Topics ---")
    topics_file = "topics.json"
    classifier.save_topics(topics_file)

    # Create a new classifier and load the topics
    new_classifier = GeneralClassifier()
    new_classifier.load_topics(topics_file)

    print("\n--- Topics Loaded into New Classifier ---")
    new_classifier.show_topics_and_categories()
    print("-----------------------------------------")


if __name__ == "__main__":
    run_test()
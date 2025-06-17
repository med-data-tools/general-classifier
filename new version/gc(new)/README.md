# general-classifier

**general-classifier** is a Python package designed for multi-topic text classification leveraging Large Language Models (LLMs). It allows users to define multiple classification topics, manage categories within each topic, classify text data using various language models, evaluate classification performance, and iteratively improve classification prompts.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [1. Define Topics and Categories](#1-define-topics-and-categories)
  - [2. Set Models](#2-set-models)
  - [3. Classify a Single Text](#3-classify-a-single-text)
  - [4. Classify a Dataset](#4-classify-a-dataset)
  - [5. Evaluate Prompt Performance](#5-evaluate-prompt-performance)
  - [6. Improve Prompts Iteratively](#6-improve-prompts-iteratively)
  - [7. Managing Topics and Categories](#7-managing-topics-and-categories)
  - [8. Saving and Loading Topics](#8-saving-and-loading-topics)
- [Advanced Features](#advanced-features)
  - [Conditional Classification](#conditional-classification)
  - [Batch Processing](#batch-processing)
  - [GPU Memory Management](#gpu-memory-management)
  - [Interactive Interface](#interactive-interface)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Multi-topic Classification**:  
  Classify text across multiple defined topics simultaneously, each with its own set of categories.

- **Memory-efficient Batch Processing**:  
  Process large datasets in batches with automatic GPU memory management and model reloading between batches.

- **Dynamic Topic & Category Management**:  
  Easily add, remove, and manage multiple classification topics and their respective categories.

- **Flexible Model Integration**:  
  Supports integration with:
  - Local Transformers models (with torch)
  - OpenAI API models
  - DeepInfra hosted models
  - Support for both direct model output and guided/constrained prediction

- **Performance Evaluation**:  
  Comprehensive evaluation metrics including:
  - Per-topic accuracy
  - Confusion matrices
  - Micro precision, recall, and F1 scores

- **Iterative Prompt Improvement**:  
  Leverage LLMs to automatically refine classification prompts and improve accuracy over time.

- **Interactive Interface**:  
  Optional Jupyter widget-based interface for easy management of topics, categories, and classification tasks.

- **Conditional Classification**:  
  Support for dependent classifications with conditions based on previous results.

## Installation

Ensure you have Python 3.7 or higher installed. Install the required dependencies using `pip`:

```bash
pip install torch transformers openai guidance ipywidgets
```

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- OpenAI Python client (for OpenAI API)
- Guidance (for guided generation)
- IPython/Jupyter (for interactive interface)

## Quick Start

### 1. Define Topics and Categories

Begin by defining classification topics and their respective categories.

```python
from general_classifier import gc

# Add a new topic with categories
gc.add_topic(
    topic_name="Car Brands",
    categories=["BMW", "Audi", "Mercedes"]
)

# Add another category to the existing topic
gc.add_category("A", categoryName="Toyota")

# Display all defined topics and their categories
gc.show_topics_and_categories()
```

### 2. Set Models

Configure the main classification model and optionally a separate model for prompt improvement.

```python

# Set the main classification model (e.g., a local Transformers model)
gc.setModel(
    newModel="meta-llama/Llama-2-7b-chat-hf", 
    newModelType="Transformers",
    newInferenceType="transformers"  # Options: "transformers", "guidance"
)

# Optionally set a separate model for prompt improvement
gc.setPromptModel(
    newPromptModel="gpt-4", 
    newPromptModelType="OpenAI", 
    api_key="your-openai-api-key",
    newInferenceType="cloud"
)
```

### 3. Classify a Single Text

Classify a single piece of text across all defined topics.

```python

text_to_classify = "The new BMW X5 has impressive features."
results, probabilities = gc.classify(
    text=text_to_classify,
    isItASingleClassification=True,  # Print results to console
    constrainedOutput=True  # Use constrained output mode
)

print(f"Classification results: {results}")
print(f"Confidence scores: {probabilities}")
```

### 4. Classify a Dataset

Classify text data from a CSV file and evaluate performance.

```python

# Classify data from 'data.csv' with evaluation enabled
gc.classify_table(
    dataset="data",  # Filename without .csv extension
    withEvaluation=True,  # Compare against ground truth
    constrainedOutput=True,  # Use constrained output mode
    BATCH_SIZE=10  # Process 10 rows per batch
)
```

This will generate `data_(result).csv` containing classification results and performance metrics.

### 5. Evaluate Prompt Performance

Assess the performance of a specific topic's prompt on a dataset.

```python

# Evaluate prompt accuracy for topic 'A' on dataset 'mydata.csv'
gc.check_prompt_performance_for_topic(
    topicId="A", 
    dataset="mydata", 
    constrainedOutput=True
)
```

### 6. Improve Prompts Iteratively

Enhance the classification prompt for a specific topic using LLM feedback.

```python

# Iteratively improve prompt for topic 'A' using dataset 'mydata.csv'
gc.improve_prompt(
    topicId="A", 
    dataset="mydata", 
    constrainedOutput=True, 
    num_iterations=10
)
```

This function will refine the prompt over multiple iterations, seeking to improve classification accuracy.

### 7. Managing Topics and Categories

Functions to manage topics and categories:

```python

# Update a topic's prompt
gc.setPrompt(topicId="A", newPrompt="New improved prompt for classification.")

# Remove a specific category from a topic
gc.remove_category(topicId="A", categoryId="a")

# Remove a specific topic
gc.remove_topic("A")

# Clear all topics
gc.removeAllTopics()
```

### 8. Saving and Loading Topics

Persist and retrieve your topic configurations:

```python

# Save topics to a JSON file
gc.save_topics("my_classification_topics")

# Load topics from a JSON file
gc.load_topics("my_classification_topics")
```

## Advanced Features

### Conditional Classification

You can create dependencies between topics using conditions:

```python
# Add a topic with a condition
condition_topic = gc.add_topic(
    topic_name="Car Features",
    categories=["Sport", "Luxury", "Economy"],
    condition="A==a"  # Only classify if topic A resulted in category a
)
```

### Batch Processing

For large datasets, the classifier processes data in batches and automatically manages GPU memory:

```python
gc.classify_table(
    dataset="large_dataset",
    withEvaluation=True,
    BATCH_SIZE=5  # Smaller batches for larger models
)
```

### GPU Memory Management

The system includes built-in memory management for GPU-based models:

```python
# These functions handle model loading and unloading between batches
# to prevent GPU memory issues
# They are automatically used in classify_table but can be called manually
model = gc.load_model()
# ... do some processing ...
gc.unload_model(model)
```

### Interactive Interface

For a more user-friendly experience, you can use the interactive Jupyter interface:

```python

# Launch the widget-based interface in a Jupyter notebook
gc.openInterface()
```

## API Reference

### Main Classification Functions

#### `classify(text: str, isItASingleClassification: bool = True, constrainedOutput: bool = True, withEvaluation: bool = False, groundTruthRow: list = None) -> tuple`

Classifies a piece of text across all defined topics.

- **Returns**: Tuple of (predictions_list, probabilities_list)

#### `classify_table(dataset: str, withEvaluation: bool = False, constrainedOutput: bool = True, BATCH_SIZE: int = 10)`

Classifies each row in a CSV dataset with optional batch processing.

### Model Management

#### `setModel(newModel: str, newModelType: str, api_key: str = "", newInferenceType: str = "transformers")`

Sets the main classification model.

#### `setPromptModel(newPromptModel: str, newPromptModelType: str, api_key: str = "", newInferenceType: str = "guidance")`

Sets the model used for prompt improvement.

#### `load_model() -> object`

Loads a fresh instance of the model to GPU.

#### `unload_model(model_to_unload: object) -> None`

Thoroughly unloads a model from GPU memory.

### Topic Management

#### `add_topic(topic_name: str, categories: list = [], condition: str = "", prompt: str = default_prompt) -> dict`

Adds a new classification topic.

#### `remove_topic(topic_id_str: str)`

Removes a topic by its ID.

#### `add_category(topicId: str, categoryName: str, Condition: str = "")`

Adds a category to a specified topic.

#### `remove_category(topicId: str, categoryId: str)`

Removes a category from a specified topic.

#### `setPrompt(topicId: str, newPrompt: str)`

Updates the prompt for a specified topic.

#### `removeAllTopics()`

Removes all defined topics and resets related counters.

### Persistence

#### `save_topics(filename: str)`

Saves all topics to a JSON file.

#### `load_topics(filename: str)`

Loads topics from a JSON file.

### Prompt Improvement

#### `check_prompt_performance_for_topic(topicId: str, dataset: str, constrainedOutput: bool = True, groundTruthCol: int = None)`

Evaluates the performance of a specific topic's prompt.

#### `improve_prompt(topicId: str, dataset: str, constrainedOutput: bool = True, groundTruthCol: int = None, num_iterations: int = 10)`

Iteratively improves a prompt using LLM feedback.

#### `getLLMImprovedPromptWithFeedback(old_prompt: str, old_accuracy: float, topic_info: dict) -> str`

Gets an improved prompt suggestion from the LLM.

### Interface

#### `openInterface()`

Opens an interactive widget-based interface in Jupyter.

## Examples

### Example 1: Medical Record Classification

```python

# Set up model
gc.setModel("meta-llama/Llama-2-7b-chat-hf", "Transformers")

# Define medical topics
diagnosis = gc.add_topic(
    topic_name="Diagnosis",
    categories=["Positive", "Negative", "Inconclusive"],
    prompt="Classify the medical report diagnosis as [CATEGORIES]. Report: '[TEXT]'. The diagnosis is:"
)

# Classify a medical report
medical_text = "Patient shows no signs of infection. All tests negative."
results, _ = gc.classify(medical_text)
print(f"Diagnosis classification: {results[0]}")  # Expected: "Negative"
```

### Example 2: Multi-stage Document Processing

```python

# First level classification
doc_type = gc.add_topic(
    topic_name="Document Type",
    categories=["Invoice", "Contract", "Report"]
)

# Second level (dependent on first)
invoice_status = gc.add_topic(
    topic_name="Invoice Status",
    categories=["Paid", "Pending", "Overdue"],
    condition=f"{doc_type['id']}==a"  # Only if doc is Invoice
)

contract_type = gc.add_topic(
    topic_name="Contract Type",
    categories=["Employment", "Service", "NDA"],
    condition=f"{doc_type['id']}==b"  # Only if doc is Contract
)

# Process a batch of documents
gc.classify_table("documents", withEvaluation=True)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements, bug fixes, or new features.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

# Disclaimer

This tool is provided "as is" without any warranty. When using API-based models (OpenAI, DeepInfra), ensure you comply with their respective terms of service and usage policies.

# Contact

For questions or support, please open an issue on the GitHub repository.
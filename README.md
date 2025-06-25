## 📊 Data

The primary dataset used in this project is the **Chinese Medical Dialogue Data**, an open-source Chinese medical question-answering dataset available at the following link:

🔗 [https://github.com/Toyhom/Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data)

For data preprocessing, we extract the `title` field from each entry in the dataset and use it as input to a prompt system specifically designed for medical knowledge graphs. Using the **LLaMA-70B** language model, we generate corresponding **Cypher query statements**. These queries are then cleaned and structurally optimized.

To ensure the quality and utility of the dataset, we query the medical knowledge graph using the generated Cypher statements. Only queries that return valid results are considered effective and retained as part of our dataset.

The processed training and test sets are stored in the `primary_data` folder.

In addition, four auxiliary datasets used for plotting the loss trend analysis are included in the `auxiliary_data` folder. These datasets support comparison across different configurations or subtasks during training.



## 🧠 Model

The core implementation of the model is located in the `model` directory.

Specifically, this project modifies and extends the **Prompt-Tuning** logic by replacing the corresponding files under:

peft/tuner/prompt_tuning/

You should overwrite the original files with the customized ones provided in this project.

Model training and inference are built upon the codebase from the `llama-factory` directory. The `llama-factory` code is responsible for orchestrating model loading, prompt handling, and training integration, while the `model` folder provides the custom tuning logic tailored for our Cypher generation task.


## 💻 Environment

All standard dependencies for this project are listed in the `requirements.txt` file.

In addition, this project relies on a modified version of **[llama-factory](https://github.com/hiyouga/llama-factory)**. To set it up properly:

1. Clone the `llama-factory` repository locally.
2. Install it in **editable mode** using version `0.9.2`, as recommended in their documentation.
3. Follow the official setup instructions provided by `llama-factory` to complete the environment configuration.

Model training and inference should be conducted using `llama-factory`'s framework, integrated with the custom prompt-tuning modifications from this project.

## 📊 Data

The primary dataset used in this project is the open-source **Chinese Medical Dialogue Data**. It is publicly available at the following link:

🔗 [https://github.com/Toyhom/Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data)

For preprocessing, we extract the `title` field from each sample as the input prompt. This prompt is then fed into the **LLaMA-70B** model, combined with a specialized prompt system tailored for a medical knowledge graph, to generate corresponding **Cypher queries**.

The generated Cypher queries undergo cleaning and optimization. After that, we query the medical knowledge graph using each Cypher statement. Only those queries that successfully return meaningful results are retained as valid entries in our final dataset.

The resulting training and testing sets are available in the `primary_data` directory.


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

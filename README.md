## 📊 Data

The primary dataset used in this project is the open-source **Chinese Medical Dialogue Data**. It is publicly available at the following link:

🔗 [https://github.com/Toyhom/Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data)

For preprocessing, we extract the `title` field from each sample as the input prompt. This prompt is then fed into the **LLaMA-70B** model, combined with a specialized prompt system tailored for a medical knowledge graph, to generate corresponding **Cypher queries**.

The generated Cypher queries undergo cleaning and optimization. After that, we query the medical knowledge graph using each Cypher statement. Only those queries that successfully return meaningful results are retained as valid entries in our final dataset.

The resulting training and testing sets are available in the `primary_data` directory.


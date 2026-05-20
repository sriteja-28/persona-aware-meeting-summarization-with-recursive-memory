# Persona-Aware Meeting Summarization with Recursive Memory

An NLP pipeline designed for role-conditioned, abstractive meeting summarization on the AMI Meeting Corpus. This project implements and evaluates architectures that dynamically tailor summaries based on a reader's organizational persona (Manager, Engineer, or Stakeholder).

---

## 📂 Repository Structure

* **`NLP_Final_Meeting_Summarization.ipynb`**: The entire end-to-end project codebase contained within a single Jupyter Notebook. It is divided into clear logical sections:
  * **Data Preprocessing**: Loads the AMI Meeting Corpus and utilizes a custom parser to segment conversational transcripts into clean, speaker-turn-preserving structural chunks.
  * **Model Configurations**: Sets up pipelines for pre-trained Hugging Face language models (**Flan-T5-base**, **DistilBART-CNN**, and **PEGASUS-XSum**) with built-in fallback mechanisms for seamless CPU execution.
  * **Summarization Core Engine**: Implements text chunk aggregation and executes three pipeline architectures:
    1. *Baseline*: Summarizes text windows entirely independently.
    2. *Method 1 (Recursive Memory)*: Passes a rolling, context-preserving text memory layer across sequential chunks.
    3. *Method 2 (Persona-Conditioned Two-Step Extraction)*: Extracts targeted, persona-specific context directly from uncompressed transcripts before synthesis.
  * **Evaluation Framework**: Computes automatic text metrics (**ROUGE-1/2/L** and **BERTScore F1** contextual embeddings) alongside keyword hit rates to evaluate summary accuracy.

---

## 🚀 How to Run the Project

Follow these steps to set up the environment and run the notebook on your local machine.

### 1. Prerequisites
Ensure you have **Python 3.8+** installed along with Jupyter Notebook or JupyterLab. A CUDA-compatible GPU is recommended for faster generation, though the project contains fallbacks to run seamlessly on a CPU.

### 2. Clone the Repository
```bash
git clone [https://github.com/sriteja-28/persona-aware-meeting-summarization-with-recursive-memory.git](https://github.com/sriteja-28/persona-aware-meeting-summarization-with-recursive-memory.git)
cd your-repo-name
```


### 3. Create and Activate a Virtual Environment
It is highly recommended to isolate your project dependencies:

# On macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

# On Windows
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies
Install the required NLP, deep learning, and notebook libraries:

```bash
pip install torch transformers datasets rouge-score bert-score jupyter notebook
```
(Note: If you prefer to use a requirements file instead, run pip install -r requirements.txt)


### 5. Launch and Run the Notebook
Open the Jupyter interface:

```bash
jupyter notebook
```

Click on NLP_Final_Meeting_Summarization.ipynb and select Kernel -> Restart & Run All from the top menu to execute the data preprocessing, model generation pipelines, and evaluation metrics step-by-step.

### 📊 Evaluation Results Summary
Method 2 (Persona-Conditioned Two-Step Extraction) achieved a human validation score of 4.41/5 for Persona Relevance, drastically outperforming the Baseline (2.03/5) and Recursive Memory (2.80/5).

Semantic embeddings measured via BERTScore proved to be a vastly more reliable quality indicator for role-conditioned abstractive tasks compared to token-matching metrics like ROUGE, which favor raw text lengths over quality constraints.
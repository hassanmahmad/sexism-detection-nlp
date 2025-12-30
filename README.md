# Sexism Detection in Social Media Text

A comprehensive NLP project implementing multi-class sexism classification using deep learning and large language models. This project explores BiLSTM networks, Transformer models (RoBERTa), and LLM-based zero-shot/few-shot prompting approaches for detecting and categorizing sexist content in tweets.

## Project Overview

This project addresses the critical challenge of automated sexism detection on social media platforms. It demonstrates proficiency in multiple NLP paradigms:

- **Traditional Deep Learning**: BiLSTM architectures with GloVe embeddings
- **Transfer Learning**: Fine-tuned Twitter-RoBERTa transformer model
- **LLM Prompting**: Zero-shot and few-shot classification with quantized LLMs

## Key Results

| Approach | F1-Score (Macro) | Accuracy |
|----------|------------------|----------|
| BiLSTM (Baseline) | 0.35 | 73.6% |
| BiLSTM (Stacked) | 0.35 | 73.6% |
| Twitter-RoBERTa | **0.46** | **77.9%** |

## Technical Implementation

### Assignment 1: Neural Networks & Transformers

**Dataset**: EXIST 2023 Task 2 (English tweets with multi-annotator labels)
- 2,867 training samples | 280 test | 150 validation
- 4-class classification: Non-sexist, Direct, Judgemental, Reported

**Data Pipeline**:
- Text cleaning (emoji removal, URL/mention handling, special characters)
- Linguistic preprocessing (lemmatization, stemming via NLTK)
- GloVe Twitter embeddings (50D) with OOV handling

**Models Implemented**:

1. **Bidirectional LSTM**
   - Single and stacked architectures
   - Spatial dropout + recurrent dropout for regularization
   - Trained across 3 random seeds for reproducibility
   - Ensemble methods explored (soft voting)

2. **Twitter-RoBERTa (Fine-tuned)**
   - Base model: `cardiffnlp/twitter-roberta-base-hate`
   - Modified classification head for 4-class output
   - Achieved 31% relative improvement in F1-score over BiLSTM

### Assignment 2: LLM-based Classification (In-Progress)

**Approach**: Zero-shot and few-shot prompting with quantized LLMs

**Models Evaluated**:
- Mistral 7B (v2/v3)
- Llama 3.1 (8B)
- Phi3-mini (3.8B)
- TinyLlama (1.1B)
- DeepSeek-R1 (7B)
- Qwen3 (1.7B)

**Dataset**: EDOS dataset (5-class: not-sexist, threats, derogation, animosity, prejudiced)

**Key Features**:
- 4-bit quantization for efficient inference
- Structured prompt engineering with clear task definitions
- Few-shot demonstrations with balanced class sampling
- Comprehensive error analysis and fail-ratio tracking

## Technical Skills Demonstrated

- **Deep Learning**: TensorFlow/Keras, PyTorch, model architecture design
- **NLP**: Text preprocessing, embeddings, sequence modeling, attention mechanisms
- **Transformers**: Hugging Face ecosystem, fine-tuning, tokenization
- **LLMs**: Prompt engineering, quantization (BitsAndBytes), inference optimization
- **ML Engineering**: Cross-validation, hyperparameter tuning, ensemble methods
- **Data Analysis**: Pandas, NumPy, Matplotlib, Seaborn, confusion matrices
- **Evaluation**: Macro F1, precision, recall, class-imbalanced metrics

## Project Structure

```
sexism-detection-nlp/
├── sexism-detection-BiLSTM-Transformer-A1.ipynb  # Neural network approaches
├── [NLP 2025-2026]A2.ipynb                       # LLM prompting approaches
└── README.md
```

## Key Findings & Insights

1. **Class Imbalance Impact**: Heavily imbalanced datasets bias models toward majority class; addressed through weighted loss and oversampling strategies

2. **Transformer Superiority**: Pre-trained transformers significantly outperform traditional RNNs on nuanced social media text classification

3. **Context Sensitivity**: Subtle sarcasm and contextual cues remain challenging for all approaches

4. **Few-Shot Improvement**: LLMs show measurable gains with few-shot demonstrations over zero-shot prompting

## Technologies Used

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

- **Frameworks**: TensorFlow, Keras, PyTorch, Hugging Face Transformers
- **NLP Tools**: NLTK, Gensim, Tokenizers
- **Data Science**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **LLM Optimization**: BitsAndBytes (4-bit quantization)

## Future Improvements

- Implement cross-lingual sexism detection (Spanish data available)
- Explore attention visualization for model interpretability
- Test retrieval-augmented generation (RAG) approaches
- Investigate chain-of-thought prompting for improved LLM reasoning

## Academic Context

This project was completed as part of the Natural Language Processing course (2025-2026), demonstrating practical application of cutting-edge NLP techniques to a real-world social impact problem.

## Contact

Feel free to reach out for questions or collaboration opportunities.

---

*This project showcases end-to-end NLP pipeline development from data preprocessing to model deployment, reflecting skills directly applicable to AI/ML engineering roles.*

# Amazon Product Review Analysis
This repository contains an end-to-end **Amazon Product Review Analysis** project, demonstrating how to build and deploy **LLM-powered sentiment analysis** using both **BERT** (from Hugging Face Transformers) and a **custom LSTM**. The project also includes steps to design an **ad ranking system** leveraging sentiment scores, and guidance for **AWS cloud deployment** using **Lambda** or **SageMaker**.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset & Preprocessing](#dataset--preprocessing)  
3. [Modeling Approaches](#modeling-approaches)  
   - [BERT (Transformers)](#bert-transformers)  
   - [Custom LSTM](#custom-lstm)  
   - [Three-Class Classification vs. Regression](#three-class-classification-vs-regression)  
4. [Ad Ranking System](#ad-ranking-system)  
5. [Deployment](#deployment)  
   - [AWS SageMaker](#aws-sagemaker)  
   - [AWS Lambda](#aws-lambda)  
6. [Visualizations](#visualizations)  
7. [Repository Structure](#repository-structure)  
8. [How to Run](#how-to-run)  
9. [Future Work](#future-work)  
10. [License](#license)

---

## Project Overview
**Goal**: Develop a comprehensive pipeline that performs:
- **Sentiment Analysis** of Amazon product reviews (negative, neutral, positive), or direct **star rating regression** (1–5).
- **Ad Ranking System** using sentiment scores to improve user engagement and click-through rates (CTR).
- **Cloud Deployment** for scalable, low-latency inference on AWS (Lambda or SageMaker).

This project was inspired by real-world scenarios where companies like Amazon, Apple, Netflix, etc., use **machine learning** to optimize user experience and product recommendations.

---

## Dataset & Preprocessing
1. **Dataset**: The project uses a `.jsonl.gz` file of Amazon product reviews (e.g., `Movies_and_TV.jsonl.gz`), which contains fields like:
   - `rating` (1–5 star rating)
   - `text` (the review content)
   - Other optional metadata (e.g., `title`, `timestamp`, `asin`, etc.)

2. **Preprocessing**:
   - **Parsing**: We read the GZ file line-by-line using Python’s `gzip` and `json`.
   - **Cleaning**: Convert text to lowercase, remove punctuation, etc. (especially for LSTM tokenization).
   - **Labeling**:
     - **3-Class Classification**: 
       - 0 = Negative (rating ≤ 2),  
       - 1 = Neutral (rating = 3),  
       - 2 = Positive (rating ≥ 4).
     - **Regression**: Predict the exact star rating from 1 to 5.

3. **Train-Test Split**: We split the dataset into **80% training** and **20% test** (or other ratios) using `sklearn.model_selection.train_test_split`, with stratification by sentiment if doing classification.

---

## Modeling Approaches

### BERT (Transformers)
- **Library**: Hugging Face Transformers (`transformers` package).  
- **Model**: `BertForSequenceClassification` for 3-class classification or regression (`num_labels=1` with `problem_type="regression"`).  
- **Tokenization**: `BertTokenizer` (`bert-base-uncased` by default).  
- **Training**: Fine-tuned with `AdamW` optimizer, typical learning rate of `2e-5` for a few epochs.  
- **Saving**: `model.save_pretrained()` + `tokenizer.save_pretrained()`.

### Custom LSTM
- **Architecture**: An embedding layer -> LSTM -> linear layer for output.  
- **Preprocessing**: Build a vocabulary (`word_to_idx`), convert each review to sequences of word indices, handle padding.  
- **Training**: Use `CrossEntropyLoss` for classification (3-class) or `MSELoss` for rating regression.  
- **Saving**: Use `torch.save(model.state_dict(), "lstm.pth")`.

### Three-Class Classification vs. Regression
1. **3-Class**:
   - Negative, Neutral, Positive
   - Helpful if you want discrete categories for your ad ranking logic.
2. **Regression**:
   - Predict star rating from 1.0 to 5.0 (continuous).
   - Offers finer granularity; can rank products by exact predicted rating.

---

## Ad Ranking System
1. **Predict Sentiment or Rating**: For each product review, use the trained model to generate a sentiment score or predicted star rating.
2. **Aggregate Scores**:  
   - For each product (identified by `asin`), compute the **average** predicted sentiment/ rating across all its reviews.
3. **Ranking**:  
   - Sort products in descending order of average sentiment/rating.
   - (Optional) Combine with other signals like historical CTR or user context.
4. **CTR Improvement**:  
   - Hypothetically, we measure CTR via an A/B test comparing the new ranking method to a baseline. In many real-world scenarios, a 15% improvement can be achieved with better product targeting.

---

## Deployment

### AWS SageMaker
1. **Save Model**:
   ```python
   model.save_pretrained('bert_3class_model')
   tokenizer.save_pretrained('bert_3class_tokenizer')
2. Upload the artifacts to S3 using awscli or boto3.
3. Create SageMaker Endpoint:
   - Define a PyTorchModel or HuggingFaceModel referencing your S3 artifacts.
   - Provide an inference.py script that loads the model and handles prediction.
   - Deploy to an instance type like ml.m5.large.
## AWS Lambda
1. Package a smaller model (e.g., DistilBERT or a minimal LSTM) into a ZIP under 250MB (unzipped).
2. Upload to AWS Lambda, potentially using a Lambda Layer for large dependencies (like torch).
3. API Gateway can trigger the function for on-demand sentiment predictions.

# SMS Spam Detection with BERT and PyTorch

A deep learning project for classifying SMS messages as spam or legitimate messages using a fine-tuned BERT model and PyTorch.

## Project Overview

This project uses the pretrained `bert-base-uncased` model to classify SMS messages into two categories:

- **Ham (0):** Legitimate messages.
- **Spam (1):** Unwanted or fraudulent messages.

The model is fine-tuned on the SMS Spam Collection dataset to learn patterns that help distinguish spam messages from legitimate ones.

## Dataset

The project uses the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

The dataset contains **5,572 SMS messages**, including:

- 4,825 legitimate messages (Ham)
- 747 spam messages

The dataset is cleaned and prepared before training. Missing or empty messages are removed, and the target labels are encoded as binary values.

The data is split into training and testing sets using an 80/20 split with stratification to preserve the class distribution.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- BERT (`bert-base-uncased`)
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Model Training

The pretrained BERT model is fine-tuned for binary text classification.

Key training settings:

| Parameter | Value |
|---|---|
| Pretrained model | `bert-base-uncased` |
| Maximum sequence length | 128 |
| Batch size | 32 |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Loss function | Cross-Entropy Loss |
| Learning rate scheduler | Linear schedule with warmup |
| Random seed | 42 |

The implementation supports GPU acceleration when CUDA is available and falls back to CPU otherwise.

## Evaluation

The model is evaluated on the held-out test set using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

These metrics provide a more complete view of the model's performance, particularly because the dataset contains significantly more legitimate messages than spam messages.

### Test Results

The model achieved the following results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 0.9937 |
| Precision | 0.9863 |
| Recall | 0.9664 |
| F1-score | 0.9763 |

The model achieved approximately **99.37% accuracy** and a **97.63% F1-score** on the test set.

## Project Workflow

1. Load and inspect the SMS Spam Collection dataset.
2. Clean the messages and encode the target labels.
3. Split the data into training and testing sets.
4. Tokenize the messages using the BERT tokenizer.
5. Fine-tune the pretrained BERT model using PyTorch.
6. Evaluate the trained model on the test set.
7. Visualize the training loss and confusion matrix.

## How to Run

### 1. Install Dependencies

```bash
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn tqdm kagglehub
```

### 2. Download the Dataset

Download the SMS Spam Collection dataset from Kaggle and update the dataset path in the notebook if necessary.

### 3. Run the Notebook

Open the Jupyter Notebook in Jupyter or Google Colab and execute the cells in order.

The notebook automatically uses a CUDA-enabled GPU when available.

## Limitations

- The dataset is relatively small and may not represent all modern spam messages.
- The dataset primarily contains English SMS messages.
- Model performance may vary when applied to messages from different sources or distributions.

## Future Improvements

- Test the model on additional SMS datasets.
- Analyze false positives and false negatives.
- Save the fine-tuned model and tokenizer for reuse.
- Build a simple application for predicting whether a message is spam.

## License

This project is intended for educational and portfolio purposes. Refer to the original dataset source for its licensing and usage terms.

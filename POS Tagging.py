from typing import List
import numpy as np
import torch
import evaluate
from sklearn.model_selection import train_test_split
import nltk
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset

# 1. Load the Penn Treebank dataset
nltk.download('treebank')

# Load the Penn Treebank dataset
tagged_sentences = nltk.corpus.treebank.tagged_sents()
print("Number of sentences in the corpus:", len(tagged_sentences))
print(tagged_sentences[0])

# Save sentences and tags
sentences, sentence_tags = [], []
for tagged_sentence in tagged_sentences:
    sentence, tags = zip(*tagged_sentence)
    sentences.append([word.lower() for word in sentence])
    sentence_tags.append([tag for tag in tags])

# 2. Preprocess the dataset
# Split the dataset into training, validation, and test sets
train_sentences, test_sentences, train_tags, test_tags = train_test_split(sentences, sentence_tags, test_size=0.3, random_state=42)
valid_sentences, test_sentences, valid_tags, test_tags = train_test_split(test_sentences, test_tags, test_size=0.5, random_state=42)

# Build a vocabulary
unique_tags = set(tag for tags in sentence_tags for tag in tags)
label2id = {tag: idx for idx, tag in enumerate(unique_tags)}
id2label = {idx: tag for tag, idx in label2id.items()}

# Tokenization
model_name = "QCRI/bert-base-multilingual-cased-pos-english"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
MAX_LEN = 256

class PosTaggingDataset(Dataset):
    def __init__(self, sentences: List[List[str]], tags: List[List[str]], tokenizer, label2id, max_len=MAX_LEN):
        super().__init__()
        self.sentences = sentences
        self.tags = tags
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        input_tokens = self.sentences[idx]
        label_tokens = self.tags[idx]

        encoding = self.tokenizer(input_tokens,
                                  is_split_into_words=True,
                                  padding="max_length",
                                  truncation=True,
                                  max_length=self.max_len,
                                  return_tensors="pt")

        labels = [self.label2id[tag] for tag in label_tokens]
        labels = self.pad_and_truncate(labels, pad_id=-100)  # -100 để tránh tính toán gradient trên padding

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    def pad_and_truncate(self, inputs: List[int], pad_id: int):
        if len(inputs) < self.max_len:
            padded_inputs = inputs + [pad_id] * (self.max_len - len(inputs))
        else:
            padded_inputs = inputs[:self.max_len]
        return padded_inputs

# Dataset loader
train_dataset = PosTaggingDataset(train_sentences, train_tags, tokenizer, label2id)
val_dataset = PosTaggingDataset(valid_sentences, valid_tags, tokenizer, label2id)
test_dataset = PosTaggingDataset(test_sentences, test_tags, tokenizer, label2id)

# 3. Modeling
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)

# 4. Metrics
accuracy = evaluate.load("accuracy")
ignore_label = -100  # Tránh tính toán loss trên padding

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    mask = labels != ignore_label  # Bỏ qua các vị trí padding
    filtered_predictions = predictions[mask]
    filtered_labels = labels[mask]

    return accuracy.compute(predictions=filtered_predictions, references=filtered_labels)

# 5. Trainer
training_args = TrainingArguments(
    output_dir="out_dir",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 6. Train the model
trainer.train()

# 7. Inference
test_sentence = "We are exploring the topic of deep learning"
tokens = tokenizer(test_sentence.split(), is_split_into_words=True, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN)

tokens = {key: val.to("cuda") for key, val in tokens.items()}  # Move to GPU

model.eval()
with torch.no_grad():
    outputs = model(**tokens)

logits = outputs.logits
predictions = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

# Decode predictions
pred_tags = [id2label[pred] for pred in predictions if pred in id2label]
print(pred_tags)

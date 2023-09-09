#!/usr/bin/env python
# coding: utf-8

# Named Entity Recognition Experiment

import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForTokenClassification
# import modin.pandas as pd
import time

# import os
# os.environ['MODIN_ENGINE'] = 'dask'

PRETRAINED = "prajjwal1/bert-tiny"


def load_dataset():
    """
    Load text dataset with entity tags
    """
    dataset = pd.read_csv("ner.csv")
    dataset["Sentence #"] = dataset["Sentence #"].fillna(method='ffill')
    sentences, targets = [], []
    for sent_i, x in dataset.groupby("Sentence #"):
        words = x["Word"].tolist()
        tags = x["Tag"].tolist()
        sentences.append(words)
        targets.append(tags)

    # Number of sentences in dataset
    len(sentences)
    return sentences, targets


def text_encoding(sentences):
    """
    Convert each word into subwords and their respective subword ids such that Bert can work with the words
    """
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED)
    sentences_encoded = tokenizer(
        sentences, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=150,
    add_special_tokens=False
    )
    return sentences_encoded, tokenizer


def target_encoding(targets):
    """
    Convert the NER tags into tensors such that Bert can work with them
    """
    # mapping from ner tag to number
    tag2idx = {tag: i for i, tag in enumerate(set(t for ts in targets for t in ts))}
    print(tag2idx)

    # Pad the target tensors because sentences have different length
    max_len = sentences_encoded["input_ids"].shape[1]
    targets_encoded = torch.empty((0, max_len), dtype=torch.long)

    for sent_idx, target in enumerate(targets):
        enc = torch.full(size=(max_len,), fill_value=tag2idx['O'], dtype=torch.long)
        # repeat ner tag for each subword
        for word_idx, tag in enumerate(target):
            span = sentences_encoded.word_to_tokens(sent_idx, word_idx)
            # ignore words that tokenizer did not understand e.g. special characters
            if span is not None:
                start, end = span
                enc[start:end] = tag2idx[tag]
        targets_encoded = torch.vstack((targets_encoded, enc))

    print(targets_encoded.shape)
    return targets_encoded, tag2idx


sentences, targets = load_dataset()
sentences_encoded, tokenizer = text_encoding(sentences)
targets_encoded, tag2idx = target_encoding(targets)
# Show the first sample and its target tensor
print(tokenizer.decode(sentences_encoded["input_ids"][0]))
print(targets_encoded[:1])

# ### Train/Test split
# Split dataset into training and test set

train_size = int(0.8 * len(sentences_encoded.input_ids))
test_size = int(0.2 * len(sentences_encoded.input_ids))
train_sentences = sentences_encoded[:train_size]
train_targets = targets_encoded[:train_size]
test_sentences = sentences_encoded[train_size:train_size + test_size]
test_targets = targets_encoded[train_size:train_size + test_size]
print(f"Train sentences: {len(train_targets)}", f"Test sentences: {len(test_targets)}")


class NERDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __getitem__(self, index):
        ids = torch.tensor(self.sentences[index].ids)
        mask = torch.tensor(self.sentences[index].attention_mask)
        labels = self.labels[index].clone()

        return {
            'ids': ids,
            'mask': mask,
            'tags': labels
        }

    def __len__(self):
        return len(self.labels)


training_set = NERDataset(train_sentences, train_targets)
testing_set = NERDataset(test_sentences, test_targets)

training_loader = DataLoader(training_set, batch_size=16, shuffle=True)
testing_loader = DataLoader(testing_set, batch_size=16, shuffle=True)

# ## Train model

# Load a pretrained Bert model to fine-tune for multi-class classification of NER tags


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = BertForTokenClassification.from_pretrained(PRETRAINED, num_labels=len(tag2idx),
                                                   output_attentions=True,
                                                   output_hidden_states=True)
model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-05)

# training loop
for epoch in range(1, 3):
    epoch_train_losses = np.zeros(shape=len(training_loader))
    model.train()
    for i, data in enumerate(training_loader, 0):
        data = {k: v.to(device) for k, v in data.items()}
        output = model(data["ids"], attention_mask=data["mask"], labels=data["tags"])
        loss = output[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_train_losses[i] = loss.detach().cpu().item()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} - train_loss: {epoch_train_losses.mean()}")

################# Test model ############################
# Compute classification metric of Bert model on test set

model.eval()
all_preds, all_trues = [], []

for data in testing_loader:
    data = {k: v.to(device) for k, v in data.items()}
    with torch.no_grad():
        output = model(data["ids"], attention_mask=data["mask"], labels=data["tags"])
    loss = output[0]
    logits = output[1].detach().cpu()
    mask = data["mask"].cpu()

    label_ids = data["tags"].cpu()
    pred_ids = torch.argmax(logits, dim=-1)

    for i in range(pred_ids.shape[0]):
        # remove pad predictions
        pred_ids_non_pad = pred_ids[i, mask[i]]
        label_ids_non_pad = label_ids[i, mask[i]]
        all_preds.append(pred_ids_non_pad)
        all_trues.append(label_ids_non_pad)

all_preds = torch.cat(all_preds)
all_trues = torch.cat(all_trues)
accuracy = accuracy_score(all_trues, all_preds)
print("Test Accuracy:", round(accuracy, 3))
import torch
import random
import numpy as np


def generate_random_dna(length=32):
    dna = ""
    for i in range(length):
        dna += random.choice(["A", "T", "G", "C"])
    return [dna]


def onehot_encoder(sequences):
    max_len = max([len(s) for s in sequences])
    dictionary = {
        "A": [1, 0, 0, 0],
        "T": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "C": [0, 0, 0, 1],
    }
    if not isinstance(sequences, (list, tuple, np.ndarray)):
        sequences = list(sequences)
    for i in range(len(sequences)):
        sequences[i] = sequences[i].upper()[:max_len]
    shape = [len(sequences), max_len, len(dictionary)]
    onehot = np.zeros(shape, dtype=np.float32)
    for i, s in enumerate(sequences):
        for j, el in enumerate(s):
            onehot[i, j] = dictionary[el]
    if len(sequences) == 1:
        onehot = np.squeeze(onehot, axis=0)
    return torch.tensor(onehot).unsqueeze(0)


def dna_decoder(dna):
    dictionary = {
        0: "A",
        1: "T",
        2: "G",
        3: "C",
    }
    return "".join([dictionary[torch.argmax(j).item()] for j in dna])

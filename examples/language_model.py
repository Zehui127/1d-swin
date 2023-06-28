from transformers import GPT2Tokenizer
import torch
import random
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
WORDS_BANK = [
    "you",
    "I",
    "me",
    "he",
    "she",
    "him",
    "they",
    "them",
    "love",
    "take",
    "seek",
    "buy",
    "dream",
    "eat",
    "play",
    "beat",
    "drink",
    "cat",
    "dog",
    "tiger",
    "chicken",
    "beef",
    "milk",
    "juice",
    "water",
    "police",
    "student",
    "teacher",
    "airplane",
    "train",
    "car",
    ",",
    ".",
    ":",
    "but",
    "and",
]
WORDS_COUNT = len(WORDS_BANK)


def tokenize(text, max_seq_length=32):
    # Tokenize the input text
    tokenized_input = tokenizer.encode(text, add_special_tokens=False)

    # Pad or truncate the input sequence to the desired length
    padded_input = tokenized_input[:max_seq_length] + [
        tokenizer.pad_token_id
    ] * (max_seq_length - len(tokenized_input))

    # Convert the input sequence to tensors
    np_padded_input = np.array(padded_input, dtype=float)
    input_ids = torch.tensor(np_padded_input, dtype=torch.float32)

    return input_ids


def random_text_generator(batch_size, seq_length=WORDS_COUNT, tokenized=False):
    def generate_sentence(seq_length):
        return " ".join(
            WORDS_BANK[random.randint(0, WORDS_COUNT - 1)]
            for _ in range(seq_length)
        )

    # return a list of sentences
    # choose seq_length words from word bank
    shape = [batch_size, seq_length, 1]
    res = np.zeros(shape, dtype=np.float32)
    if tokenized:
        for i in range(batch_size):
            res[i] = tokenize(
                generate_sentence(seq_length), seq_length
            ).unsqueeze(1)
        return torch.tensor(res)
    else:
        return [generate_sentence(seq_length) for _ in range(batch_size)]


def text_decoder(input_ids):
    text = tokenizer.decode(input_ids, skip_special_tokens=True)
    return text

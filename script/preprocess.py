def filter_empty_rows(example):
    return example["text"].strip() != ""


def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True, max_length=512, truncation=True, padding="max_length")


def preprocess(tokenizer, dataset):
    dataset = dataset.filter(filter_empty_rows)
    dataset = dataset.map(lambda examples: tokenize_function(tokenizer, examples), batched=True)
    return dataset
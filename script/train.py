from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import AdamW, GPT2LMHeadModel, GPT2Tokenizer
import yaml

from objective import bigphi_loss
from preprocess import preprocess
from utils import set_seed

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_objects(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2LMHeadModel.from_pretrained(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    # https://github.com/huggingface/transformers/issues/12594
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    dataset_name, subset_name = args.dataset.split(',')
    dataset = load_dataset(dataset_name, subset_name, split='train')
    return model, tokenizer, dataset, device


def train(args):
    model, tokenizer, dataset, device = load_objects(args)
    dataset = preprocess(tokenizer, dataset)
    # Move model to GPU if available
    model.to(device)
    # Set random seeds for reproducibility
    set_seed(args.seed)
    config=load_config(args.config_path)
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    vocab_size = tokenizer.vocab_size
    results = {}
    for epoch in range(args.epochs):
        for i, data in enumerate(dataset):
            # Tokenize the input text
            inputs = tokenizer(data["text"], return_tensors="pt", max_length=512, truncation=True, padding="max_length")
            # inputs = tokenizer(data["text"], return_tensors="pt", max_length=512, truncation=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to GPU
            labels = inputs["input_ids"].clone()
            # Shift the inputs and labels for CLM
            # # print input for debug
            # print(inputs["input_ids"].shape, labels.shape)
            # torch.Size([1, 512]) torch.Size([1, 512])
            inputs = inputs["input_ids"][:, :-1].contiguous()
            labels = labels[:, 1:].contiguous()
            # # print input for debug
            # print(inputs.shape, labels.shape)
            # torch.Size([1, 511]) torch.Size([1, 511])
            outputs = model(inputs, labels=labels)
            loss_ce = outputs.loss
            # Calculate mutual information loss (this is a simplified version)
            # loss_mi = mutual_information_loss(model, inputs, labels)
            loss_phi = bigphi_loss(model, config, inputs, labels)
            # Combine losses
            # loss = loss_ce + loss_mi
            loss = loss_ce + loss_phi
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            if i % 100 == 0:
                # print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataset)}], Base Loss: {loss_ce.item()}, Mesa Loss: {loss_mi.item()}, Total Loss: {loss.item()}")
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i}/{len(dataset)}], Base Loss: {loss_ce.item()}, Mesa Loss: {loss_phi.item()}, Total Loss: {loss.item()}")
                results["epoch"] = epoch
                results["step"] = i
                results["base_loss"] = loss_ce.item()
                # results["mesa_loss"] = loss_mi.item()
                results["mesa_loss"] = loss_phi.item()
                results["total_loss"] = loss.item()
            del inputs, labels, outputs, loss_ce, loss_phi, loss
            torch.cuda.empty_cache()
    return results


def train_batch(args):
    model, tokenizer, dataset, device = load_objects(args)
    # Set random seeds for reproducibility
    set_seed(args.seed)
    config = load_config(args.config_path)
    dataset = preprocess(tokenizer, dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    vocab_size = tokenizer.vocab_size
    results = {}
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("Starting training...")  # Debugging statement

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")  # Debugging statement
        for i, batch in enumerate(tqdm(dataloader)):
            # print(f"Processing batch {i+1}")  # Debugging statement
            # Tokenize the input text
            inputs = tokenizer(batch["text"], return_tensors="pt", max_length=512, truncation=True, padding="max_length")
            inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to GPU
            labels = inputs["input_ids"].clone()
            # # print input for debug
            # print(inputs["input_ids"].shape, labels.shape)
            # torch.Size([8, 512]) torch.Size([8, 512])
            # Shift the inputs and labels for CLM
            inputs = inputs["input_ids"][:, :-1].contiguous()
            labels = labels[:, 1:].contiguous()
            # # print input for debug
            # print(inputs.shape, labels.shape)
            # torch.Size([8, 511]) torch.Size([8, 511])
            outputs = model(inputs, labels=labels)
            loss_ce = outputs.loss
            loss_phi = bigphi_loss(model, config, inputs, labels)
            # Combine losses
            loss = loss_ce + loss_phi
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i}/{len(dataloader)}], Base Loss: {loss_ce.item()}, Mesa Loss: {loss_phi.item()}, Total Loss: {loss.item()}")
                results["epoch"] = epoch
                results["step"] = i
                results["base_loss"] = loss_ce.item()
                results["mesa_loss"] = loss_phi.item()
                results["total_loss"] = loss.item()
            del inputs, labels, outputs, loss_ce, loss_phi, loss
            torch.cuda.empty_cache()

    print("Training complete.")  # Debugging statement
    return results
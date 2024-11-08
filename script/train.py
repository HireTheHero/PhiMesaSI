import datetime
import os

from datasets import load_dataset

# from tqdm import tqdm
import torch
from transformers import AdamW, GPT2LMHeadModel, GPT2Tokenizer
import yaml
import wandb

from objective import bigphi_loss, conditional_entropy_loss
from preprocess import preprocess
from utils import (
    free_memory,
    get_module_logger,
    log_console_and_or_wandb,
    set_seed,
    wandb_tabularize_loss,
)


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_objects(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    # https://github.com/huggingface/transformers/issues/12594
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    dataset_name, subset_name = args.dataset.split(",")
    dataset = load_dataset(dataset_name, subset_name, split="train")
    return model, tokenizer, dataset, device


def train(args):
    logger = get_module_logger(__name__)
    model, tokenizer, dataset, device = load_objects(args)
    logger.info("Objects loaded.")
    dataset = preprocess(tokenizer, dataset)
    logger.info("Dataset preprocessed.")
    # Move model to GPU if available
    model.to(device)
    # Set random seeds for reproducibility
    set_seed(args.seed)
    config = load_config(args.config_path)
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    vocab_size = tokenizer.vocab_size
    results = []
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        for i, data in enumerate(dataset):
            # Tokenize the input text
            inputs = tokenizer(
                data["text"],
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length",
            )
            # inputs = tokenizer(data["text"], return_tensors="pt", max_length=512, truncation=True)
            inputs = {
                key: val.to(device) for key, val in inputs.items()
            }  # Move inputs to GPU
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
            # loss_ce = outputs.loss
            loss_ce = conditional_entropy_loss(outputs.logits, labels)
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
                log_console_and_or_wandb(
                    f"base_loss: {loss_ce.item()}",
                    log_wandb=args.log_wandb,
                    logger=logger,
                    epoch=epoch,
                    step=i,
                )
                log_console_and_or_wandb(
                    f"mesa_loss: {loss_phi.item()}",
                    log_wandb=args.log_wandb,
                    logger=logger,
                    epoch=epoch,
                    step=i,
                )
                log_console_and_or_wandb(
                    f"total_loss: {loss.item()}",
                    log_wandb=args.log_wandb,
                    logger=logger,
                    epoch=epoch,
                    step=i,
                )
                # print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataset)}], Base Loss: {loss_ce.item()}, Mesa Loss: {loss_mi.item()}, Total Loss: {loss.item()}")
                # print(
                #     f"Epoch [{epoch+1}/{args.epochs}], Step [{i}/{len(dataset)}], Base Loss: {loss_ce.item()}, Mesa Loss: {loss_phi.item()}, Total Loss: {loss.item()}"
                # )
                results.append(
                    {
                        "epoch": epoch,
                        "step": i,
                        "base_loss": loss_ce.item(),
                        "mesa_loss": loss_phi.item(),
                        "total_loss": loss.item(),
                    }
                )
                wandb_tabularize_loss(results)
            del inputs, labels, outputs, loss_ce, loss_phi, loss
            free_memory()

        logger.info(f"Epoch {epoch+1}/{args.epochs} completed.")
        # save model with datetime string
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f"{args.output_dir}/{dt_str}", exist_ok=True)
        dt_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        model.save_pretrained(f"{args.output_dir}/model_{dt_str}_epoch{i}.pt")
        if args.log_wandb:
            wandb.save(f"{args.output_dir}/model_{dt_str}_epoch{i}.pt")
    return results


def train_batch(args):
    logger = get_module_logger(__name__)
    model, tokenizer, dataset, device = load_objects(args)
    logger.info("Objects loaded.")
    # Set random seeds for reproducibility
    set_seed(args.seed)
    config = load_config(args.config_path)
    dataset = preprocess(tokenizer, dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )
    logger.info("Dataset preprocessed.")
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    results = []
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info("Starting training...")

    for epoch in range(args.epochs):
        for i, batch in enumerate(dataloader):
            # Tokenize the input text
            inputs = tokenizer(
                batch["text"],
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length",
            )
            inputs = {
                key: val.to(device) for key, val in inputs.items()
            }  # Move inputs to GPU
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
            # loss_ce = outputs.loss
            loss_ce = conditional_entropy_loss(outputs.logits, labels)
            loss_phi = bigphi_loss(model, config, inputs, labels)
            # Combine losses
            loss = loss_ce + loss_phi
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log_console_and_or_wandb(
                f"base_loss: {loss_ce.item()}",
                log_wandb=args.log_wandb,
                logger=logger,
                epoch=epoch,
                step=i,
            )
            log_console_and_or_wandb(
                f"mesa_loss: {loss_phi.item()}",
                log_wandb=args.log_wandb,
                logger=logger,
                epoch=epoch,
                step=i,
            )
            log_console_and_or_wandb(
                f"total_loss: {loss.item()}",
                log_wandb=args.log_wandb,
                logger=logger,
                epoch=epoch,
                step=i,
            )
            # print(
            #     f"Epoch [{epoch+1}/{args.epochs}], Step [{i}/{len(dataloader)}], Base Loss: {loss_ce.item()}, Mesa Loss: {loss_phi.item()}, Total Loss: {loss.item()}"
            # )
            results.append(
                {
                    "epoch": epoch,
                    "step": i,
                    "base_loss": loss_ce.item(),
                    "mesa_loss": loss_phi.item(),
                    "total_loss": loss.item(),
                }
            )
            wandb_tabularize_loss(results)
            del inputs, labels, outputs, loss_ce, loss_phi, loss
            free_memory()
        logger.info(f"Epoch {epoch+1}/{args.epochs} completed.")
        # save model with datetime string
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f"{args.output_dir}/{dt_str}", exist_ok=True)
        dt_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        model.save_pretrained(f"{args.output_dir}/{dt_str}/model_epoch{i}.pt")
        if args.log_wandb:
            wandb.save(f"{args.output_dir}/{dt_str}/model_epoch{i}.pt")
    return results

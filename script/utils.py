from logging import getLogger, Formatter, StreamHandler, DEBUG, INFO
import random

import numpy as np
import torch
import wandb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_handler(logger, handler, verbose: bool):
    """
    Prep handler
    """
    if verbose:
        handler.setLevel(DEBUG)
    else:
        handler.setLevel(INFO)
    formatter = Formatter(
        "%(asctime)s %(name)s:%(lineno)s [%(levelname)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_module_logger(verbose: bool = False, level=DEBUG):
    """
    Create logger
    """
    logger = getLogger(__name__)
    logger = _set_handler(logger, StreamHandler(), verbose)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def log_console_and_or_wandb(
    message,
    log_wandb=False,
    logger=None,
    epoch=None,
    step=None,
):
    """
    Log to console and/or wandb
    message format: "key: value"
    """
    if log_wandb:
        # convert to dict
        ky, metric = message.split(": ")
        if epoch is not None and step is not None:
            wandb.log({"epoch": epoch, "step": step, ky: float(metric)})
        elif epoch is not None:
            wandb.log({"epoch": epoch, ky: float(metric)})
        else:
            wandb.log({ky: float(metric)})
    else:
        pass
    if logger:
        logger.info(message)
    else:
        print(message)


def _filter_table_by_epoch(table, epoch_value):
    filtered_data = [row for row in table.data if row[0] == epoch_value]
    new_table = wandb.Table(columns=table.columns)
    for row in filtered_data:
        new_table.add_data(*row)
    return new_table


def wandb_tabularize_loss(entries):
    """
    Create loss table to wandb
    """
    # Create a table to hold the data
    table = wandb.Table(
        columns=["epoch", "step", "base_loss", "mesa_loss", "total_loss"]
    )
    # Add data to the table
    for entry in entries:
        table.add_data(
            entry["epoch"],
            entry["step"],
            entry["base_loss"],
            entry["mesa_loss"],
            entry["total_loss"],
        )
    wandb.log({"losses": table})
    # collect losses per epoch
    losses_epoch = {}
    for epoch in sorted(set(table.get_column("epoch"))):
        # Filter the table by epoch
        table_epoch = _filter_table_by_epoch(table, epoch)
        losses_epoch[epoch] = {
            "base_loss": table_epoch.get_column("base_loss"),
            "mesa_loss": table_epoch.get_column("mesa_loss"),
            "total_loss": table_epoch.get_column("total_loss"),
        }
    # Create the line series plot per loss type
    for loss_type in ["base_loss", "mesa_loss", "total_loss"]:
        line_plot = wandb.plot.line_series(
            xs=table.get_column("step"),
            ys=[losses_epoch[epoch][loss_type] for epoch in losses_epoch],
            keys=sorted(losses_epoch.keys()),
            title=f"{loss_type} per Step grouped by Epoch",
            xname="Step",
        )
        # Log the plot
        wandb.log({f"{loss_type}_plot": line_plot})
    # # Create the line series plot
    # line_plot = wandb.plot.line_series(
    #     xs=table.get_column("step"),
    #     ys=[
    #         table.get_column("base_loss"),
    #         table.get_column("mesa_loss"),
    #         table.get_column("total_loss"),
    #     ],
    #     keys=table.get_column("epoch"),
    #     title="Loss per Step grouped by Epoch",
    #     xname="Step",
    # )
    # Log the plot
    # wandb.log({"loss_plot": line_plot})


def free_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

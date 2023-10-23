import random
import torch
import transformers
import warnings

# Create logger with standard time based formatting
import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Ray imports
import ray
# Ray distributed training imports
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.huggingface import HuggingFaceTrainer
from ray.train.torch import TorchTrainer
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from ray.train.huggingface.transformers import (
    RayTrainReportCallback,
    prepare_trainer,
)

# HuggingFace imports
from ray.data.preprocessors import BatchMapper
from transformers import T5Tokenizer, T5ForConditionalGeneration

import numpy as np
import pandas as pd

from IPython.display import display, HTML
from typing import Any, Dict, List, Optional

transformers.set_seed(42)
warnings.simplefilter("ignore")

import datasets
#from utils import get_random_elements

# Global variables, can go in a config file
MODEL_NAME = "google/flan-t5-small"
NUM_WORKERS = 2
USE_GPU = False #True


def get_hf_dataset():
    hf_dataset = datasets.load_dataset("tatsu-lab/alpaca", split="train").train_test_split(
        test_size=0.2, seed=57
    )
    return hf_dataset

def train_test_split(hf_dataset):
    SMALL_DATA = True
    if SMALL_DATA:
        train_dataset =  ray.data.from_huggingface(hf_dataset['train']).limit(100)#ray_dataset["train"].limit(100)
        validation_dataset = ray.data.from_huggingface(hf_dataset['test']).limit(100)#ray_dataset["test"].limit(100)
    else:
        train_dataset = ray.data.from_huggingface(hf_dataset['train'])#ray_dataset["train"]
        validation_dataset = ray.data.from_huggingface(hf_dataset['test'])#ray_dataset["test"]
    return train_dataset, validation_dataset
    
def preprocess_function(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tokenizes the input and instruction pairs in a batch using the T5 tokenizer
    from the Google/flan-t5-base model, and returns a dictionary containing the
    encoded inputs and labels.

    Args:
        batch: A dictionary containing at least two keys, "instruction" and
        "input", whose values are lists of strings.

    Returns:
        A dictionary containing the encoded inputs and labels, as returned by
        the T5 tokenizer.
    """
    model_name = MODEL_NAME
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    encoded_inputs = tokenizer(
        list(batch["instruction"]),
        list(batch["input"]),
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    encoded_inputs["labels"] = encoded_inputs["input_ids"].copy()

    return dict(encoded_inputs)  


 
# [1] Define the full training function
# =====================================
def train_func(config): #Q: is this config used anywhere?
    #MODEL_NAME = "google/flan-t5-small"#"gpt2"
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, config=model_config)#AutoModelForCausalLM.from_config(model_config)

    # [2] Build Ray Data iterables
    # ============================
    train_dataset = ray.train.get_dataset_shard("train")
    eval_dataset = ray.train.get_dataset_shard("evaluation")

    train_iterable_ds = train_dataset.iter_torch_batches(batch_size=8)
    eval_iterable_ds = eval_dataset.iter_torch_batches(batch_size=8)

    args = transformers.TrainingArguments(
        report_to="none", #disable wandb
        output_dir=f"{MODEL_NAME}-alpaca-data",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        max_steps=100,
    )

    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=train_iterable_ds,
        eval_dataset=eval_iterable_ds,
    )

    # [3] Inject Ray Train Report Callback
    # ====================================
    trainer.add_callback(RayTrainReportCallback())

    # [4] Prepare your trainer
    # ========================
    trainer = prepare_trainer(trainer)
    trainer.train()

def get_finetuned_model(trained_model):
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    checkpoint = trained_model.checkpoint
    finetuned_model = checkpoint.get_model(model)
    return finetuned_model

def llm_inference(instruction: str, input_query: str, finetuned_model):
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    inputs = tokenizer(instruction, input_query, return_tensors="pt")
    outputs = finetuned_model.generate(**inputs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
def main():
    # Initialize Ray cluster
    ray.init(address="auto")
    #ray.init()
    logging.info(f"Ray cluster initialized with {ray.nodes()}")
        
    # Grab the HuggingFace dataset
    hf_dataset = get_hf_dataset()
    logging.info(f"Dataset: {hf_dataset}")
    
    # # To show there is some data, let's print out a few random examples
    # df = get_random_elements(dataset=hf_dataset["train"], num_examples=3)
    # logging.info(f"Random examples: {df}")
    
    # Get train, test, validation splits
    train_dataset, validation_dataset = train_test_split(hf_dataset)
    logging.info(f"Train dataset: {train_dataset}")
    logging.info(f"Validation dataset: {validation_dataset}")
    
    # Create a batch preprocessor based on the preprocess_function
    batch_preprocessor = BatchMapper(preprocess_function, batch_format="pandas", batch_size=4096)
    logging.info(f"Batch preprocessor: {batch_preprocessor}")
    
    # Build a Ray TorchTrainer
    scaling_config = ScalingConfig(num_workers=NUM_WORKERS, use_gpu=USE_GPU)
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        datasets={"train": train_dataset, "evaluation": validation_dataset},
    )
    logging.info(f"Ray trainer: {ray_trainer}")
    
    # Finetune the model
    result = ray_trainer.fit()
    logging.info(f"Resulting model: {result}")
    
    # finetuned_model = get_finetuned_model(result)
    # output = llm_inference(
    #         instruction="What is the capital of France?",
    #         input_query="The capital of France is",
    #         finetuned_model=finetuned_model)
    # logging.info(f"Output: {output}")


if __name__ == "__main__":
    main()
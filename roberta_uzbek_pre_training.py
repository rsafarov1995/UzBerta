import os

os.system("pip list | grep -E 'transformers|tokenizers'")

#%%time 
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path(".").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

os.system("mkdir RoBert_uzbek_568")
tokenizer.save_model("RoBert_uzbek_568")

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


tokenizer = ByteLevelBPETokenizer(
    "./RoBert_pubchem_500k/vocab.json",
    "./RoBert_pubchem_500k/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

#tokenizer.encode("Mi estas Julien.")

#tokenizer.encode("Mi estas Julien.").tokens
import torch
torch.cuda.is_available()

from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./RoBert_uzbek_568", max_len=512)

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)

model.num_parameters()

#%%time
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./kun_malumotlari.txt",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./RoBert_uzbek_568",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

from time import process_time_ns
t1_start = process_time_ns() 

trainer.train()

t1_stop = process_time_ns()
   
print("Ishladi:", t1_stop-t1_start) 

trainer.save_model("./RoBert_uzbek_568")
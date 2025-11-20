import loadTrainingData
from transformers import AutoTokenizer, AutoModelForCausalLM, LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os
from tqdm import tqdm

def train_gpt2(training_folder='training_data', model_name='gpt2', output_dir='./gpt2-finetuned', pad_token=None, extra_tokens=None):
    # Load training data
    all_training_data = loadTrainingData.load_training_data(training_folder)

    # Check if training data is empty
    if not all_training_data:
        print(f"No training data found in '{training_folder}'. Aborting training.")
        return

    # Concatenate all training data into a single string and save to a temporary file
    # This is because TextDataset expects a file path
    train_file_path = os.path.join(output_dir, "train.txt")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(train_file_path, "w") as f:
        for content in tqdm(all_training_data.values(), desc="Writing training data"):
            f.write(content + "\n")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # --- Handle Special Tokens ---
    special_tokens_to_add = {}
    if pad_token:
        special_tokens_to_add['pad_token'] = pad_token
    if extra_tokens:
        special_tokens_to_add['additional_special_tokens'] = extra_tokens

    if special_tokens_to_add:
        tokenizer.add_special_tokens(special_tokens_to_add)
        model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=train_file_path,
        block_size=128)

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=64,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        disable_tqdm=False,  # Explicitly enable the progress bar
        logging_strategy="epoch", # Log progress at the end of each epoch
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == '__main__':
    train_gpt2(extra_tokens=["<code>","</code>","<vars>","</vars>","<|sep|>","<|stop|>"])
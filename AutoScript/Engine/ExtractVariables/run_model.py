import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse

def generate_text(model_path, prompt, max_length=150, num_return_sequences=1):
    """
    Generates text using a fine-tuned GPT-2 model.

    Args:
        model_path (str): Path to the directory containing the fine-tuned model and tokenizer.
        prompt (str): The initial text to seed the generation.
        max_length (int): The maximum length of the generated text sequence.
        num_return_sequences (int): The number of different sequences to generate.
    """
    try:
        # Load the fine-tuned tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
    except OSError:
        print(f"Error: Model not found at '{model_path}'.")
        print("Please ensure you have trained the model and provided the correct path.")
        return

    # Encode the prompt text
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # --- Set Stop Token ---
    # Get the token ID for your custom stop token.
    stop_token_id = tokenizer.convert_tokens_to_ids("<|stop|>")

    # Generate text
    # Using torch.no_grad() is a good practice as we are not training
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            repetition_penalty=1.2,
            do_sample=False,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id, # Pad with EOS token
            eos_token_id=stop_token_id # Stop generation at this token

        )

    # Decode and print the generated text
    for i, generated_sequence in enumerate(output_sequences):
        text = tokenizer.decode(generated_sequence, skip_special_tokens=False)
        print(f"--- Generated Sequence {i+1} ---")
        print(text)

if __name__ == '__main__':
    # The default model_dir should match the output_dir in your training script
    model_dir = './gpt2-finetuned'
    example_prompt = "def my_function(variable_a, variable_b):"
    example_prompt = "function(foo(x,y),z)"
    example_prompt = "<code>foo(x,y)</code><vars>"
    generate_text(model_dir, example_prompt)
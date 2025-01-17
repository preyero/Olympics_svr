""" 
Run inference in HuggingFace models using transformers.pipeline text-generation abstraction.

"""
import argparse
import csv, os
import pandas as pd
import transformers
from huggingface_hub import login
import torch

# Default arguments

INPUT_FILE = "comments_sample.csv"
TEXT_COL = "text"

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PROMPTS = "Is the following post Misogynistic? (yes/no). Post:;Classify this text as Misogynistic or Non-Misogynistic. Text:"

def prompting(comment, prompts, model_pipeline):
    
    outputs = []
    for p in prompts:
        input = f"{p} {comment}. Answer:"
        out = model_pipeline(
            input,
            max_new_tokens=50,
        )
        outputs.append(out[0]["generated_text"].split("Answer:")[-1])

    return outputs

def analyse_comments(comments_tuples, prompts, model_pipeline, csv_file_name):

    n_comments = len(comments_tuples)
    with open(csv_file_name,'a') as f:
        writer = csv.writer(f)

        for i,c_tuple in enumerate(comments_tuples):
            print(f'{i}/{n_comments}')
            comment_id, comment = c_tuple

            try:
                response = prompting(comment, prompts, model_pipeline)

                writer.writerow([comment_id] + response)

            except Exception as e:
                print(f"An error occurred: {e}")

    return

def main():
    
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--hf_access_token", default=None, type=str, required=True, help="Access token to access Huggingface HUB")


    # Optional arguments
    parser.add_argument("--input_file", default=INPUT_FILE, type=str, required=False, help=f"Filename in CSV format (default: {INPUT_FILE})")
    parser.add_argument("--text_col", default=TEXT_COL, type=str, required=False, help=f"Name of column with texts (default: {TEXT_COL})")
    parser.add_argument("--model_id", default=MODEL_ID, type=str, required=False, help=f"Model_id from HuggingFace library (default: {MODEL_ID})")
    parser.add_argument("--prompts", default=PROMPTS, type=str, required=False, help=f"Prompts separated by ; (default: {PROMPTS}")

    args = parser.parse_args()

    # Login to huggingface hub using an access token (generate in Hugginface account)
    login(token=args.hf_access_token)
    print(f"HuggingFace HUB Authentication successful: {args.hf_access_token}")
    
    # Get videos ids by loading the CSV file into a DataFrame
    df = pd.read_csv(args.input_file)
    df = df.sample(10, random_state=1313).copy()
    comments = list(df[['id', 'text']].itertuples(index=False, name=None))
    print(f'Finished importing data from: {args.input_file}')

    # Set up large language model pipeline 
    model_pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_id,
        model_kwargs={"torch_dtype": torch.float16},
        device_map="auto",
    )
    print(f'Finished loading {args.model_id} using transformers pipeline')

    prompts = args.prompts.split(';')
    print(f'Prompting: {prompts}')

    # Set up output file
    model_name = args.model_id.split('/')[-1]
    csv_file_name = f"{args.input_file}_{model_name}_score.csv"

    if not os.path.exists(csv_file_name):
        with open(csv_file_name,'w') as f:
            writer = csv.writer(f)
            header = ['id'] + prompts
            writer.writerow(header)
        print(f'New output file created: {csv_file_name}')
    else:
        print(f'Updating existing output file at: {csv_file_name}')


    # Misogyny detection with LLMs
    analyse_comments(comments, prompts, model_pipeline, csv_file_name)
    print('Comments evaluated (original).')

    

if __name__ == '__main__':
    main()
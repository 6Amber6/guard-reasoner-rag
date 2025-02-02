import pandas as pd
import json

def merge(file1, file2, output_file):
    data1 = pd.read_json(file1)
    data2 = pd.read_json(file2)

    combined_df = pd.concat([data1, data2])

    result_df = combined_df.drop_duplicates(subset='input')

    save_dict_list = result_df.to_dict(orient='records')

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in save_dict_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')


model_list = ["guardreasoner_rsft_1b", "guardreasoner_rsft_3b", "guardreasoner_rsft_8b"]
dataset = ["ToxicChat", "Aegis", "BeaverTails"]

for data in dataset:
    for model in model_list:
        
        # merge data according to the training data 
        if data=="ToxicChat":
            merge(f"{data}_hard_sample_{model}_1.json", f"{data}_hard_sample_{model}_2.json", f"{data}_hard_sample_{model}_merge.json")
        elif data=="Aegis":
            merge(f"{data}_hard_sample_{model}_2.json", f"{data}_hard_sample_{model}_3.json", f"{data}_hard_sample_{model}_merge.json")
        elif data=="BeaverTails":
            merge(f"{data}_hard_sample_{model}_1.json", f"{data}_hard_sample_{model}_3.json", f"{data}_hard_sample_{model}_merge.json")


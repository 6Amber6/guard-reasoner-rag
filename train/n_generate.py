import os
import json
from vllm import LLM, SamplingParams

model_path_list = ["./saves/Llama-3.2-1B/full/guardreasoner_rsft_1b", "./saves/Llama-3.2-3B/full/guardreasoner_rsft_3b", "./saves/Llama-3.1-8B/full/guardreasoner_rsft_8b"]

for model_path in model_path_list:

    best_of_n=4
    vllm_model = LLM(model=model_path, gpu_memory_utilization=0.90, max_model_len=2048, max_num_seqs=128)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=2048, n=best_of_n)

    dataset = ["ToxicChat", "Aegis", "BeaverTails", "WildGuard"]

    for idx, data_name in enumerate(["ToxicChatTrainR", "AegisTrainR", "BeaverTailsTrainR", "WildGuardTrainR"]):
        with open(f"./data/{data_name}.json") as file:
            data = json.load(file)

        prompt_list = []

        for i, sample in enumerate(data):
            prompt_list.append(sample['instruction'] + "\n" + sample['input'])
        outputs = vllm_model.generate(prompt_list, sampling_params)

        for j in range(len(outputs[0].outputs)):
            save_dict_list = []
            
            for i, output in enumerate(outputs):
                prompt = output.prompt
                generated_text = output.outputs[j].text
                save_dict = {"prompt": prompt, "label": data[i]["output"], "predict": generated_text}
                save_dict_list.append(save_dict)

            
            file_path = f"{model_path}/{dataset[idx]}/{j}/generated_predictions.jsonl"
            
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in save_dict_list:
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + '\n')



# improve diversity
for model_path in model_path_list:
    for model_version in ["1", "2", "3"]:
        model_path += "_" + model_version
            
        best_of_n=4
        vllm_model = LLM(model=model_path, gpu_memory_utilization=0.90, max_model_len=2048, max_num_seqs=128)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=2048, n=best_of_n)

        dataset = ["ToxicChat", "Aegis", "BeaverTails", "WildGuard"]

        for idx, data_name in enumerate(["ToxicChatTrainR", "AegisTrainR", "BeaverTailsTrainR", "WildGuardTrainR"]):
            with open(f"./data/{data_name}.json") as file:
                data = json.load(file)

            prompt_list = []

            for i, sample in enumerate(data):
                prompt_list.append(sample['instruction'] + "\n" + sample['input'])
            outputs = vllm_model.generate(prompt_list, sampling_params)

            for j in range(len(outputs[0].outputs)):
                save_dict_list = []
                
                for i, output in enumerate(outputs):
                    prompt = output.prompt
                    generated_text = output.outputs[j].text
                    save_dict = {"prompt": prompt, "label": data[i]["output"], "predict": generated_text}
                    save_dict_list.append(save_dict)

                
                file_path = f"{model_path}/{dataset[idx]}/{j}/generated_predictions.jsonl"
                
                directory = os.path.dirname(file_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    for item in save_dict_list:
                        json_line = json.dumps(item, ensure_ascii=False)
                        f.write(json_line + '\n')


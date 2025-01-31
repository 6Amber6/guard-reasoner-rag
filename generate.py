import os
import json
from transformers import AutoModel
from vllm import LLM, SamplingParams

for model_size in ["8B", "3B", "1B"]:

    model = AutoModel.from_pretrained(f"yueliu1999/GuardReasoner-{model_size}")
    vllm_model = LLM(model=f"yueliu1999/GuardReasoner-{model_size}", gpu_memory_utilization=0.95, max_num_seqs=256)
    sampling_params = SamplingParams(temperature=0., top_p=1., max_tokens=2048)

    dataset = ["SimpleSafetyTests", "AegisSafetyTest", "OpenAIModeration", "HarmBenchPrompt", 
            "WildGuardTest", "HarmBenchResponse", "XSTestReponseHarmful", "XSTestResponseRefusal", 
            "ToxicChat", "SafeRLHF", "BeaverTails"]

    for idx, data_name in enumerate(["0_0_SimpleSafetyTests", "0_1_AegisSafetyTest", "0_2_OpenAIModeration", "0_3_harmbench_prompt", 
                                    "0_4_wild_guard_test", "0_5_harmbench_response", "0_6_xstest_response_harmful", 
                                    "0_7_xstest_response_refusal", "0_8_toxic_chat", "0_9_safe_rlhf", "0_A_beaver_tails"]):
        with open(f"./data/benchmark/{data_name}.json") as file:
            data = json.load(file)

        prompt_list = []

        for i, sample in enumerate(data):
            prompt_list.append(sample['instruction'] + "\n" + sample['input'])
            
        outputs = vllm_model.generate(prompt_list, sampling_params)

        save_dict_list = []
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            save_dict = {"prompt": prompt, "label": data[i]["output"], "predict": generated_text}
            save_dict_list.append(save_dict)

        
        file_path = f"./data/test/{model_size}/{dataset[idx]}/generated_predictions.jsonl"
        
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in save_dict_list:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')

    del vllm_model
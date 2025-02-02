import re
import json
import numpy as np
import pandas as pd
from collections import Counter

def find_mode(result_list):
    if not result_list:
        return None  
    counter = Counter(result_list)
    max_count = max(counter.values())
    modes = [k for k, v in counter.items() if v == max_count]
    return modes

model_path_list = ["./saves/Llama-3.2-1B/full/guardreasoner_rsft_1b", "./saves/Llama-3.2-3B/full/guardreasoner_rsft_3b", "./saves/Llama-3.1-8B/full/guardreasoner_rsft_8b"]
model_path_list_merge = []

# improve diversity
for model_path in model_path_list:
    model_path_list_merge.append(model_path)
    for model_version in ["1", "2", "3"]:
        model_path += "_" + model_version
        model_path_list_merge.append(model_path)
        
for model_path in model_path_list:
    # ------------------------------------------------------------------------------------
    data_list = []
    save_dict_list = []

    best_of_n = 4

    for i in range(best_of_n):
        file_name = f"{model_path_list}/WildGuard/{i}/generated_predictions.jsonl"
        with open(file_name) as file:
            data = pd.read_json(file, lines=True)
        data_list.append(data)

    for i in range(len(data_list[0]['predict'])):
        request_result_list = []
        response_result_list = []
        completion_result_list = []
        data_tmp = []
        try:
            pattern = r"Request:\s*(\w+)"
            ground_truth_request = re.search(pattern, data_list[0]['label'][i]).group(1)
            pattern = r"Response:\s*(\w+)"
            ground_truth_response = re.search(pattern, data_list[0]['label'][i]).group(1)
            pattern = r"Completion:\s*(\w+)"
            ground_truth_completion = re.search(pattern, data_list[0]['label'][i]).group(1)
        except:
            continue
        
        
        for j in range(best_of_n):
            
            try:
                pattern = r"Request:\s*(\w+)"
                match_request = re.search(pattern, data_list[j]['predict'][i]).group(1)
                
                pattern = r"Response:\s*(\w+)"
                match_response = re.search(pattern, data_list[j]['predict'][i]).group(1)
                
                pattern = r"Completion:\s*(\w+)"
                match_completion = re.search(pattern, data_list[j]['predict'][i]).group(1)
                
                response_result_list.append(match_response)
                request_result_list.append(match_request)
                completion_result_list.append(match_completion)
                
                data_tmp.append(data_list[j].iloc[i])

                
            except:
                pass

        
        try:
            request_mode = find_mode(request_result_list)[0]
            response_mode = find_mode(response_result_list)[0]
            completion_mode = find_mode(completion_result_list)[0]
        except:
            continue

        # only for request
        count = sum(1 for result in request_result_list if result == ground_truth_request)
        no_count = sum(1 for result in request_result_list if result != ground_truth_request)
        
        
        if no_count!=0 and count!=0:
            different_indices = [index for index, element in enumerate(request_result_list) if element != ground_truth_request]
            same_indices = [index for index, element in enumerate(request_result_list) if element == ground_truth_request]

            
            max_reject_index=0
            max_reject_len = -1
            for diff in different_indices:
                if max_reject_len < len(data_tmp[diff]["predict"]):
                    max_reject_len = len(data_tmp[diff]["predict"])
                    max_reject_index = diff
                    
            max_chosen_index=0
            max_chosen_len = -1
            for same in same_indices:
                if max_chosen_len < len(data_tmp[same]["predict"]):
                    max_chosen_len = len(data_tmp[same]["predict"])
                    max_chosen_index = same
            
            save_dict = {"instruction": data_tmp[max_chosen_index]["prompt"][:489], 
            "input": data_tmp[max_chosen_index]["prompt"][491:],
            "chosen": data_tmp[max_chosen_index]["predict"],
            "rejected": data_tmp[max_reject_index]["predict"],
            "sample_weight": len(different_indices)-len(same_indices),
            }
            save_dict_list.append(save_dict)

            continue
        
        # only for response
        pattern = r"AI assistant:\s*(\w+)"
        try:
            AI_reponse = re.search(pattern, data_list[0]['prompt'][i]).group(1)
        except:
            AI_reponse = ""
            
        if ground_truth_response != "None" and AI_reponse != "None":
            count = sum(1 for result in response_result_list if result == ground_truth_response)
            no_count = sum(1 for result in response_result_list if result != ground_truth_response)
            
            if no_count!=0 and count!=0:
                different_indices = [index for index, element in enumerate(response_result_list) if element != ground_truth_response]
                same_indices = [index for index, element in enumerate(response_result_list) if element == ground_truth_response]
            
                max_reject_index=0
                max_reject_len = -1
                for diff in different_indices:
                    if max_reject_len < len(data_tmp[diff]["predict"]):
                        max_reject_len = len(data_tmp[diff]["predict"])
                        max_reject_index = diff
                        
                max_chosen_index=0
                max_chosen_len = -1
                for same in same_indices:
                    if max_chosen_len < len(data_tmp[same]["predict"]):
                        max_chosen_len = len(data_tmp[same]["predict"])
                        max_chosen_index = same
                
                save_dict = {"instruction": data_tmp[max_chosen_index]["prompt"][:489], 
                "input": data_tmp[max_chosen_index]["prompt"][491:],
                "chosen": data_tmp[max_chosen_index]["predict"],
                "rejected": data_tmp[max_reject_index]["predict"],
                "sample_weight": len(different_indices)-len(same_indices),
                }
                save_dict_list.append(save_dict)
                continue
        
        # only for completion
        pattern = r"AI assistant:\s*(\w+)"
        try:
            AI_reponse = re.search(pattern, data_list[0]['prompt'][i]).group(1)
        except:
            AI_reponse = ""
            
        if ground_truth_completion != "None" and AI_reponse != "None":
            count = sum(1 for result in completion_result_list if result == ground_truth_completion)
            no_count = sum(1 for result in completion_result_list if result != ground_truth_completion)
            
            if no_count!=0 and count!=0:
                different_indices = [index for index, element in enumerate(completion_result_list) if element != ground_truth_completion]
                same_indices = [index for index, element in enumerate(completion_result_list) if element == ground_truth_completion]
            
                max_reject_index=0
                max_reject_len = -1
                for diff in different_indices:
                    if max_reject_len < len(data_tmp[diff]["predict"]):
                        max_reject_len = len(data_tmp[diff]["predict"])
                        max_reject_index = diff
                        
                max_chosen_index=0
                max_chosen_len = -1
                for same in same_indices:
                    if max_chosen_len < len(data_tmp[same]["predict"]):
                        max_chosen_len = len(data_tmp[same]["predict"])
                        max_chosen_index = same
                
                save_dict = {"instruction": data_tmp[max_chosen_index]["prompt"][:489], 
                "input": data_tmp[max_chosen_index]["prompt"][491:],
                "chosen": data_tmp[max_chosen_index]["predict"],
                "rejected": data_tmp[max_reject_index]["predict"],
                "sample_weight": len(different_indices)-len(same_indices),
                }
                save_dict_list.append(save_dict)
                continue
        


    sample_weight_list = []
    for i in save_dict_list:
        sample_weight_list.append(i["sample_weight"])
    sample_weight_list = np.array(sample_weight_list)
    max_norm = 0.2
    sample_weight_list = 1 + (sample_weight_list - min(sample_weight_list)) * max_norm*2 / (max(sample_weight_list) - min(sample_weight_list)) - max_norm

    for i, _ in enumerate(save_dict_list):
        save_dict_list[i]["sample_weight"] = sample_weight_list[i]

    with open(f"WildGuard_hard_sample_{model_path.split("/")[-1]}.json", 'w', encoding='utf-8') as f:
        json.dump(save_dict_list, f, ensure_ascii=False, indent=4)
    print("WildGuard:", len(save_dict_list))
    # ------------------------------------------------------------------------------------



    # ------------------------------------------------------------------------------------
    data_list = []
    save_dict_list = []
    for i in range(best_of_n):
        file_name = f"{model_path_list}/Aegis/{i}/generated_predictions.jsonl"
        with open(file_name) as file:
            data = pd.read_json(file, lines=True)
        data_list.append(data)
        

    for i in range(len(data_list[0]['predict'])):
        request_result_list = []
        response_result_list = []
        completion_result_list = []
        data_tmp = []
        try:
            pattern = r"Request:\s*(\w+)"
            ground_truth_request = re.search(pattern, data_list[0]['label'][i]).group(1)
            pattern = r"Response:\s*(\w+)"
            ground_truth_response = re.search(pattern, data_list[0]['label'][i]).group(1)
            pattern = r"Completion:\s*(\w+)"
            ground_truth_completion = re.search(pattern, data_list[0]['label'][i]).group(1)
        except:
            continue
        
        
        for j in range(best_of_n):
            
            try:
                pattern = r"Request:\s*(\w+)"
                match_request = re.search(pattern, data_list[j]['predict'][i]).group(1)
                
                pattern = r"Response:\s*(\w+)"
                match_response = re.search(pattern, data_list[j]['predict'][i]).group(1)
                
                pattern = r"Completion:\s*(\w+)"
                match_completion = re.search(pattern, data_list[j]['predict'][i]).group(1)
                
                response_result_list.append(match_response)
                request_result_list.append(match_request)
                completion_result_list.append(match_completion)
                
                data_tmp.append(data_list[j].iloc[i])

                
            except:
                pass

        try:
            request_mode = find_mode(request_result_list)[0]
            response_mode = find_mode(response_result_list)[0]
            completion_mode = find_mode(completion_result_list)[0]
        except:
            continue

        # only for request
        count = sum(1 for result in request_result_list if result == ground_truth_request)
        no_count = sum(1 for result in request_result_list if result != ground_truth_request)
        
            
        if no_count!=0 and count!=0:
            different_indices = [index for index, element in enumerate(request_result_list) if element != ground_truth_request]
            same_indices = [index for index, element in enumerate(request_result_list) if element == ground_truth_request]
        
            max_reject_index=0
            max_reject_len = -1
            for diff in different_indices:
                if max_reject_len < len(data_tmp[diff]["predict"]):
                    max_reject_len = len(data_tmp[diff]["predict"])
                    max_reject_index = diff
                    
            max_chosen_index=0
            max_chosen_len = -1
            for same in same_indices:
                if max_chosen_len < len(data_tmp[same]["predict"]):
                    max_chosen_len = len(data_tmp[same]["predict"])
                    max_chosen_index = same
            
            save_dict = {"instruction": data_tmp[max_chosen_index]["prompt"][:489], 
            "input": data_tmp[max_chosen_index]["prompt"][491:],
            "chosen": data_tmp[max_chosen_index]["predict"],
            "rejected": data_tmp[max_reject_index]["predict"],
                "sample_weight": len(different_indices)-len(same_indices),
            }
            save_dict_list.append(save_dict)
            continue
        
        # only for response
        pattern = r"AI assistant:\s*(\w+)"
        try:
            AI_reponse = re.search(pattern, data_list[0]['prompt'][i]).group(1)
        except:
            AI_reponse = ""
            
        if ground_truth_response != "None" and AI_reponse != "None":
            count = sum(1 for result in response_result_list if result == ground_truth_response)
            no_count = sum(1 for result in response_result_list if result != ground_truth_response)
            
            if no_count!=0 and count!=0:
                different_indices = [index for index, element in enumerate(response_result_list) if element != ground_truth_response]
                same_indices = [index for index, element in enumerate(response_result_list) if element == ground_truth_response]
            
                max_reject_index=0
                max_reject_len = -1
                for diff in different_indices:
                    if max_reject_len < len(data_tmp[diff]["predict"]):
                        max_reject_len = len(data_tmp[diff]["predict"])
                        max_reject_index = diff
                        
                max_chosen_index=0
                max_chosen_len = -1
                for same in same_indices:
                    if max_chosen_len < len(data_tmp[same]["predict"]):
                        max_chosen_len = len(data_tmp[same]["predict"])
                        max_chosen_index = same
                
                save_dict = {"instruction": data_tmp[max_chosen_index]["prompt"][:489], 
                "input": data_tmp[max_chosen_index]["prompt"][491:],
                "chosen": data_tmp[max_chosen_index]["predict"],
                "rejected": data_tmp[max_reject_index]["predict"],
                "sample_weight": len(different_indices)-len(same_indices),
                }
                save_dict_list.append(save_dict)
                continue
        
        # only for completion
        pattern = r"AI assistant:\s*(\w+)"
        try:
            AI_reponse = re.search(pattern, data_list[0]['prompt'][i]).group(1)
        except:
            AI_reponse = ""
            
        if ground_truth_completion != "None" and AI_reponse != "None":
            count = sum(1 for result in completion_result_list if result == ground_truth_completion)
            no_count = sum(1 for result in completion_result_list if result != ground_truth_completion)
            
            if no_count!=0 and count!=0:
                different_indices = [index for index, element in enumerate(completion_result_list) if element != ground_truth_completion]
                same_indices = [index for index, element in enumerate(completion_result_list) if element == ground_truth_completion]
            
                max_reject_index=0
                max_reject_len = -1
                for diff in different_indices:
                    if max_reject_len < len(data_tmp[diff]["predict"]):
                        max_reject_len = len(data_tmp[diff]["predict"])
                        max_reject_index = diff
                        
                max_chosen_index=0
                max_chosen_len = -1
                for same in same_indices:
                    if max_chosen_len < len(data_tmp[same]["predict"]):
                        max_chosen_len = len(data_tmp[same]["predict"])
                        max_chosen_index = same
                
                save_dict = {"instruction": data_tmp[max_chosen_index]["prompt"][:489], 
                "input": data_tmp[max_chosen_index]["prompt"][491:],
                "chosen": data_tmp[max_chosen_index]["predict"],
                "rejected": data_tmp[max_reject_index]["predict"],
                "sample_weight": len(different_indices)-len(same_indices),
                }
                save_dict_list.append(save_dict)
                continue
        



    sample_weight_list = []
    for i in save_dict_list:
        sample_weight_list.append(i["sample_weight"])
    sample_weight_list = np.array(sample_weight_list)
    max_norm = 0.2
    sample_weight_list = 1 + (sample_weight_list - min(sample_weight_list)) * max_norm*2 / (max(sample_weight_list) - min(sample_weight_list)) - max_norm


    for i, _ in enumerate(save_dict_list):
        save_dict_list[i]["sample_weight"] = sample_weight_list[i]


    with open(f"Aegis_hard_sample_{model_path.split("/")[-1]}.json", 'w', encoding='utf-8') as f:
        json.dump(save_dict_list, f, ensure_ascii=False, indent=4)

    print("Aegis", len(save_dict_list))
        

    # ------------------------------------------------------------------------------------
    data_list = []
    save_dict_list = []
    for i in range(best_of_n):
        file_name = f"{model_path_list}/ToxicChat/{i}/generated_predictions.jsonl"
        with open(file_name) as file:
            data = pd.read_json(file, lines=True)
        data_list.append(data)
        

    for i in range(len(data_list[0]['predict'])):
        request_result_list = []
        response_result_list = []
        completion_result_list = []
        data_tmp = []
        try:
            pattern = r"Request:\s*(\w+)"
            ground_truth_request = re.search(pattern, data_list[0]['label'][i]).group(1)
            pattern = r"Response:\s*(\w+)"
            ground_truth_response = re.search(pattern, data_list[0]['label'][i]).group(1)
            pattern = r"Completion:\s*(\w+)"
            ground_truth_completion = re.search(pattern, data_list[0]['label'][i]).group(1)
        except:
            continue
        
        
        for j in range(best_of_n):
            
            try:
                pattern = r"Request:\s*(\w+)"
                match_request = re.search(pattern, data_list[j]['predict'][i]).group(1)
                
                pattern = r"Response:\s*(\w+)"
                match_response = re.search(pattern, data_list[j]['predict'][i]).group(1)
                
                pattern = r"Completion:\s*(\w+)"
                match_completion = re.search(pattern, data_list[j]['predict'][i]).group(1)
                
                response_result_list.append(match_response)
                request_result_list.append(match_request)
                completion_result_list.append(match_completion)
                
                data_tmp.append(data_list[j].iloc[i])

                
            except:
                pass

        try:
            request_mode = find_mode(request_result_list)[0]
            response_mode = find_mode(response_result_list)[0]
            completion_mode = find_mode(completion_result_list)[0]
        except:
            continue

        # only for request
        count = sum(1 for result in request_result_list if result == ground_truth_request)
        no_count = sum(1 for result in request_result_list if result != ground_truth_request)
        
            
        if no_count!=0 and count!=0:
            different_indices = [index for index, element in enumerate(request_result_list) if element != ground_truth_request]
            same_indices = [index for index, element in enumerate(request_result_list) if element == ground_truth_request]
        
            max_reject_index=0
            max_reject_len = -1
            for diff in different_indices:
                if max_reject_len < len(data_tmp[diff]["predict"]):
                    max_reject_len = len(data_tmp[diff]["predict"])
                    max_reject_index = diff
                    
            max_chosen_index=0
            max_chosen_len = -1
            for same in same_indices:
                if max_chosen_len < len(data_tmp[same]["predict"]):
                    max_chosen_len = len(data_tmp[same]["predict"])
                    max_chosen_index = same
            
            save_dict = {"instruction": data_tmp[max_chosen_index]["prompt"][:489], 
            "input": data_tmp[max_chosen_index]["prompt"][491:],
            "chosen": data_tmp[max_chosen_index]["predict"],
            "rejected": data_tmp[max_reject_index]["predict"],
            "sample_weight": len(different_indices)-len(same_indices),
            }
            save_dict_list.append(save_dict)
            continue
        
        # only for response
        pattern = r"AI assistant:\s*(\w+)"
        try:
            AI_reponse = re.search(pattern, data_list[0]['prompt'][i]).group(1)
        except:
            AI_reponse = ""
            
        if ground_truth_response != "None" and AI_reponse != "None":
            count = sum(1 for result in response_result_list if result == ground_truth_response)
            no_count = sum(1 for result in response_result_list if result != ground_truth_response)
            
            if no_count!=0 and count!=0:
                different_indices = [index for index, element in enumerate(response_result_list) if element != ground_truth_response]
                same_indices = [index for index, element in enumerate(response_result_list) if element == ground_truth_response]
            
                max_reject_index=0
                max_reject_len = -1
                for diff in different_indices:
                    if max_reject_len < len(data_tmp[diff]["predict"]):
                        max_reject_len = len(data_tmp[diff]["predict"])
                        max_reject_index = diff
                        
                max_chosen_index=0
                max_chosen_len = -1
                for same in same_indices:
                    if max_chosen_len < len(data_tmp[same]["predict"]):
                        max_chosen_len = len(data_tmp[same]["predict"])
                        max_chosen_index = same
                
                save_dict = {"instruction": data_tmp[max_chosen_index]["prompt"][:489], 
                "input": data_tmp[max_chosen_index]["prompt"][491:],
                "chosen": data_tmp[max_chosen_index]["predict"],
                "rejected": data_tmp[max_reject_index]["predict"],
                "sample_weight": len(different_indices)-len(same_indices),
                }
                save_dict_list.append(save_dict)
                continue
        
        # only for completion
        pattern = r"AI assistant:\s*(\w+)"
        try:
            AI_reponse = re.search(pattern, data_list[0]['prompt'][i]).group(1)
        except:
            AI_reponse = ""
            
        if ground_truth_completion != "None" and AI_reponse != "None":
            count = sum(1 for result in completion_result_list if result == ground_truth_completion)
            no_count = sum(1 for result in completion_result_list if result != ground_truth_completion)
            
            if no_count!=0 and count!=0:
                different_indices = [index for index, element in enumerate(completion_result_list) if element != ground_truth_completion]
                same_indices = [index for index, element in enumerate(completion_result_list) if element == ground_truth_completion]
                
                max_reject_index=0
                max_reject_len = -1
                for diff in different_indices:
                    if max_reject_len < len(data_tmp[diff]["predict"]):
                        max_reject_len = len(data_tmp[diff]["predict"])
                        max_reject_index = diff
                        
                max_chosen_index=0
                max_chosen_len = -1
                for same in same_indices:
                    if max_chosen_len < len(data_tmp[same]["predict"]):
                        max_chosen_len = len(data_tmp[same]["predict"])
                        max_chosen_index = same
                
                save_dict = {"instruction": data_tmp[max_chosen_index]["prompt"][:489], 
                "input": data_tmp[max_chosen_index]["prompt"][491:],
                "chosen": data_tmp[max_chosen_index]["predict"],
                "rejected": data_tmp[max_reject_index]["predict"],
                "sample_weight": len(different_indices)-len(same_indices),
                }
                save_dict_list.append(save_dict)
                continue
            
        


    sample_weight_list = []
    for i in save_dict_list:
        sample_weight_list.append(i["sample_weight"])
    sample_weight_list = np.array(sample_weight_list)
    max_norm = 0.2
    sample_weight_list = 1 + (sample_weight_list - min(sample_weight_list)) * max_norm*2 / (max(sample_weight_list) - min(sample_weight_list)) - max_norm


    for i, _ in enumerate(save_dict_list):
        save_dict_list[i]["sample_weight"] = sample_weight_list[i]


    with open(f"Toxic_hard_sample_{model_path.split("/")[-1]}.json", 'w', encoding='utf-8') as f:
        json.dump(save_dict_list, f, ensure_ascii=False, indent=4)
        

    print("Toxic", len(save_dict_list))

    # ------------------------------------------------------------------------------------
    data_list = []
    save_dict_list = []

    for i in range(best_of_n):
        file_name = f"{model_path_list}/BeaverTails/{i}/generated_predictions.jsonl"
        with open(file_name) as file:
            data = pd.read_json(file, lines=True)
        data_list.append(data)
        
    for i in range(len(data_list[0]['predict'])):
        request_result_list = []
        response_result_list = []
        completion_result_list = []
        data_tmp = []
        try:
            pattern = r"Request:\s*(\w+)"
            ground_truth_request = re.search(pattern, data_list[0]['label'][i]).group(1)
            pattern = r"Response:\s*(\w+)"
            ground_truth_response = re.search(pattern, data_list[0]['label'][i]).group(1)
            pattern = r"Completion:\s*(\w+)"
            ground_truth_completion = re.search(pattern, data_list[0]['label'][i]).group(1)
        except:
            continue
        
        
        for j in range(best_of_n):
            
            try:
                pattern = r"Request:\s*(\w+)"
                match_request = re.search(pattern, data_list[j]['predict'][i]).group(1)
                
                pattern = r"Response:\s*(\w+)"
                match_response = re.search(pattern, data_list[j]['predict'][i]).group(1)
                
                pattern = r"Completion:\s*(\w+)"
                match_completion = re.search(pattern, data_list[j]['predict'][i]).group(1)
                
                response_result_list.append(match_response)
                request_result_list.append(match_request)
                completion_result_list.append(match_completion)
                
                data_tmp.append(data_list[j].iloc[i])

                
            except:
                pass

        if ground_truth_request=="None":
            pass
        
        try:
            request_mode = find_mode(request_result_list)[0]
            response_mode = find_mode(response_result_list)[0]
            completion_mode = find_mode(completion_result_list)[0]
        except:
            continue

        # only for request
        count = sum(1 for result in request_result_list if result == ground_truth_request)
        no_count = sum(1 for result in request_result_list if result != ground_truth_request)
        
            
        if no_count!=0 and count!=0:
            different_indices = [index for index, element in enumerate(request_result_list) if element != ground_truth_request]
            same_indices = [index for index, element in enumerate(request_result_list) if element == ground_truth_request]
        
            max_reject_index=0
            max_reject_len = -1
            for diff in different_indices:
                if max_reject_len < len(data_tmp[diff]["predict"]):
                    max_reject_len = len(data_tmp[diff]["predict"])
                    max_reject_index = diff
                    
            max_chosen_index=0
            max_chosen_len = -1
            for same in same_indices:
                if max_chosen_len < len(data_tmp[same]["predict"]):
                    max_chosen_len = len(data_tmp[same]["predict"])
                    max_chosen_index = same
            
            save_dict = {"instruction": data_tmp[max_chosen_index]["prompt"][:489], 
            "input": data_tmp[max_chosen_index]["prompt"][491:],
            "chosen": data_tmp[max_chosen_index]["predict"],
            "rejected": data_tmp[max_reject_index]["predict"],
            "sample_weight": len(different_indices)-len(same_indices),
            }
            save_dict_list.append(save_dict)
            continue
        
        # only for response
        pattern = r"AI assistant:\s*(\w+)"
        try:
            AI_reponse = re.search(pattern, data_list[0]['prompt'][i]).group(1)
        except:
            AI_reponse = ""
            
        if ground_truth_response != "None" and AI_reponse != "None":
            count = sum(1 for result in response_result_list if result == ground_truth_response)
            no_count = sum(1 for result in response_result_list if result != ground_truth_response)
            
            if no_count!=0 and count!=0:
                different_indices = [index for index, element in enumerate(response_result_list) if element != ground_truth_response]
                same_indices = [index for index, element in enumerate(response_result_list) if element == ground_truth_response]
            
                max_reject_index=0
                max_reject_len = -1
                for diff in different_indices:
                    if max_reject_len < len(data_tmp[diff]["predict"]):
                        max_reject_len = len(data_tmp[diff]["predict"])
                        max_reject_index = diff
                        
                max_chosen_index=0
                max_chosen_len = -1
                for same in same_indices:
                    if max_chosen_len < len(data_tmp[same]["predict"]):
                        max_chosen_len = len(data_tmp[same]["predict"])
                        max_chosen_index = same
                
                save_dict = {"instruction": data_tmp[max_chosen_index]["prompt"][:489], 
                "input": data_tmp[max_chosen_index]["prompt"][491:],
                "chosen": data_tmp[max_chosen_index]["predict"],
                "rejected": data_tmp[max_reject_index]["predict"],
                "sample_weight": len(different_indices)-len(same_indices),
                }
                save_dict_list.append(save_dict)
                continue
        
        # only for completion
        pattern = r"AI assistant:\s*(\w+)"
        try:
            AI_reponse = re.search(pattern, data_list[0]['prompt'][i]).group(1)
        except:
            AI_reponse = ""
            
        if ground_truth_completion != "None" and AI_reponse != "None":
            count = sum(1 for result in completion_result_list if result == ground_truth_completion)
            no_count = sum(1 for result in completion_result_list if result != ground_truth_completion)
            
            if no_count!=0 and count!=0:
                different_indices = [index for index, element in enumerate(completion_result_list) if element != ground_truth_completion]
                same_indices = [index for index, element in enumerate(completion_result_list) if element == ground_truth_completion]
            
                max_reject_index=0
                max_reject_len = -1
                for diff in different_indices:
                    if max_reject_len < len(data_tmp[diff]["predict"]):
                        max_reject_len = len(data_tmp[diff]["predict"])
                        max_reject_index = diff
                        
                max_chosen_index=0
                max_chosen_len = -1
                for same in same_indices:
                    if max_chosen_len < len(data_tmp[same]["predict"]):
                        max_chosen_len = len(data_tmp[same]["predict"])
                        max_chosen_index = same
                
                save_dict = {"instruction": data_tmp[max_chosen_index]["prompt"][:489], 
                "input": data_tmp[max_chosen_index]["prompt"][491:],
                "chosen": data_tmp[max_chosen_index]["predict"],
                "rejected": data_tmp[max_reject_index]["predict"],
                "sample_weight": len(different_indices)-len(same_indices),
                }
                save_dict_list.append(save_dict)
                continue
        
            
    sample_weight_list = []
    for i in save_dict_list:
        sample_weight_list.append(i["sample_weight"])
    sample_weight_list = np.array(sample_weight_list)
    max_norm = 0.2
    sample_weight_list = 1 + (sample_weight_list - min(sample_weight_list)) * max_norm*2 / (max(sample_weight_list) - min(sample_weight_list)) - max_norm


    for i, _ in enumerate(save_dict_list):
        save_dict_list[i]["sample_weight"] = sample_weight_list[i]


    with open(f"Beavertails_hard_sample_{model_path.split("/")[-1]}.json", 'w', encoding='utf-8') as f:
        json.dump(save_dict_list, f, ensure_ascii=False, indent=4)

    print("Beavertails", len(save_dict_list))
    # ------------------------------------------------------------------------------------

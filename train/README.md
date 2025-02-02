## Training Pipeline of GuardReasoner

### Step 1: Reasoning Data Synthesis
We release the synthesized reasoning data [GuardReasonerTrain](https://huggingface.co/datasets/yueliu1999/GuardReasonerTrain). Use it directly.

If you plan to synthesize the reasoning steps on your own data, run the following code.

```
export OPENAI_API_KEY=your_openai_api_key

python reasoning_data_synthesis.py
```


### Step 2: Reasoning SFT
1. Prepare the rasoning data for R-SFT.

    ```
    python prepare_data_rsft.py
    ```
2. Move `WildGuardTrainR.json`, `AegisTrainR.json`, `BeaverTailsTrainR.json`, `ToxicChatTrainR.json` to data folder in LLaMA-Factory and configre `dataset_info.json` as follows.

    ```
    "WildGuardTrainR": {
      "file_name": "WildGuardTrainR.json"
    },
    "AegisTrainR": {
      "file_name": "AegisTrainR.json"
    },
    "BeaverTailsTrainR": {
      "file_name": "BeaverTailsTrainR.json"
    },
    "ToxicChatTrainR": {
      "file_name": "ToxicChatTrainR.json"
    },
    ```
3. Run R-SFT via LLaMA-Factory. The scripts are provided.  
    ```
    bash R-SFT-1b.sh
    bash R-SFT-3b.sh
    bash R-SFT-8b.sh
    ```





### Step 3: Hard Sample DPO

1. Train reasoning models to improve the diversity of hard samples.
    ```
    R-SFT_1b_diversity.sh
    R-SFT_3b_diversity.sh
    R-SFT_8b_diversity.sh
    ```

2. Conduct n generations.
    ```
    python n_generate.py
    ```

3. Hard sample mining
    ```
    python hard_sample_mining.py
    python merge_hard_sample.py
    ```

4. Move `WildGuard_hard_sample_*.json`, `Aegis_hard_sample_*.json`, `BeaverTails_hard_sample_*.json`, `ToxicChat_hard_sample_*.json` to data folder in LLaMA-Factory and configre `dataset_info.json` as follows. For example, 

    ```
    "WildGuardTrainHS_8B": {
      "file_name": "WildGuard_hard_sample_guardreasoner_rsft_8b.json",
      "ranking": true,
      "columns": {
        "prompt": "instruction",
        "query": "input",
        "chosen": "chosen",
        "rejected": "rejected"
      }
    },

    "AegisTrainHS_8B_merge": {
      "file_name": "Aegis_hard_sample_guardreasoner_rsft_8b_merge.json",
      "ranking": true,
      "columns": {
        "prompt": "instruction",
        "query": "input",
        "chosen": "chosen",
        "rejected": "rejected"
      }
    },
    "BeaverTailsTrainHS_8B_merge": {
      "file_name": "BeaverTails_hard_sample_guardreasoner_rsft_8b_merge.json",
      "ranking": true,
      "columns": {
        "prompt": "instruction",
        "query": "input",
        "chosen": "chosen",
        "rejected": "rejected"
      }
    },
    "ToxicChatTrainHS_8B_merge": {
      "file_name": "ToxicChat_hard_sample_guardreasoner_rsft_8b_merge.json",
      "ranking": true,
      "columns": {
        "prompt": "instruction",
        "query": "input",
        "chosen": "chosen",
        "rejected": "rejected"
      }
    },
    ```

5. Conduct HS-DPO via LLaMA-Factory. You need to modify LLaMA-Factory and assign `sample_weight` in datasets for each sample during the DPO process. The scripts are provided. 
    ```
    bash HS-DPO_1b.sh
    bash HS-DPO_3b.sh
    bash HS-DPO_8b.sh
    ```

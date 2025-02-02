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

1. Hard sample mining


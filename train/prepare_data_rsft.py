from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("yueliu1999/GuardReasonerTrain")

ds["WildGuardTrainR"].to_json("WildGuardTrainR.json")
ds["AegisTrainR"].to_json("AegisTrainR.json")
ds["BeaverTailsTrainR"].to_json("BeaverTailsTrainR.json")
ds["ToxicChatTrainR"].to_json("ToxicChatTrainR.json")


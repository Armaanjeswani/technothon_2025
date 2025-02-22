import pandas as pd
import numpy as np
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Set paths
model_path = r"C:\Users\jeswa\.llama\checkpoints\Llama3.2-3B"  # Converted LLaMA model path
data_path = r"D:\Psy\psy_prediction\physiomize data2.csv"  # Path to your CSV dataset

# Load tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# Load dataset
data = pd.read_csv(data_path)

# Generate missing parameters (Pain Intensity: 1-10, Pain Location)
locations = ["Anterior Knee", "Medial Knee", "Lateral Knee", "Posterior Knee"]
data["Pain_Intensity"] = np.random.randint(1, 11, size=len(data))
data["Pain_Location"] = np.random.choice(locations, size=len(data))

# Convert data to text format for LLaMA
data["text"] = data.apply(lambda row: 
    f"Pain Intensity: {row['Pain_Intensity']}, Pain Location: {row['Pain_Location']}, ROM Progression: {list(row[:-1])}, Predicted Weeks: {row['time']}",
    axis=1)

# Convert to Hugging Face Dataset format
hf_dataset = Dataset.from_pandas(data[["text"]])

# Load model
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir="./llama_trained",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset
)

# Train model
trainer.train()

# Save model
trainer.save_model("./llama_trained")

# =========================== PREDICTION ===========================

# Load trained model
model = LlamaForCausalLM.from_pretrained("./llama_trained", torch_dtype=torch.float16, device_map="auto")

# User input
pain_intensity = int(input("Enter pain intensity (1-10): "))
pain_location = input("Enter pain location (Anterior/Medial/Lateral/Posterior Knee): ")
num_weeks = int(input("Enter number of weeks for ROM progression: "))

rom_values = []
for i in range(num_weeks):
    rom_values.append(float(input(f"Enter ROM for week {i+1}: ")))

# Prepare input text
input_text = f"Pain Intensity: {pain_intensity}, Pain Location: {pain_location}, ROM Progression: {rom_values}, Predicted Weeks:"

# Tokenize input
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

# Generate prediction
output = model.generate(input_ids, max_length=50)
predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Extract predicted weeks
predicted_weeks = predicted_text.split("Predicted Weeks:")[-1].strip()

print(f"Predicted time to reach 140° ROM: {predicted_weeks} weeks")

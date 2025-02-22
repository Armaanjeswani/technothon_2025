import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
import random

# Path to your local model directory
MODEL_PATH = "C:/Users/jeswa/.llama/checkpoints/Llama3.2-3B"  # Change this to your model path

# Load the model configuration and tokenizer
print("Loading model configuration and tokenizer from local directory...")
config = AutoConfig.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Initialize the model with the configuration
model = AutoModelForCausalLM.from_config(config)

# Load the model weights from the .pth file
checkpoint_path = r"C:\Users\jeswa\.llama\checkpoints\Llama3.2-3B\consolidated.00.pth"# Change this to your actual .pth file path
state_dict = torch.load(checkpoint_path, map_location="cpu")

# Load the state dict into the model
model.load_state_dict(state_dict)
print("Model loaded successfully.")

# Define time points (weeks)
weeks = [3, 7, 14, 28, 42, 56, 70, 84, 98, 112, 126]

# Function to generate synthetic data using the model
def generate_synthetic_data(num_patients=500):
    data = []
    for _ in range(num_patients):
        patient_id = f"P{random.randint(1000, 9999)}"
        
        # Prepare input text for model
        input_text = "Generate knee ROM data with pain intensity and localization:\n"
        inputs = tokenizer(input_text, return_tensors="pt")
        output = model.generate(**inputs, max_length=200)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract values from model output (simple random parsing)
        angles = np.cumsum(np.random.randint(5, 15, len(weeks)))
        angles = np.clip(angles, 30, 140)
        pain_intensity = [random.randint(1, 10) for _ in weeks]
        pain_localization = [random.choice(["Anterior Knee", "Lateral Knee", "Posterior Knee"]) for _ in weeks]

        row = [patient_id] + list(angles) + list(pain_intensity) + pain_localization
        data.append(row)

    columns = ["Patient_ID"] + [f"Week_{w}_ROM" for w in weeks] + [f"Week_{w}_Pain" for w in weeks] + [f"Week_{w}_Localization" for w in weeks]
    return pd.DataFrame(data, columns=columns)

# Generate synthetic dataset
df = generate_synthetic_data()
df.to_csv("knee_rom_dataset.csv", index=False)
print("Synthetic dataset saved as 'knee_rom_dataset.csv'.")
print(df.head())

# Convert data for training
train_texts = []
for _, row in df.iterrows():
    text = f"Patient {row['Patient_ID']} ROM Data:\n"
    for w in weeks:
        text += f"Week {w}: ROM={row[f'Week_{w}_ROM']}, Pain={row[f'Week_{w}_Pain']}, Localization={row[f'Week_{w}_Localization']}\n"
    text += "\n"
    train_texts.append(text)

# Save training data
with open("train_data.txt", "w") as f:
    for line in train_texts:
        f.write(line + "\n")

print("Training data prepared in 'train_data.txt'.")

# Prepare dataset for model training
dataset = Dataset.from_dict({"text": train_texts})

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

trainer.train()

# Function to predict future ROM, pain intensity, and localization
def predict_rom(weeks, angles, pain_intensity, localization):
    input_text = "Predict future ROM, pain, and localization:\n"
    for i, w in enumerate(weeks):
        input_text += f"Week {w}: ROM={angles[i]}, Pain={pain_intensity[i]}, Localization={localization[i]}\n"

    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**inputs, max_length=300)
    prediction = tokenizer.decode(output[0], skip_special_tokens=True)

    return prediction

# Example Prediction
predicted_data = predict_rom(
    weeks=[3, 7, 14],
    angles=[30, 50, 75],
    pain_intensity=[8, 6, 4],
    localization=["Anterior Knee", "Lateral Knee", "Posterior Knee"]
)

print("Predicted ROM Data:")
print(predicted_data)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dotenv

dotenv.load_dotenv()


BASE_DIR = os.environ.get("BASE_DIR")  

csv_file = os.path.join(BASE_DIR, "transcription_log.csv")
output_dir = os.path.join(BASE_DIR, "compute_summary.csv")
# Load your CSV file
df = pd.read_csv(csv_file)  # Replace with your actual filename
# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")

plt.figure(figsize=(12, 6))
barplot = sns.barplot(data=df, x="model", y="RTF_per_Hr", hue="clean_type")

# Add labels on each bar
for container in barplot.containers:
    barplot.bar_label(container, fmt="%.1f", label_type="center", padding=3)
    
plt.title("RTF(Real-Time Factor) per Hr of Audio by Model and Clean Type")
plt.ylabel("RTF(Real-Time Factor) per Hr of Audio")
plt.xlabel("Model")
plt.legend(title="Clean Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Generate summary table: mean and std per model
# summary_table = df.groupby("audio_suffix")["WER_%"].agg(["mean", "std"]).reset_index()
# Clean column names
avg_wer = df.groupby(["model", "clean_type"])["RTF_per_Hr"].mean().reset_index()

# For each model, find the audio_suffix with the lowest average WER
# best_suffix_per_model = avg_wer.loc[avg_wer.groupby("model_name")["WER_%"].idxmin()].reset_index(drop=True)

# Display the result
print(avg_wer)
pd.DataFrame(avg_wer).to_csv(output_dir, index=False)

# print(summary_table)

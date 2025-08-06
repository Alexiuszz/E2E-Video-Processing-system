import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dotenv

dotenv.load_dotenv()


BASE_DIR = os.environ.get("BASE_DIR")  
csv_file = os.path.join(BASE_DIR, "WER_Full.csv")
output_dir = os.path.join(BASE_DIR, "WER_summary2.csv")

# Load your CSV file
df = pd.read_csv(csv_file)  # Replace with your actual filename
# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")

# Replace with the suffix you're interested in


# # Filter the DataFrame
# filtered_df = df[df["clean_type"] == "noisy" ]

# # Plot WER distribution per model
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=df, x="model", y="WER_%")
# sns.stripplot(data=df, x="model", y="WER_%", hue="clean_type", dodge=True, marker='o', alpha=0.7)
# plt.title("WER Distribution by Model (Colored by Audio preprocessing algorithm)")
# plt.ylabel("WER (%)")
# plt.xlabel("Model")
# plt.legend(title="Clean Type", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# Group by model and audio suffix, then compute average WER
avg_wer = df.groupby(["model", "clean_type"])["WER_%"].mean().reset_index()

# For each model, find the audio_suffix with the lowest average WER
best_suffix_per_model = avg_wer.loc[avg_wer.groupby("model")["WER_%"].idxmin()].reset_index(drop=True)

# Display the result
# print(avg_wer)
# # Generate summary table: mean and std per model
# summary_table = filtered_df.groupby("model")["WER_%"].agg(["mean", "std"]).reset_index()
# # Clean column names
# avg_wer = df.groupby(["model", "clean_type"])["WER_%"].mean().reset_index()

# # For each model, find the clean_type with the lowest average WER
# best_suffix_per_model = avg_wer.loc[avg_wer.groupby("model")["WER_%"].idxmin()].reset_index(drop=True)

# # Display the result
# print(avg_wer)
pd.DataFrame(avg_wer).to_csv(output_dir, index=False)

# print(summary_table)

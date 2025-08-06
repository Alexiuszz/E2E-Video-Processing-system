import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data from image as structured by user
data = {
    "model": [
        "fastconformer", "fastconformer", "fastconformer", "fastconformer",
        "large", "large", "large", "large",
        "medium", "medium", "medium", "medium",
        "parakeet", "parakeet", "parakeet", "parakeet",
        "openai", "openai", "openai", "openai"
    ],
    "clean_type": [
        "denoised_nr", "denoised_vad", "noisy", "noisy_DeepFilterNet2",
        "denoised_nr", "denoised_vad", "noisy", "noisy_DeepFilterNet2",
        "denoised_nr", "denoised_vad", "noisy", "noisy_DeepFilterNet2",
        "denoised_nr", "denoised_vad", "noisy", "noisy_DeepFilterNet2",
        "denoised_nr", "denoised_vad", "noisy", "noisy_DeepFilterNet2"
    ],
    "WER_%": [
        21.50, 17.84, 16.96, 16.24,
        10.90, 9.80, 8.19, 9.33,
        10.70, 9.55, 7.62, 8.27,
        11.52, 9.95, 9.34, 9.47,
        8.29, 6.14, 5.71, 6.10
    ]
}

df = pd.DataFrame(data)

# Generate LaTeX table
latex_table = df.to_latex(index=False, label="tab:wer_results", caption="Word Error Rates (WER\%) for Each Model and Audio Type")

# Plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(x="model", y="WER_%", hue="clean_type", data=df)
plt.title("WER% by ASR Model and Audio Enhancement Type")
plt.ylabel("WER (%)")
plt.xlabel("ASR Model")
plt.xticks(rotation=45)
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.2f}', 
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                fontsize=8, color='black', xytext=(0, 2),
                textcoords='offset points')
plt.tight_layout()
plt.show()


plt.show()



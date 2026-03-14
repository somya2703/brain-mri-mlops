import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("batch_predictions.csv")  
y_pred_str = df.prediction.map({0: 'no', 1: 'yes'})

cm = confusion_matrix(df.true_label, y_pred_str)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['no','yes'], yticklabels=['no','yes'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("results/confusion_matrix.png")

#plt.show()


# Confidence Distribution
plt.figure()

plt.hist(df.confidence, bins=20)

plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")

plt.savefig("results/confidence_distribution.png")

#plt.show()

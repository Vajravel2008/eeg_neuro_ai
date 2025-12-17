import gradio as gr
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px

# -------- LOAD MODEL --------
model = tf.keras.models.load_model("best_model.h5")

labels = ["Seizure", "Epilepsy", "Sleep Disorder", "Tumor"]

# -------- PREDICTION FUNCTION --------
def predict_eeg(file):
    df = pd.read_csv(file.name, header=None)
    eeg = df.values.flatten()

    eeg = eeg.reshape(1, -1)
    prediction = model.predict(eeg)[0]

    prob_df = pd.DataFrame({
        "Condition": labels,
        "Probability": prediction
    })

    fig = px.bar(
        prob_df,
        x="Condition",
        y="Probability",
        title="Prediction Probabilities"
    )

    alert = "⚠️ HIGH RISK DETECTED" if max(prediction) > 0.7 else "✅ No Critical Risk"

    return fig, alert

# -------- GRADIO UI --------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 EEG Neurological Disorder Prediction")
    gr.Markdown("Seizure • Epilepsy • Sleep Disorder • Tumor")

    file_input = gr.File(label="Upload EEG CSV File")
    predict_btn = gr.Button("Predict")

    graph_output = gr.Plot(label="Prediction Graph")
    alert_output = gr.Textbox(label="Alert")

    predict_btn.click(
        fn=predict_eeg,
        inputs=file_input,
        outputs=[graph_output, alert_output]
    )

demo.launch()

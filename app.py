import gradio as gr
import joblib
import numpy as np
import os

# Load model
model = joblib.load("loan_model.pkl")

# Uncomment these if you used StandardScaler
# scaler = joblib.load("scaler.pkl")

def predict(
    dependents,
    education,
    self_employed,
    income,
    loan_amount,
    loan_term,
    cibil,
    residential_assets,
    commercial_assets,
    luxury_assets,
    bank_assets
):

    # Label Encoding
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0

    x = np.array([[

        dependents,
        education,
        self_employed,
        income,
        loan_amount,
        loan_term,
        cibil,
        residential_assets,
        commercial_assets,
        luxury_assets,
        bank_assets

    ]])

    # Uncomment if scaler was used
    # x = scaler.transform(x)

    prediction = model.predict(x)[0]

    if prediction == 1:
        return "✅ Loan Approved"
    else:
        return "❌ Loan Rejected"


demo = gr.Interface(

    fn=predict,

    inputs=[

        gr.Slider(0,5,step=1,label="Number of Dependents"),

        gr.Radio(
            ["Graduate","Not Graduate"],
            label="Education"
        ),

        gr.Radio(
            ["Yes","No"],
            label="Self Employed"
        ),

        gr.Slider(
            minimum=0,
            maximum=50000000,
            step=100000,
            label="Annual Income"
        ),

        gr.Slider(
            minimum=0,
            maximum=60000000,
            step=100000,
            label="Loan Amount"
        ),

        gr.Slider(
            minimum=2,
            maximum=20,
            step=1,
            label="Loan Term (Years)"
        ),

        gr.Slider(
            minimum=300,
            maximum=900,
            step=1,
            label="CIBIL Score"
        ),

        gr.Slider(
            minimum=0,
            maximum=30000000,
            step=100000,
            label="Residential Assets Value"
        ),

        gr.Slider(
            minimum=0,
            maximum=20000000,
            step=100000,
            label="Commercial Assets Value"
        ),

        gr.Slider(
            minimum=0,
            maximum=40000000,
            step=100000,
            label="Luxury Assets Value"
        ),

        gr.Slider(
            minimum=0,
            maximum=20000000,
            step=100000,
            label="Bank Assets Value"
        )

    ],

    outputs=gr.Textbox(label="Prediction"),

    title="🏦 Loan Approval Prediction",

    description="""
Predict whether a loan application will be approved using a Machine Learning model.
""",

    article="""
---
### 👩‍💻 Project Information

**Name:** Shreya Goel

**Department:** Computer Science & Engineering (CSE)

**College:** Panipat Institute of Engineering & Technology (PIET)

**LinkedIn:** https://www.linkedin.com/in/YOUR-LINKEDIN-USERNAME
""",

    theme=gr.themes.Soft()

)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT",7860))
    )

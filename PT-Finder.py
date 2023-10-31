import streamlit as st
import torch
import pickle
from transformers import BertTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from combined_model import CombinedModel  # Import your CombinedModel class
import numpy as np
import pandas as pd
import joblib
import altair as alt
from io import BytesIO
import base64

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set page title and favicon
st.set_page_config(page_title="PT-Finder", page_icon="🎯")

# Load the trained model
model_path = 'combined_model_epoch_20.pkl'
with open(model_path, 'rb') as model_file:
    trained_model = pickle.load(model_file)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_model.to(device)
trained_model.eval()

# Load the trained tokenizer
tokenizer = BertTokenizer.from_pretrained('trained_bert_tokenizer')

# Load the label mapping dictionary
with open('label_mapping.pkl', 'rb') as label_map_file:
    label_mapping = joblib.load(label_map_file)

# Define a function to calculate molecular properties
def calculate_molecular_properties(mol):
    morgan_fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    mol_logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    no_count = Descriptors.NOCount(mol)
    return morgan_fingerprint, mol_logp, tpsa, no_count

# Define a function to prepare input data for prediction
def prepare_input(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    input_ids = tokenizer(smi, truncation=True, padding='max_length', max_length=128, return_tensors='pt')['input_ids'].to(device)
    attention_mask = tokenizer(smi, truncation=True, padding='max_length', max_length=128, return_tensors='pt')['attention_mask'].to(device)

    morgan_fingerprint, mol_logp, tpsa, no_count = calculate_molecular_properties(mol)

    amino_acid_composition = torch.zeros(1, 20).to(device)
    sequence_motifs = torch.zeros(1, 2).to(device)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'morgan_fingerprint': torch.tensor(morgan_fingerprint, dtype=torch.float32).unsqueeze(0).to(device),
        'mol_logp': torch.tensor(mol_logp, dtype=torch.float32).unsqueeze(0).to(device),
        'tpsa': torch.tensor(tpsa, dtype=torch.float32).unsqueeze(0).to(device),
        'no_count': torch.tensor(no_count, dtype=torch.float32).unsqueeze(0).to(device),
        'amino_acid_composition': amino_acid_composition,
        'sequence_motifs': sequence_motifs
    }

# Add title at the top-left
st.title("PT-Finder")

# Add "Created by Hossam Nada" to the top-right
st.sidebar.markdown("Created by Hossam Nada")

# Add logo at the top-center, resized to half
st.image('logo.png', use_column_width=True, caption='Logo')

# Add file uploader for input CSV to the sidebar with a unique key
uploaded_file = st.sidebar.file_uploader("Upload CSV file with SMILES", type="csv", key="file_uploader")

# Define a function to display target names
def display_target_names():
    # Load the target names from "targets.csv"
    targets_df = pd.read_csv("targets.csv")
    target_names = targets_df["target name"].unique()
    
    # Display the target names
    st.sidebar.subheader("Target Names")
    st.sidebar.write(target_names)

# Main Streamlit app code
def main():
    # Display target names when the button is clicked
    if st.sidebar.button("Show Target Names"):
        display_target_names()

    # Check if the file is uploaded
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        smiles_list = data['smiles']
        name_list = data['name']  # Add the "Name" column

        # Prepare predictions and store in separate lists
        target_names_list = []
        target_probabilities_list = []

        for smiles in smiles_list:
            input_data = prepare_input(smiles)

            if input_data is not None:
                with torch.no_grad():
                    logits = trained_model(**input_data)
                    probabilities = torch.softmax(logits, dim=1)

                topk_probabilities, topk_indices = torch.topk(probabilities, k=5)
                target_names = []
                target_probs = []
                for i in range(5):
                    predicted_label_index = topk_indices[0][i].item()
                    predicted_probability = topk_probabilities[0][i].item()
                    predicted_target_name = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_label_index)]
                    target_names.append(predicted_target_name)
                    target_probs.append(predicted_probability)

                target_names_list.append(target_names)
                target_probabilities_list.append(target_probs)
            else:
                target_names_list.append([None] * 5)
                target_probabilities_list.append([np.nan] * 5)

        # Create a DataFrame from the separate lists
        results_df = pd.DataFrame({
            "Name": name_list,  # Add the "Name" column
            "Target 1 Name": [names[0] for names in target_names_list],
            "Target 1 Probability": [probs[0] for probs in target_probabilities_list],
            "Target 2 Name": [names[1] for names in target_names_list],
            "Target 2 Probability": [probs[1] for probs in target_probabilities_list],
            "Target 3 Name": [names[2] for names in target_names_list],
            "Target 3 Probability": [probs[2] for probs in target_probabilities_list],
            "Target 4 Name": [names[3] for names in target_names_list],
            "Target 4 Probability": [probs[3] for probs in target_probabilities_list],
            "Target 5 Name": [names[4] for names in target_names_list],
            "Target 5 Probability": [probs[4] for probs in target_probabilities_list]
        })

        # Export button to the sidebar
        export_button = st.sidebar.button("Export CSV")

        # Export button functionality
        if export_button:
            csv_data = results_df.to_csv(index=False).encode()
            b64 = base64.b64encode(csv_data).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download CSV</a>'
            st.sidebar.markdown(href, unsafe_allow_html=True)

        # Display results table in the middle
        st.subheader("Predicted Targets, Probabilities, and Molecules")
        st.dataframe(results_df, width=800)

        # Data Visualization
        st.subheader("Data Visualization")

        # Create a bar chart to visualize target probabilities
        chart_data = results_df.dropna(subset=["Target 1 Probability"])
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("Target 1 Name", sort="-y"),
            y=alt.Y("Target 1 Probability:Q"),
        ).properties(
            width=600,
            title="Top Predicted Targets and Probabilities"
        )

        st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()


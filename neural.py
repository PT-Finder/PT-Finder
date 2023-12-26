import re
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import joblib
import shap
from transformers import BertTokenizer, BertForSequenceClassification
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data cleaning function to remove invalid SMILES
def clean_data(data_df):
    valid_smiles = []
    invalid_smiles = []

    for smi in data_df['smiles']:
        if isinstance(smi, str):  # Check if the value is a string
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(smi)
            else:
                invalid_smiles.append(smi)
        else:
            invalid_smiles.append(smi)

    if len(invalid_smiles) > 0:
        with open('invalid_smiles.log', 'w') as f:
            for smi in invalid_smiles:
                f.write(f"Invalid SMILES: {smi}\n")
        print(f"Found {len(invalid_smiles)} invalid SMILES. See 'invalid_smiles.log' for details.")

    data_df = data_df[data_df['smiles'].isin(valid_smiles)]

    return data_df

# Load the labeled data and clean it
data_df = pd.read_csv('test.csv')
data_df = clean_data(data_df)

# Create train and test splits without stratification
train_data, test_data = train_test_split(data_df, test_size=0.2, random_state=42, stratify=None)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Regular expressions for sequence motifs
def check_sequence_motifs(sequence):
    motifs = ["PQITLWQR", "GIGGFIKVR"]
    motif_presence = [1 if re.search(motif, sequence) else 0 for motif in motifs]
    return motif_presence

# Calculate amino acid composition
def calculate_amino_acid_composition(sequence):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    composition = {aa: sequence.count(aa) / len(sequence) for aa in amino_acids}
    return composition

class CustomTextDataset(Dataset):
    def __init__(self, data, label_map, tokenizer, max_length=128, default_label='default'):
        self.data = data
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.default_label = default_label
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        smi = item['smiles']
        label = item['target name']
        sequence = item['sequence']
        
        # Handle invalid SMILES data
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None

        if pd.isna(label) or label not in self.label_map:
            label = self.default_label
        else:
            label = self.label_map[label]
        
        encoding = self.tokenizer(smi, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        morgan_fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        mol_logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        no_count = Descriptors.NOCount(mol)

        amino_acid_composition = calculate_amino_acid_composition(sequence)
        sequence_motifs = check_sequence_motifs(sequence)

        # Create 2-dimensional tensors with a single row
        amino_acid_composition_tensor = torch.tensor(list(amino_acid_composition.values()), dtype=torch.float32).unsqueeze(0)
        sequence_motifs_tensor = torch.tensor(sequence_motifs, dtype=torch.float32).unsqueeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label,
            'morgan_fingerprint': torch.tensor(morgan_fingerprint, dtype=torch.float32),
            'mol_logp': torch.tensor(mol_logp, dtype=torch.float32),
            'tpsa': torch.tensor(tpsa, dtype=torch.float32),
            'no_count': torch.tensor(no_count, dtype=torch.float32),
            'amino_acid_composition': amino_acid_composition_tensor,
            'sequence_motifs': sequence_motifs_tensor,
        }


# Create train and test splits
train_data, test_data = train_test_split(data_df, test_size=0.2, random_state=42)

# Create a common label mapping
common_label_map = {label: idx for idx, label in enumerate(set(data_df['target name']))}

try:
    # Save the common label mapping
    label_mapping_filename = 'label_mapping.pkl'
    with open(label_mapping_filename, 'wb') as label_map_file:
        joblib.dump(common_label_map, label_map_file)
    print(f"Common label mapping saved as '{label_mapping_filename}'.")
except Exception as e:
    print(f"Error saving common label mapping: {type(e).__name__} - {e}")
    
# Use the common label mapping for both training and validation sets
train_label_map = common_label_map

# Define labeled datasets using the CustomTextDataset class
train_dataset = CustomTextDataset(train_data, train_label_map, tokenizer, max_length=128)
test_dataset = CustomTextDataset(test_data, train_label_map, tokenizer, max_length=128)

# Create DataLoaders for the labeled datasets
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=20, pin_memory=True)

# Define the number of labels
num_labels = len(train_label_map)

# Calculate class weights based on class frequencies
class_counts = data_df['target name'].value_counts().sort_index()
total_samples = len(data_df)
class_weights = torch.tensor([total_samples / (class_counts[i] * len(class_counts)) for i in range(len(class_counts))], dtype=torch.float32).to(device)

# Define the model architecture to handle combined features
class CombinedModel(nn.Module):
    def __init__(self, num_labels):
        super(CombinedModel, self).__init__()
        self.mol_features_layer = nn.Sequential(
            nn.Linear(2048 + 1 + 1 + 1 + 20 + 2, 256),  # Wider layer
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 256),  # Deeper layer
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),  # Another deeper layer
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
        ).to(device)
        self.final_classifier = nn.Linear(128, num_labels).to(device)

    def forward(self, input_ids, attention_mask, morgan_fingerprint, mol_logp, tpsa, no_count, amino_acid_composition, sequence_motifs):
        amino_acid_composition_reshaped = amino_acid_composition.view(-1, amino_acid_composition.shape[-1])
        sequence_motifs_reshaped = sequence_motifs.view(-1, sequence_motifs.shape[-1])
        
        mol_features = self.mol_features_layer(torch.cat([morgan_fingerprint, mol_logp.unsqueeze(1), tpsa.unsqueeze(1), no_count.unsqueeze(1), amino_acid_composition_reshaped, sequence_motifs_reshaped], dim=1))
        logits = self.final_classifier(mol_features)
        return logits

# Initialize the combined model
combined_model = CombinedModel(num_labels).to(device)

# Define the loss function with weighted loss
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(combined_model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Initialize the GradScaler for mixed-precision training
scaler = GradScaler()

# Define the function to calculate permutation feature importance
def calculate_permutation_importance(model, data_loader, device):
    model.eval()
    importance_scores = {}

    with torch.no_grad():
        for batch in data_loader:
            # Move data to the device
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            morgan_fingerprint = batch['morgan_fingerprint'].to(device)
            mol_logp = batch['mol_logp'].to(device)
            tpsa = batch['tpsa'].to(device)
            no_count = batch['no_count'].to(device)
            amino_acid_composition = batch['amino_acid_composition'].to(device)
            sequence_motifs = batch['sequence_motifs'].to(device)
            labels = batch['label'].to(device)

            # Calculate the baseline accuracy
            baseline_predictions = model(inputs, attention_mask, morgan_fingerprint, mol_logp, tpsa, no_count, amino_acid_composition, sequence_motifs)
            baseline_accuracy = accuracy_score(labels.cpu().numpy(), baseline_predictions.argmax(axis=1).cpu().numpy())

            # Initialize importance scores
            scores = {col: 0.0 for col in range(inputs.size(1))}

            # Permute each feature and calculate accuracy
            for col in range(inputs.size(1)):
                permuted_inputs = inputs.clone()
                permuted_inputs[:, col] = torch.randperm(permuted_inputs.size(0))
                permuted_predictions = model(permuted_inputs, attention_mask, morgan_fingerprint, mol_logp, tpsa, no_count, amino_acid_composition, sequence_motifs)
                permuted_accuracy = accuracy_score(labels.cpu().numpy(), permuted_predictions.argmax(axis=1).cpu().numpy())
                scores[col] = baseline_accuracy - permuted_accuracy

            # Update importance scores
            for col, score in scores.items():
                if col in importance_scores:
                    importance_scores[col] += score
                else:
                    importance_scores[col] = score

    # Normalize importance scores
    num_samples = len(data_loader.dataset)
    for col, score in importance_scores.items():
        importance_scores[col] /= num_samples

    return importance_scores

# Define the number of epochs
num_epochs = 20

# Initialize TensorBoard writer
tensorboard_writer = SummaryWriter()

# Create an empty DataFrame for validation metrics
validation_metrics_df = pd.DataFrame(columns=['Epoch', 'Accuracy', 'Precision', 'Recall', 'F1-score'])

# Define early stopping parameters
patience = 2  # Number of consecutive epochs without improvement before stopping
best_accuracy = 0.0
best_epoch = 0
no_improvement_count = 0
best_model_state = None 

# Training loop
for epoch in range(num_epochs):
    combined_model.train()
    total_loss = 0.0
    y_true_train = []  # Initialize y_true_train
    y_pred_train = []  # Initialize y_pred_train

    all_training_labels = []  # Initialize all_training_labels

    # Training phase
    for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Training)", leave=False)):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        morgan_fingerprint = data['morgan_fingerprint'].to(device)
        mol_logp = data['mol_logp'].to(device)
        tpsa = data['tpsa'].to(device)
        no_count = data['no_count'].to(device)
        amino_acid_composition = data['amino_acid_composition'].to(device)
        sequence_motifs = data['sequence_motifs'].to(device)
        labels = data['label'].to(device)

        # Append the training labels to all_training_labels
        all_training_labels.extend(labels.cpu().numpy())

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = combined_model(input_ids, attention_mask, morgan_fingerprint, mol_logp, tpsa, no_count, amino_acid_composition, sequence_motifs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(torch.argmax(logits, dim=1).cpu().numpy())

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} (Training), Loss: {average_loss:.4f}")

    tensorboard_writer.add_scalar('Train/Loss', average_loss, epoch)

    # Calculate and print training evaluation metrics
    accuracy_train = accuracy_score(y_true_train, y_pred_train)
    precision_train = precision_score(y_true_train, y_pred_train, average='weighted')
    recall_train = recall_score(y_true_train, y_pred_train, average='weighted')
    f1_train = f1_score(y_true_train, y_pred_train, average='weighted')

    print(f"Training Accuracy: {accuracy_train * 100:.2f}%")
    print(f"Training Precision: {precision_train:.4f}")
    print(f"Training Recall: {recall_train:.4f}")
    print(f"Training F1-score: {f1_train:.4f}")

    tensorboard_writer.add_scalar('Train/Accuracy', accuracy_train, epoch)
    tensorboard_writer.add_scalar('Train/Precision', precision_train, epoch)
    tensorboard_writer.add_scalar('Train/Recall', recall_train, epoch)
    tensorboard_writer.add_scalar('Train/F1-score', f1_train, epoch)

    # Save the saved training labels to disk
    training_labels_filename = 'training_labels.npy'
    np.save(training_labels_filename, np.array(all_training_labels))
    print(f"Training labels saved as '{training_labels_filename}'.")

    # Define the path to the training labels file
    training_labels_filename = 'training_labels.npy'

    # Validation phase
    combined_model.eval()  # Set the model to evaluation mode
    y_true_val = []
    y_pred_val = []

    if len(test_loader) == 0:
        print("Validation dataset is empty.")
    else:
        print("Validation dataset size:", len(test_loader.dataset))  # Print the size of the validation dataset
        with torch.no_grad(), torch.cuda.amp.autocast():
            for data in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Validation)", leave=False):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                morgan_fingerprint = data['morgan_fingerprint'].to(device)
                mol_logp = data['mol_logp'].to(device)
                tpsa = data['tpsa'].to(device)
                no_count = data['no_count'].to(device)
                amino_acid_composition = data['amino_acid_composition'].to(device)
                sequence_motifs = data['sequence_motifs'].to(device)
                labels = data['label'].to(device)  # Keep labels in integer format

                logits = combined_model(input_ids, attention_mask, morgan_fingerprint, mol_logp, tpsa, no_count, amino_acid_composition, sequence_motifs)
                _, predicted = torch.max(logits, 1)

                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())

            # Calculate validation evaluation metrics
            accuracy_val = accuracy_score(y_true_val, y_pred_val)
            precision_val = precision_score(y_true_val, y_pred_val, average='weighted', zero_division=1)
            recall_val = recall_score(y_true_val, y_pred_val, average='weighted', zero_division=1)
            f1_val = f1_score(y_true_val, y_pred_val, average='weighted', zero_division=1)


            # Early stopping logic
            if accuracy_val > best_accuracy:
                best_accuracy = accuracy_val
                best_epoch = epoch
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(f"Early stopping: No improvement for {patience} consecutive epochs.")
                break

            # Add metrics to the validation metrics DataFrame
            validation_metrics_df = validation_metrics_df.append({
                'Epoch': epoch + 1,
                'Accuracy': accuracy_val,
                'Precision': precision_val,
                'Recall': recall_val,
                'F1-score': f1_val
            }, ignore_index=True)

            print(f"Validation Accuracy: {accuracy_val * 100:.2f}%")
            print(f"Validation Precision: {precision_val:.4f}")
            print(f"Validation Recall: {recall_val:.4f}")
            print(f"Validation F1-score: {f1_val:.4f}")

            tensorboard_writer.add_scalar('Validation/Accuracy', accuracy_val, epoch)
            tensorboard_writer.add_scalar('Validation/Precision', precision_val, epoch)
            tensorboard_writer.add_scalar('Validation/Recall', recall_val, epoch)
            tensorboard_writer.add_scalar('Validation/F1-score', f1_val, epoch)

            # Save validation results
            validation_results = pd.DataFrame({'True_Label': y_true_val, 'Predicted_Label': y_pred_val})
            validation_results.to_csv('validation_results.csv', index=False)

    # Calculate and save ROC curve data
    y_probs = []
    for data in test_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        morgan_fingerprint = data['morgan_fingerprint'].to(device)
        mol_logp = data['mol_logp'].to(device)
        tpsa = data['tpsa'].to(device)
        no_count = data['no_count'].to(device)
        amino_acid_composition = data['amino_acid_composition'].to(device)
        sequence_motifs = data['sequence_motifs'].to(device)
        labels = data['label'].to(device)

        with torch.no_grad():
            logits = combined_model(input_ids, attention_mask, morgan_fingerprint, mol_logp, tpsa, no_count, amino_acid_composition, sequence_motifs)
            probs = torch.softmax(logits, dim=1)
            y_probs.extend(probs.cpu().numpy())

    y_true_roc = np.array(y_true_val)
    y_probs_roc = np.array(y_probs)
    roc_data = {
        'y_true_roc': y_true_roc,
        'y_probs_roc': y_probs_roc,
    }
    roc_filename = f'roc_data_epoch_{epoch + 1}.pkl'
    with open(roc_filename, 'wb') as roc_file:
        pickle.dump(roc_data, roc_file)
    print(f"ROC curve data saved as '{roc_filename}'.")

    # Calculate and save permutation feature importance
    permutation_importance = calculate_permutation_importance(combined_model, test_loader, device)
    permutation_importance_filename = f'permutation_importance_epoch_{epoch + 1}.pkl'
    with open(permutation_importance_filename, 'wb') as pi_file:
        pickle.dump(permutation_importance, pi_file)
    print(f"Permutation feature importance saved as '{permutation_importance_filename}'.")

# Save the trained model using pickle
try:
    model_filename = f'combined_model_epoch_{num_epochs}.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(combined_model, model_file)
    print(f"Trained model saved as '{model_filename}'.")
except Exception as e:
    print(f"Error saving trained model: {type(e).__name__} - {e}")

# Save the trained tokenizer and label mapping using pickle
try:
    tokenizer.save_pretrained('trained_bert_tokenizer')
    print("Trained tokenizer saved.")
except Exception as e:
    print(f"Error saving trained tokenizer: {type(e).__name__} - {e}")

try:
    label_mapping_filename = 'label_mapping.pkl'
    with open(label_mapping_filename, 'wb') as label_map_file:
        pickle.dump(common_label_map, label_map_file)
    print(f"Common label mapping saved as '{label_mapping_filename}'.")
except Exception as e:
    print(f"Error saving common label mapping: {type(e).__name__} - {e}")


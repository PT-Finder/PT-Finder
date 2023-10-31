import torch
import torch.nn as nn
        
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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class RankingModel(pl.LightningModule):
    def __init__(self, embedding_dim=896, user_feature_dim=50, hidden_dim=256):
        super().__init__()
        
        self.product_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128)
        )
        
        self.user_encoder = nn.Sequential(
            nn.Linear(user_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        self.ranking_head = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, product_embedding, user_features):
        prod_encoded = self.product_encoder(product_embedding)
        user_encoded = self.user_encoder(user_features)
        
        combined = torch.cat([prod_encoded, user_encoded], dim=1)
        score = self.ranking_head(combined)
        
        return score
    
    def training_step(self, batch, batch_idx):
        product_emb, user_feat, label = batch
        pred = self(product_emb, user_feat)
        loss = nn.BCELoss()(pred, label.unsqueeze(1))
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class ProductInteractionDataset(Dataset):
    def __init__(self, df_interactions, df_embeddings):
        """
        df_interactions: user_id, product_id, label (1=click/purchase, 0=impression)
        df_embeddings: product_id, embedding, category, price, etc.
        """
        self.interactions = df_interactions.merge(
            df_embeddings, on='product_id', how='left'
        )
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        
        product_emb = torch.tensor(row['embedding'], dtype=torch.float32)
        user_features = torch.tensor([
            row['user_age'],
            row['user_gender_encoded'],
            row['avg_session_duration'],
            row['purchase_history_count'],
            # ... more user features
        ], dtype=torch.float32)
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        return product_emb, user_features, label

# Training
train_dataset = ProductInteractionDataset(df_train, df_embeddings)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

model = RankingModel()
trainer = pl.Trainer(max_epochs=10, accelerator='gpu')
trainer.fit(model, train_loader)

# Save model
torch.save(model.state_dict(), '/mnt/models/ranking_model.pth')
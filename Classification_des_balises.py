import pandas as pd
import os
import torch
import numpy as np
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List, Tuple
from pathlib import Path


class PersTecData(torch.utils.data.Dataset):
    """
    Dataset class for multi-label text classification.
    """
    
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, binarizer: MultiLabelBinarizer, max_length: int):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame containing text and labels
            tokenizer: HuggingFace tokenizer
            binarizer: MultiLabelBinarizer for label encoding
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.binarizer = binarizer
        self.labels = binarizer.transform(data['labels'])
        self.encoded_data = tokenizer(
            data['text'].astype(str).tolist(), 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.
        
        Args:
            index: Index of the item
            
        Returns:
            Tuple of (input_ids, attention_mask, labels)
        """
        input_ids = torch.tensor(self.encoded_data['input_ids'][index])
        attention_mask = torch.tensor(self.encoded_data['attention_mask'][index])
        labels = torch.tensor(self.labels[index]).float()
        return input_ids, attention_mask, labels

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)


class PLMClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for multi-label classification.
    """
    
    def __init__(self, model: AutoModelForSequenceClassification, learning_rate: float = 2e-5):
        """
        Initialize the classifier.
        
        Args:
            model: Pre-trained transformer model
            learning_rate: Learning rate for training
        """
        super(PLMClassifier, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Sigmoid activated logits
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return torch.sigmoid(outputs.logits)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Training batch
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        input_ids, attention_mask, labels = batch
        preds = self(input_ids, attention_mask)
        loss = self.criterion(preds, labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


class MultiLabelTextClassifier:
    """
    Main class for multi-label text classification training.
    """
    
    def __init__(
        self,
        raw_documents_dir: str = 'EN/raw-documents',
        annotations_file: str = 'EN/no_trans_ues_subtask-2-annotations.txt',
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        learning_rate: float = 2e-5,
        max_epochs: int = 30,
        threshold: float = 0.2,
        batch_size: int = 8,
        output_model_path: str = "trained_model_no_trans.pth"
    ):
        """
        Initialize the classifier.
        
        Args:
            raw_documents_dir: Directory containing text files
            annotations_file: Path to annotations file
            model_name: HuggingFace model name
            max_length: Maximum sequence length
            learning_rate: Learning rate for training
            max_epochs: Maximum training epochs
            threshold: Classification threshold (not used in training)
            batch_size: Training batch size
            output_model_path: Path to save the trained model
        """
        self.raw_documents_dir = raw_documents_dir
        self.annotations_file = annotations_file
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.threshold = threshold
        self.batch_size = batch_size
        self.output_model_path = output_model_path
        
        # Initialize components
        self.binarizer = MultiLabelBinarizer()
        self.tokenizer = None
        self.classification_model = None
        self.model = None
        self.trainer = None
        self.data_df = None
    
    def load_model_and_tokenizer(self) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
        """
        Load tokenizer and model.
        
        Returns:
            Tuple of (tokenizer, model)
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=len(self.binarizer.classes_)
        )
        return tokenizer, model
    
    def load_data(self) -> None:
        """Load and preprocess data."""
        print("Loading data...")
        
        # Read annotations file
        annotations_df = pd.read_csv(
            self.annotations_file, 
            sep='\t', 
            header=None, 
            names=['article_id', 'labels']
        )
        annotations_df['labels'] = annotations_df['labels'].apply(lambda x: x.split(';'))
        annotations_df['article_id'] = annotations_df['article_id'].str.replace('.txt', '', regex=False)

        # Fit binarizer
        self.binarizer.fit(annotations_df['labels'])

        # Read articles
        articles = []
        for filename in os.listdir(self.raw_documents_dir):
            if filename.endswith('.txt'):
                article_id = filename.split('.')[0]
                with open(os.path.join(self.raw_documents_dir, filename), 'r', encoding='utf-8') as file:
                    article_text = file.read()
                articles.append({'article_id': article_id, 'text': article_text})

        # Convert to DataFrame
        articles_df = pd.DataFrame(articles)

        # Merge articles and annotations
        self.data_df = pd.merge(articles_df, annotations_df, left_on='article_id', right_on='article_id', how='inner')
        
        print(f"Loaded {len(self.data_df)} articles")
    
    def prepare_model(self) -> None:
        """Prepare tokenizer and model."""
        print("Preparing model...")
        self.tokenizer, self.classification_model = self.load_model_and_tokenizer()
    
    def create_dataset(self) -> None:
        """Create dataset and data loader."""
        print("Creating dataset...")
        
        # Create dataset
        dataset = PersTecData(self.data_df, self.tokenizer, self.binarizer, self.max_length)

        # Use all data for training (as in original code)
        self.train_dataset = dataset

        # Create data loader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def setup_training(self) -> None:
        """Setup training components."""
        print("Setting up training...")
        
        # Initialize model
        self.model = PLMClassifier(model=self.classification_model, learning_rate=self.learning_rate)

        # Logger and Checkpoint
        logger = CSVLogger("lightning_logs", name="model_logs")
        checkpoint_callback = ModelCheckpoint(
            dirpath='.', 
            filename='best-checkpoint', 
            save_top_k=1, 
            monitor='train_loss', 
            mode='min'
        )

        # Trainer
        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs, 
            callbacks=[checkpoint_callback], 
            accelerator='cpu', 
            logger=logger
        )
    
    def train(self) -> None:
        """Train the model."""
        print("Starting training...")
        
        # Train model
        self.trainer.fit(self.model, self.train_loader)
        
        print("Training completed")
    
    def save_model(self) -> None:
        """Save the trained model."""
        print(f"Saving model to {self.output_model_path}")
        
        # Save trained model
        torch.save(self.model.model.state_dict(), self.output_model_path)
        
        print(f"Training complete. Model saved as {self.output_model_path}")
    
    def run_training_pipeline(self) -> None:
        """Run the complete training pipeline."""
        self.load_data()
        self.prepare_model()
        self.create_dataset()
        self.setup_training()
        self.train()
        self.save_model()


def main():
    """
    Main function to run the training.
    """
    # Initialize classifier with default parameters
    classifier = MultiLabelTextClassifier()
    
    # Run training pipeline
    classifier.run_training_pipeline()


if __name__ == "__main__":
    main()

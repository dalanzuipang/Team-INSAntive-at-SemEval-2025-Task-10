import os
import pandas as pd
import torch
import numpy as np
import pytorch_lightning as pl
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import yaml
import pickle

import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger


class MultiLabelTextDataset(torch.utils.data.Dataset):
    """
    Dataset class for multi-label text classification.
    
    Handles tokenization and label binarization for transformer models.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[List[str]],
        tokenizer: AutoTokenizer,
        binarizer: MultiLabelBinarizer,
        max_length: int = 128
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text strings
            labels: List of label lists for each text
            tokenizer: HuggingFace tokenizer
            binarizer: Fitted MultiLabelBinarizer
            max_length: Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.binarizer = binarizer
        self.max_length = max_length
        
        # Pre-compute tokenization and label binarization
        self.encoded_texts = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        self.binary_labels = torch.tensor(
            self.binarizer.transform(labels),
            dtype=torch.float32
        )
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (input_ids, attention_mask, labels)
        """
        return (
            self.encoded_texts['input_ids'][idx],
            self.encoded_texts['attention_mask'][idx],
            self.binary_labels[idx]
        )


class MultiLabelClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for multi-label text classification.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 10,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        threshold: float = 0.5,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize the classifier.
        
        Args:
            model_name: HuggingFace model name
            num_labels: Number of labels for classification
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            threshold: Threshold for binary classification
            class_names: List of class names for logging
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.threshold = threshold
        self.class_names = class_names or [f"label_{i}" for i in range(num_labels)]
        
        # Load pre-trained model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Model logits
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Batch of (input_ids, attention_mask, labels)
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        preds = torch.sigmoid(logits) > self.threshold
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Store outputs for epoch-end calculations
        self.training_step_outputs.append({
            'loss': loss.detach(),
            'preds': preds.detach().cpu(),
            'targets': labels.detach().cpu()
        })
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        
        Args:
            batch: Batch of (input_ids, attention_mask, labels)
            batch_idx: Batch index
            
        Returns:
            Validation loss
        """
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        preds = torch.sigmoid(logits) > self.threshold
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store outputs for epoch-end calculations
        self.validation_step_outputs.append({
            'loss': loss.detach(),
            'preds': preds.detach().cpu(),
            'targets': labels.detach().cpu()
        })
        
        return loss
    
    def on_training_epoch_end(self) -> None:
        """Calculate and log training metrics at epoch end."""
        if self.training_step_outputs:
            all_preds = torch.cat([x['preds'] for x in self.training_step_outputs])
            all_targets = torch.cat([x['targets'] for x in self.training_step_outputs])
            
            # Calculate metrics
            metrics = self._calculate_metrics(all_preds, all_targets, prefix='train')
            self.log_dict(metrics, on_epoch=True)
            
            # Clear outputs
            self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self) -> None:
        """Calculate and log validation metrics at epoch end."""
        if self.validation_step_outputs:
            all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
            all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
            
            # Calculate metrics
            metrics = self._calculate_metrics(all_preds, all_targets, prefix='val')
            self.log_dict(metrics, on_epoch=True)
            
            # Clear outputs
            self.validation_step_outputs.clear()
    
    def _calculate_metrics(self, preds: torch.Tensor, targets: torch.Tensor, prefix: str) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            preds: Predicted labels
            targets: True labels
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        preds_np = preds.numpy()
        targets_np = targets.numpy()
        
        metrics = {
            f'{prefix}_f1_micro': f1_score(targets_np, preds_np, average='micro'),
            f'{prefix}_f1_macro': f1_score(targets_np, preds_np, average='macro'),
            f'{prefix}_precision_micro': precision_score(targets_np, preds_np, average='micro'),
            f'{prefix}_precision_macro': precision_score(targets_np, preds_np, average='macro'),
            f'{prefix}_recall_micro': recall_score(targets_np, preds_np, average='micro'),
            f'{prefix}_recall_macro': recall_score(targets_np, preds_np, average='macro'),
        }
        
        return metrics
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Tuple of (probabilities, binary_predictions)
        """
        self.eval()
        with torch.no_grad():
            logits = self(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            preds = probs > self.threshold
        return probs, preds


class MultiLabelTextClassificationTrainer:
    """
    Main trainer class for multi-label text classification.
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
            **kwargs: Additional configuration parameters
        """
        # Load configuration
        self.config = self._load_config(config_path, **kwargs)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.tokenizer = None
        self.binarizer = None
        self.model = None
        self.trainer = None
        
        # Data storage
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _load_config(self, config_path: Optional[str], **kwargs) -> Dict[str, Any]:
        """Load configuration from file or kwargs."""
        default_config = {
            'data': {
                'raw_documents_dir': 'EN/raw-documents',
                'annotations_file': 'EN/no_trans_ues_subtask-2-annotations.txt',
                'max_length': 128,
                'train_split': 0.8,
                'val_split': 0.1
            },
            'model': {
                'model_name': 'bert-base-uncased',
                'learning_rate': 2e-5,
                'weight_decay': 0.01,
                'threshold': 0.5
            },
            'training': {
                'batch_size': 8,
                'max_epochs': 30,
                'accelerator': 'auto',
                'devices': 'auto',
                'precision': 16
            },
            'logging': {
                'log_dir': 'lightning_logs',
                'model_name': 'multilabel_classifier',
                'save_top_k': 1,
                'monitor': 'val_f1_macro',
                'mode': 'max'
            },
            'output': {
                'model_save_path': 'trained_model.pth',
                'binarizer_save_path': 'label_binarizer.pkl'
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            # Merge configurations
            default_config.update(file_config)
        
        # Override with kwargs
        for key, value in kwargs.items():
            if '.' in key:
                section, param = key.split('.', 1)
                if section in default_config:
                    default_config[section][param] = value
            else:
                default_config[key] = value
        
        return default_config
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> None:
        """Load and preprocess data."""
        self.logger.info("Loading data...")
        
        # Load annotations
        annotations_df = pd.read_csv(
            self.config['data']['annotations_file'],
            sep='\t',
            header=None,
            names=['article_id', 'labels']
        )
        annotations_df['labels'] = annotations_df['labels'].apply(lambda x: x.split(';'))
        annotations_df['article_id'] = annotations_df['article_id'].str.replace('.txt', '', regex=False)
        
        # Load articles
        articles = []
        raw_docs_dir = Path(self.config['data']['raw_documents_dir'])
        
        for filename in raw_docs_dir.glob('*.txt'):
            article_id = filename.stem
            try:
                with open(filename, 'r', encoding='utf-8') as file:
                    article_text = file.read().strip()
                    if article_text:  # Only add non-empty articles
                        articles.append({'article_id': article_id, 'text': article_text})
            except Exception as e:
                self.logger.warning(f"Error reading {filename}: {e}")
        
        # Create DataFrame and merge
        articles_df = pd.DataFrame(articles)
        data_df = pd.merge(articles_df, annotations_df, on='article_id', how='inner')
        
        self.logger.info(f"Loaded {len(data_df)} articles with {len(data_df['labels'].iloc[0])} unique labels")
        
        # Initialize binarizer
        self.binarizer = MultiLabelBinarizer()
        self.binarizer.fit(data_df['labels'])
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['model_name'],
            use_fast=True
        )
        
        # Create datasets
        self._create_datasets(data_df)
    
    def _create_datasets(self, data_df: pd.DataFrame) -> None:
        """Create train/val/test datasets."""
        texts = data_df['text'].tolist()
        labels = data_df['labels'].tolist()
        
        # Create full dataset
        full_dataset = MultiLabelTextDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            binarizer=self.binarizer,
            max_length=self.config['data']['max_length']
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(self.config['data']['train_split'] * total_size)
        val_size = int(self.config['data']['val_split'] * total_size)
        test_size = total_size - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        self.logger.info(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    def create_model(self) -> None:
        """Create the model."""
        self.logger.info("Creating model...")
        
        self.model = MultiLabelClassifier(
            model_name=self.config['model']['model_name'],
            num_labels=len(self.binarizer.classes_),
            learning_rate=self.config['model']['learning_rate'],
            weight_decay=self.config['model']['weight_decay'],
            threshold=self.config['model']['threshold'],
            class_names=list(self.binarizer.classes_)
        )
    
    def setup_trainer(self) -> None:
        """Setup PyTorch Lightning trainer."""
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config['logging']['log_dir'],
            filename='best-checkpoint',
            save_top_k=self.config['logging']['save_top_k'],
            monitor=self.config['logging']['monitor'],
            mode=self.config['logging']['mode']
        )
        
        early_stopping = EarlyStopping(
            monitor=self.config['logging']['monitor'],
            patience=5,
            mode=self.config['logging']['mode']
        )
        
        # Loggers
        csv_logger = CSVLogger(
            self.config['logging']['log_dir'],
            name=self.config['logging']['model_name']
        )
        
        tb_logger = TensorBoardLogger(
            self.config['logging']['log_dir'],
            name=self.config['logging']['model_name']
        )
        
        # Trainer
        self.trainer = pl.Trainer(
            max_epochs=self.config['training']['max_epochs'],
            accelerator=self.config['training']['accelerator'],
            devices=self.config['training']['devices'],
            precision=self.config['training']['precision'],
            callbacks=[checkpoint_callback, early_stopping],
            logger=[csv_logger, tb_logger],
            enable_progress_bar=True,
            enable_model_summary=True
        )
    
    def train(self) -> None:
        """Train the model."""
        self.logger.info("Starting training...")
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        ) if self.val_dataset and len(self.val_dataset) > 0 else None
        
        # Train
        self.trainer.fit(self.model, train_loader, val_loader)
        
        self.logger.info("Training completed!")
    
    def save_model(self) -> None:
        """Save the trained model and components."""
        self.logger.info("Saving model...")
        
        # Save model state dict
        torch.save(
            self.model.model.state_dict(),
            self.config['output']['model_save_path']
        )
        
        # Save binarizer
        with open(self.config['output']['binarizer_save_path'], 'wb') as f:
            pickle.dump(self.binarizer, f)
        
        # Save tokenizer
        tokenizer_dir = Path(self.config['output']['model_save_path']).parent / 'tokenizer'
        self.tokenizer.save_pretrained(tokenizer_dir)
        
        # Save configuration
        config_path = Path(self.config['output']['model_save_path']).parent / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        self.logger.info(f"Model saved to {self.config['output']['model_save_path']}")
    
    def run_training_pipeline(self) -> None:
        """Run the complete training pipeline."""
        try:
            self.load_data()
            self.create_model()
            self.setup_trainer()
            self.train()
            self.save_model()
            self.logger.info("Training pipeline completed successfully!")
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main function to run the training pipeline."""
    # Example configuration
    config = {
        'data.raw_documents_dir': 'EN/raw-documents',
        'data.annotations_file': 'EN/no_trans_ues_subtask-2-annotations.txt',
        'model.model_name': 'bert-base-uncased',
        'training.batch_size': 8,
        'training.max_epochs': 30,
        'output.model_save_path': 'trained_model_no_trans.pth'
    }
    
    # Create trainer
    trainer = MultiLabelTextClassificationTrainer(**config)
    
    # Run training pipeline
    trainer.run_training_pipeline()


if __name__ == "__main__":
    main()

import os
import torch
import numpy as np
import pickle
import gc
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset

class TextClassificationTrainer:
    """
    A multi-label text classification training system that trains separate models
    for each top-level label category.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_base_dir: str = "sub_label_models",
        max_length: int = 128,
        batch_size: int = 3,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        weight_decay: float = 0.01
    ):
        """
        Initialize the text classification trainer.
        
        Args:
            model_name: Pre-trained model name from HuggingFace
            output_base_dir: Directory to save trained models
            max_length: Maximum sequence length for tokenization
            batch_size: Training batch size
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            weight_decay: Weight decay for regularization
        """
        self.model_name = model_name
        self.output_base_dir = Path(output_base_dir)
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        
        # Setup device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create output directory
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """
        Tokenize text examples for model input.
        
        Args:
            examples: Dictionary containing text examples
            
        Returns:
            Tokenized examples
        """
        return self.tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length
        )
    
    def load_articles_and_labels(self, top_label_dir: Path) -> Tuple[List[str], List[str]]:
        """
        Load articles and their corresponding sub-labels from a top-level label directory.
        
        Args:
            top_label_dir: Path to the top-level label directory
            
        Returns:
            Tuple of (article_texts, article_sub_labels)
        """
        top_label = top_label_dir.name
        labels_file = top_label_dir / f"{top_label}_labels.txt"
        
        if not labels_file.exists():
            return [], []
        
        article_texts = []
        article_sub_labels = []
        
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                    
                article_id, sub_label_str = parts
                article_file = top_label_dir / f"{article_id}.txt"
                
                if not article_file.exists():
                    continue
                    
                try:
                    with open(article_file, "r", encoding="utf-8") as f_art:
                        text = f_art.read().strip()
                        if text:  # Only add non-empty texts
                            article_texts.append(text)
                            article_sub_labels.append(sub_label_str)
                except Exception as e:
                    print(f"Error reading article {article_file}: {e}")
                    continue
        
        return article_texts, article_sub_labels
    
    def prepare_dataset(self, texts: List[str], labels: List[str]) -> Dataset:
        """
        Prepare dataset for training.
        
        Args:
            texts: List of article texts
            labels: List of corresponding labels
            
        Returns:
            Prepared dataset ready for training
        """
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        # Create dataset
        dataset = Dataset.from_dict({
            "text": texts, 
            "labels": encoded_labels
        })
        
        # Apply preprocessing
        train_dataset = dataset.map(self.preprocess_function, batched=True)
        
        # Convert labels to proper tensor format
        def convert_labels_to_long(example):
            example["labels"] = torch.tensor(int(example["labels"]), dtype=torch.long)
            return example
        
        train_dataset = train_dataset.map(convert_labels_to_long)
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        return train_dataset, label_encoder
    
    def train_model_for_category(self, top_label_dir: Path) -> bool:
        """
        Train a classification model for a specific top-level category.
        
        Args:
            top_label_dir: Path to the top-level label directory
            
        Returns:
            True if training was successful, False otherwise
        """
        top_label = top_label_dir.name
        print(f"Processing top label: {top_label}")
        
        # Load data
        article_texts, article_sub_labels = self.load_articles_and_labels(top_label_dir)
        
        if len(article_texts) == 0:
            print(f"No articles found for top label {top_label}. Skipping.")
            return False
        
        print(f"Found {len(article_texts)} articles for {top_label}")
        
        # Prepare dataset
        train_dataset, label_encoder = self.prepare_dataset(article_texts, article_sub_labels)
        
        # Initialize model
        num_sub_labels = len(label_encoder.classes_)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_sub_labels,
            ignore_mismatched_sizes=True
        )
        model.to(self.device)
        
        # Setup training arguments
        safe_top_label = top_label.replace(":", "_").replace(" ", "_")
        output_dir = self.output_base_dir / f"{safe_top_label}_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            save_strategy="no",
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            logging_steps=10,
            logging_dir=str(output_dir / "logs"),
            report_to=None  # Disable wandb logging
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset
        )
        
        try:
            # Train model
            print(f"Training model for top label: {top_label}")
            trainer.train()
            
            # Save model weights
            print(f"Saving model weights for {top_label}")
            model_path = output_dir / "model_weights.pth"
            torch.save(model.state_dict(), model_path)
            
            # Save tokenizer and label encoder
            self.tokenizer.save_pretrained(output_dir)
            
            with open(output_dir / "label_encoder.pkl", "wb") as f:
                pickle.dump(label_encoder, f)
            
            # Save model configuration
            config = {
                "model_name": self.model_name,
                "num_labels": num_sub_labels,
                "max_length": self.max_length,
                "label_classes": label_encoder.classes_.tolist()
            }
            
            with open(output_dir / "config.pkl", "wb") as f:
                pickle.dump(config, f)
            
            print(f"Successfully trained and saved model for {top_label}")
            return True
            
        except Exception as e:
            print(f"Error training model for {top_label}: {e}")
            return False
            
        finally:
            # Cleanup memory
            del trainer, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def train_all_categories(self, classified_data_root: str = "classified_by_top_label") -> Dict[str, bool]:
        """
        Train models for all top-level categories.
        
        Args:
            classified_data_root: Root directory containing classified data
            
        Returns:
            Dictionary mapping category names to training success status
        """
        classified_root = Path(classified_data_root)
        
        if not classified_root.exists():
            raise FileNotFoundError(f"Classified data root directory not found: {classified_root}")
        
        results = {}
        
        # Process each top-level category
        for top_label_dir in classified_root.iterdir():
            if not top_label_dir.is_dir():
                continue
            
            success = self.train_model_for_category(top_label_dir)
            results[top_label_dir.name] = success
            
            print(f"Finished processing {top_label_dir.name}. Success: {success}\n")
        
        # Print summary
        successful = sum(results.values())
        total = len(results)
        print(f"Training completed. Successfully trained {successful}/{total} models.")
        
        return results


class TextClassificationPredictor:
    """
    Predictor class for making predictions with trained text classification models.
    """
    
    def __init__(self, model_dir: str):
        """
        Initialize the predictor with a trained model directory.
        
        Args:
            model_dir: Path to the trained model directory
        """
        self.model_dir = Path(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load configuration
        with open(self.model_dir / "config.pkl", "rb") as f:
            self.config = pickle.load(f)
        
        # Load label encoder
        with open(self.model_dir / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config["model_name"], 
            num_labels=self.config["num_labels"]
        )
        self.model.load_state_dict(
            torch.load(self.model_dir / "model_weights.pth", map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict the sub-label for a given text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (predicted_label, confidence_score)
        """
        # Tokenize input
        encoded_input = self.tokenizer(
            text, 
            truncation=True, 
            padding="max_length", 
            max_length=self.config["max_length"], 
            return_tensors="pt"
        ).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(**encoded_input).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            predicted_label_index = np.argmax(probs, axis=1)[0]
            confidence = np.max(probs, axis=1)[0]
        
        # Convert to label name
        predicted_label = self.label_encoder.inverse_transform([predicted_label_index])[0]
        
        return predicted_label, float(confidence)
    
    def predict_top_k(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Predict top-k sub-labels for a given text.
        
        Args:
            text: Input text to classify
            k: Number of top predictions to return
            
        Returns:
            List of (label, confidence) tuples sorted by confidence
        """
        # Tokenize input
        encoded_input = self.tokenizer(
            text, 
            truncation=True, 
            padding="max_length", 
            max_length=self.config["max_length"], 
            return_tensors="pt"
        ).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(**encoded_input).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Get top-k predictions
        top_k_indices = np.argsort(probs)[-k:][::-1]
        top_k_labels = self.label_encoder.inverse_transform(top_k_indices)
        top_k_probs = probs[top_k_indices]
        
        return list(zip(top_k_labels, top_k_probs.astype(float)))


def main():
    """
    Main function to demonstrate usage of the text classification system.
    """
    # Initialize trainer
    trainer = TextClassificationTrainer(
        model_name="bert-base-uncased",
        output_base_dir="sub_label_models",
        batch_size=4,
        num_epochs=3
    )
    
    # Train all categories
    try:
        results = trainer.train_all_categories("classified_by_top_label")
        print("Training Results:")
        for category, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {category}")
    except Exception as e:
        print(f"Training failed: {e}")


if __name__ == "__main__":
    main()
    

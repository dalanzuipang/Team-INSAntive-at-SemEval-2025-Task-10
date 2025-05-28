import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging


class TextDataOrganizer:
    """
    Organizes text files into hierarchical classification structure based on annotation files.
    
    This class processes annotation files containing top-level and sub-level labels,
    then organizes text files into corresponding directory structures for training
    hierarchical classification models.
    """
    
    def __init__(
        self, 
        annotation_file: str,
        text_dir: str,
        output_root: str = "classified_by_top_label",
        encoding: str = "utf-8"
    ):
        """
        Initialize the TextDataOrganizer.
        
        Args:
            annotation_file: Path to the annotation file (TSV format)
            text_dir: Directory containing the text files
            output_root: Root directory for organized output
            encoding: File encoding (default: utf-8)
        """
        self.annotation_file = Path(annotation_file)
        self.text_dir = Path(text_dir)
        self.output_root = Path(output_root)
        self.encoding = encoding
        
        # Create output directory
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Dictionary to store top label -> [(article_id, sub_labels)] mapping
        self.top_label_articles: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            'total_rows': 0,
            'valid_rows': 0,
            'files_processed': 0,
            'files_not_found': 0,
            'top_labels_created': 0
        }
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('text_organizer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_inputs(self) -> bool:
        """
        Validate input files and directories exist.
        
        Returns:
            True if all inputs are valid, False otherwise
        """
        if not self.annotation_file.exists():
            self.logger.error(f"Annotation file not found: {self.annotation_file}")
            return False
        
        if not self.text_dir.exists():
            self.logger.error(f"Text directory not found: {self.text_dir}")
            return False
        
        if not self.text_dir.is_dir():
            self.logger.error(f"Text path is not a directory: {self.text_dir}")
            return False
        
        self.logger.info("Input validation passed")
        return True
    
    def parse_labels(self, label_string: str) -> List[str]:
        """
        Parse semicolon-separated label string into a list.
        
        Args:
            label_string: String containing labels separated by semicolons
            
        Returns:
            List of cleaned label strings
        """
        if not label_string or not label_string.strip():
            return []
        
        return [label.strip() for label in label_string.split(";") if label.strip()]
    
    def read_text_file(self, file_path: Path) -> Optional[str]:
        """
        Read text content from file with error handling.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Text content or None if reading failed
        """
        try:
            with open(file_path, "r", encoding=self.encoding) as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def write_text_file(self, file_path: Path, content: str) -> bool:
        """
        Write text content to file with error handling.
        
        Args:
            file_path: Path where to write the file
            content: Text content to write
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "w", encoding=self.encoding) as f:
                f.write(content)
            return True
        except Exception as e:
            self.logger.error(f"Error writing file {file_path}: {e}")
            return False
    
    def filter_sub_labels(self, sub_labels: List[str], top_label: str) -> List[str]:
        """
        Filter sub-labels that belong to the given top-level label.
        
        Args:
            sub_labels: List of all sub-labels
            top_label: Top-level label to filter by
            
        Returns:
            List of filtered sub-labels
        """
        return [sub for sub in sub_labels if sub.startswith(top_label)]
    
    def process_annotation_file(self) -> bool:
        """
        Process the annotation file and organize text files.
        
        Returns:
            True if processing was successful, False otherwise
        """
        if not self.validate_inputs():
            return False
        
        self.logger.info(f"Processing annotation file: {self.annotation_file}")
        
        try:
            with open(self.annotation_file, "r", encoding=self.encoding) as f:
                reader = csv.reader(f, delimiter="\t")
                
                for row_num, row in enumerate(reader, 1):
                    self.stats['total_rows'] += 1
                    
                    # Skip invalid rows
                    if len(row) < 3:
                        self.logger.warning(f"Row {row_num}: Invalid format, skipping")
                        continue
                    
                    filename, top_label_str, sub_label_str = row[0], row[1], row[2]
                    
                    # Parse labels
                    top_labels = self.parse_labels(top_label_str)
                    sub_labels = self.parse_labels(sub_label_str)
                    
                    if not top_labels:
                        self.logger.warning(f"Row {row_num}: No valid top labels found")
                        continue
                    
                    # Check if text file exists
                    text_path = self.text_dir / filename
                    if not text_path.is_file():
                        self.logger.warning(f"Text file not found: {text_path}")
                        self.stats['files_not_found'] += 1
                        continue
                    
                    # Read article text
                    article_text = self.read_text_file(text_path)
                    if article_text is None:
                        continue
                    
                    # Extract article ID (filename without extension)
                    article_id = text_path.stem
                    
                    # Process each top label
                    self._process_article_for_labels(
                        article_id, article_text, top_labels, sub_labels
                    )
                    
                    self.stats['valid_rows'] += 1
                    self.stats['files_processed'] += 1
                    
                    if self.stats['files_processed'] % 100 == 0:
                        self.logger.info(f"Processed {self.stats['files_processed']} files...")
        
        except Exception as e:
            self.logger.error(f"Error processing annotation file: {e}")
            return False
        
        return True
    
    def _process_article_for_labels(
        self, 
        article_id: str, 
        article_text: str, 
        top_labels: List[str], 
        sub_labels: List[str]
    ) -> None:
        """
        Process a single article for all its top-level labels.
        
        Args:
            article_id: Unique identifier for the article
            article_text: Full text content of the article
            top_labels: List of top-level labels for this article
            sub_labels: List of sub-level labels for this article
        """
        for top_label in top_labels:
            # Create directory for this top label
            label_dir = self.output_root / top_label
            label_dir.mkdir(parents=True, exist_ok=True)
            
            # Write article file
            article_file = label_dir / f"{article_id}.txt"
            if not self.write_text_file(article_file, article_text):
                continue
            
            # Filter sub-labels for this top label
            filtered_sub_labels = self.filter_sub_labels(sub_labels, top_label)
            
            # Store mapping for later label file generation
            self.top_label_articles[top_label].append((article_id, filtered_sub_labels))
    
    def generate_label_files(self) -> bool:
        """
        Generate label files for each top-level category.
        
        Each label file contains:
        - One line per article
        - Tab-separated: article_id<TAB>sub_labels_separated_by_semicolon
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Generating label files...")
        
        try:
            for top_label, articles_info in self.top_label_articles.items():
                label_dir = self.output_root / top_label
                label_file = label_dir / f"{top_label}_labels.txt"
                
                with open(label_file, "w", encoding=self.encoding) as f:
                    for article_id, sub_labels in articles_info:
                        # Format: article_id<TAB>sub_labels_joined_by_semicolon
                        sub_labels_str = ";".join(sub_labels) if sub_labels else ""
                        line = f"{article_id}\t{sub_labels_str}\n"
                        f.write(line)
                
                self.stats['top_labels_created'] += 1
                self.logger.info(f"Created label file for '{top_label}' with {len(articles_info)} articles")
        
        except Exception as e:
            self.logger.error(f"Error generating label files: {e}")
            return False
        
        return True
    
    def organize_data(self) -> bool:
        """
        Main method to organize all data.
        
        Returns:
            True if organization was successful, False otherwise
        """
        self.logger.info("Starting data organization...")
        
        # Process annotation file and organize texts
        if not self.process_annotation_file():
            self.logger.error("Failed to process annotation file")
            return False
        
        # Generate label files
        if not self.generate_label_files():
            self.logger.error("Failed to generate label files")
            return False
        
        self._print_summary()
        self.logger.info("Data organization completed successfully!")
        return True
    
    def _print_summary(self) -> None:
        """Print summary statistics."""
        print("\n" + "="*50)
        print("DATA ORGANIZATION SUMMARY")
        print("="*50)
        print(f"Total rows processed: {self.stats['total_rows']}")
        print(f"Valid rows: {self.stats['valid_rows']}")
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files not found: {self.stats['files_not_found']}")
        print(f"Top-level categories created: {self.stats['top_labels_created']}")
        print(f"Output directory: {self.output_root}")
        
        if self.top_label_articles:
            print("\nTop-level categories:")
            for label, articles in self.top_label_articles.items():
                print(f"  - {label}: {len(articles)} articles")
        print("="*50)
    
    def get_statistics(self) -> Dict:
        """
        Get organization statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        return {
            **self.stats,
            'categories': {
                label: len(articles) 
                for label, articles in self.top_label_articles.items()
            }
        }


def main():
    """
    Main function demonstrating usage of TextDataOrganizer.
    """
    # Configuration
    config = {
        'annotation_file': "EN/folder2_tags.txt",
        'text_dir': "Folder2",
        'output_root': "classified_by_top_label"
    }
    
    # Create organizer
    organizer = TextDataOrganizer(**config)
    
    # Organize data
    success = organizer.organize_data()
    
    if success:
        # Get and display statistics
        stats = organizer.get_statistics()
        print(f"\nProcessing completed successfully!")
        print(f"Check '{config['output_root']}' for organized files.")
    else:
        print("Data organization failed. Check logs for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

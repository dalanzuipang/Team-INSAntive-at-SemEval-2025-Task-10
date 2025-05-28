import os
import openai
import shutil
from typing import List, Optional

"""
Text Translation Tool for Long Articles
Handles oversized documents by processing them in segments
"""


class TextTranslator:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the translator with OpenAI API key
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        openai.api_key = self.api_key
        
        # Default folder paths
        self.input_folder = "training_data_2/failed_files"
        self.output_folder = "training_data_2/EN/chatgpt_data"
        self.failed_folder = "training_data_2/failed_files"
        
        # Ensure output and failed folders exist
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.failed_folder, exist_ok=True)

    def set_folders(self, input_folder: str, output_folder: str, failed_folder: str):
        """
        Set custom folder paths
        
        Args:
            input_folder: Path to input files
            output_folder: Path to save translated files
            failed_folder: Path to save failed translation files
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.failed_folder = failed_folder
        
        # Ensure folders exist
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.failed_folder, exist_ok=True)

    def read_file(self, file_path: str) -> Optional[str]:
        """
        Read file content
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            File content as string or None if error occurred
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def split_text(self, text: str, max_length: int = 3000) -> List[str]:
        """
        Split long text into smaller chunks, each chunk not exceeding max_length characters
        
        Args:
            text: Text to split
            max_length: Maximum length per chunk
            
        Returns:
            List of text chunks
        """
        paragraphs = text.split("\n")
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 1 > max_length:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n" + paragraph if current_chunk else paragraph

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def translate_text_gpt(self, text: str, source_language: str = "Bulgarian", target_language: str = "English") -> Optional[str]:
        """
        Translate text using GPT
        
        Args:
            text: Text to translate
            source_language: Source language (default: Bulgarian)
            target_language: Target language (default: English)
            
        Returns:
            Translated text or None if translation failed
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional translator."},
                    {"role": "user", "content": f"Translate the following text from {source_language} to {target_language}:\n\n{text}"}
                ],
                temperature=0.1  # Lower temperature for more consistent translations
            )
            return response.choices[0].message.content.strip()
        except openai.error.AuthenticationError:
            print("Authentication Error: Invalid API Key.")
        except openai.error.RateLimitError:
            print("Rate Limit Exceeded. Please try again later.")
        except openai.error.OpenAIError as e:
            print(f"OpenAI API Error: {e}")
        except Exception as e:
            print(f"Unexpected error during translation: {e}")
        return None

    def translate_long_text(self, text: str, max_length: int = 3000) -> Optional[str]:
        """
        Translate long text by splitting it into chunks and combining the results
        
        Args:
            text: Text to translate
            max_length: Maximum length per chunk
            
        Returns:
            Complete translated text or None if translation failed
        """
        chunks = self.split_text(text, max_length)
        translated_chunks = []

        for i, chunk in enumerate(chunks):
            print(f"Translating chunk {i + 1}/{len(chunks)}...")
            translated_chunk = self.translate_text_gpt(chunk)
            if translated_chunk:
                translated_chunks.append(translated_chunk)
            else:
                print(f"Translation failed for chunk {i + 1}/{len(chunks)}.")
                return None

        return "\n\n".join(translated_chunks)

    def save_translation(self, file_path: str, translated_text: str) -> bool:
        """
        Save translation result to file
        
        Args:
            file_path: Path to save the file
            translated_text: Translated text content
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(translated_text)
            print(f"Translation saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving file {file_path}: {e}")
            return False

    def translate_all_files(self) -> List[str]:
        """
        Translate all files in the input folder and save to output folder
        Failed files are copied to the failed folder
        
        Returns:
            List of failed file names
        """
        if not os.path.exists(self.input_folder):
            print(f"Input folder {self.input_folder} does not exist.")
            return []
            
        files = os.listdir(self.input_folder)
        failed_files = []

        for file_name in files:
            input_path = os.path.join(self.input_folder, file_name)
            
            # Skip directories, only process files
            if not os.path.isfile(input_path):
                print(f"Skipping directory: {input_path}")
                continue

            output_path = os.path.join(self.output_folder, file_name)

            # Read file content
            text = self.read_file(input_path)
            if not text:
                print(f"Failed to read {input_path}. Skipping.")
                failed_files.append(file_name)
                try:
                    shutil.copy(input_path, os.path.join(self.failed_folder, file_name))
                except Exception as e:
                    print(f"Error copying failed file {file_name}: {e}")
                continue

            print(f"Translating file: {input_path}...")
            
            # Translate file content
            translated_text = self.translate_long_text(text)
            if not translated_text:
                print(f"Translation failed for {input_path}. Skipping.")
                failed_files.append(file_name)
                try:
                    shutil.copy(input_path, os.path.join(self.failed_folder, file_name))
                except Exception as e:
                    print(f"Error copying failed file {file_name}: {e}")
                continue

            # Save translation result
            if not self.save_translation(output_path, translated_text):
                failed_files.append(file_name)

        # Print and return failed files
        if failed_files:
            print(f"\nTranslation failed for {len(failed_files)} files:")
            for failed_file in failed_files:
                print(f"  - {failed_file}")
        else:
            print("\nAll files were successfully translated.")

        return failed_files

    def run(self):
        """
        Main method to run the translation process
        """
        print(f"Starting translation of files in {self.input_folder}...")
        failed_files = self.translate_all_files()
        print("Translation process completed.")
        return failed_files


def main():
    """
    Main function: Translate all files
    Usage: Set OPENAI_API_KEY environment variable before running
    """
    try:
        translator = TextTranslator()
        translator.run()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please set your OpenAI API key as an environment variable:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()

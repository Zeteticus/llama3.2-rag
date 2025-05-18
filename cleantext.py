import os
import re

def clean_text_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.endswith('.txt'):
            input_file = os.path.join(input_dir, file)
            output_file = os.path.join(output_dir, file)

            with open(input_file, 'r') as f:
                text = f.read()

            # Remove unwanted characters
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            # Convert to lowercase
            text = text.lower()

            with open(output_file, 'w') as f:
                f.write(text)

if __name__ == "__main__":
    input_dir = 'textfiles'
    output_dir = 'content'

    clean_text_files(input_dir, output_dir)

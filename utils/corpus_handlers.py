from bs4 import BeautifulSoup
import os
import numpy as np
import glob

def clean_html(input_file_path, output_file_path):
    # Read the file
    with open(input_file_path, 'r', encoding='utf8') as file:
        file_content = file.read()

    # Parse with BeautifulSoup and get text
    soup = BeautifulSoup(file_content, 'html.parser')
    cleaned_text = soup.get_text(separator='\n')  # Using '\n' as separator to keep text well-structured

    # Write the cleaned text to a new file
    with open(output_file_path, 'w', encoding='utf8') as file:
        file.write(cleaned_text)
    

def corpus_handlers(input_file_path, output_file_path):
    list_dir = os.listdir(input_file_path)
    list_corpus = []
    for item in list_dir:
        if item != ".DS_Store":  # Exclude .DS_Store file
            list_corpus.append(item)
    arr_corpus = np.asarray(list_corpus)
    np.savetxt("./data/label.txt", arr_corpus)
    source = os.path.join(input_file_path, file)
    destination = os.path.join(output_file_path, file)
    for file in arr_corpus:
        try:
        # Your code that might raise an error goes here
            clean_html(source, destination)
            break
        except Exception as e:
            print(f"file name: {input_file_path}:")  # Print the error message
        # You can also add additional handling code here if needed


def load_dataset(dataset_name, split_ratio):
    dataset = load_dataset(dataset_name, split="train[:5000]")
    X_train, Y_train, X_test, Y_test = dataset.train_test_split(test_size=split_ratio)
    return dataset, X_train, Y_train, X_test, Y_test


from .utils.corpus_handlers import corpus_handlers

def main():
    input_file_path = "./dataset/corpus"
    output_file_path = "./dataset/clean_corpus"
    corpus_handlers(input_file_path, output_file_path)

if __name__ == "__main__":
    main()
from bs4 import BeautifulSoup
import os
import numpy as np
from typing import List, Tuple
import re
from collections import defaultdict
import glob

eliminate_tags = ["iframe", "blockquote"]

class DocumentManager():

    def __init__(self):
        self.document_chunks = []
        self.next_id = 1
    
    def extract_metadata(self, chunk):
        doc = "".join(chunk)
        lines = doc.split("\n")
        diseaseName = lines[5].split(":")[0].split("?")[0]
        category = lines[4].split(":")[0].split("?")[0].split('>')[0]
        title = self.clean_text(lines[6])
        content = self.clean_text(''.join(lines[7:]))
        return diseaseName, category, title, content

    def extract_content(self, chunk):
        doc = "".join(chunk)
        lines = doc.split("\n")
        title_section = None
        for tag in ["</h2>", "</h3>", "</h1>"]:
            title_section = lines[0].split(tag)
            if len(title_section) > 1:
                break
        title = self.clean_text(title_section[0])
        content_lines = [title_section[1]] if len(title_section) > 1 else []
        content_lines.extend(lines[1:])
        content = self.clean_text(''.join(content_lines))
        return title, content
    
    def extract_last_content(self, chunk):
        title, content = self.extract_content(chunk)
        content = content.lower().split("hệ thống bệnh viện đa khoa tâm anh")[0]
        return title, content
    def extract_html(self, data):
        parse_content = defaultdict(dict)
        for index, element in enumerate(data.find_all("h2")):
            parse_content[str(index)]["label"] = element.text
            inner_text = []
            for elt in element.nextSiblingGenerator():
                if elt.name == "h2":
                    break
                if elt.name == "em":
                    if "Để đặt lịch thăm khám và điều trị các bệnh về gan với các chuyên gia bác sĩ về Tiêu hóa của Hệ thống Bệnh viện Đa khoa Tâm Anh, xin vui lòng liên hệ" in elt.text:
                        break
                if elt.name not in eliminate_tags:
                    if elt.name == "h3":
                        inner_text.append(elt.text + ":")
                        continue
                    if hasattr(elt, "text"):
                        inner_text.append(elt.text)
                else:
                    continue
        parse_content[str(index)]["text"] = re.sub(r'\s+', ' ', (' ').join(inner_text)).strip()
        return parse_content

    def process_chunk(self, doc_chunks):
        if not doc_chunks:
            raise ValueError("doc_chunks cannot be empty.")
        diseaseName, category, title, content = self.extract_metadata(doc_chunks[0])
        for chunk in doc_chunks[1:-1]:
            title, content = self.extract_content(chunk)
            self.add_chunk(diseaseName, category, title, content)
        title, content = self.extract_last_content(doc_chunks[-1])
        self.add_chunk(diseaseName, category, title, content)



class DocumentProcessor:
    def __init__(self, corpus_directory: str):
        self.corpus_directory = corpus_directory
        self.corpus = self.load_corpus()

    def load_corpus(self) -> List[str]:
        corpus = []
        for file_name in os.listdir(self.corpus_directory):
            if not file_name.startswith(('.DS_Store', '.ipynb_checkpoints')):
                file_path = os.path.join(self.corpus_directory, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc = f.readlines()
                corpus.append(doc)
        return corpus
    def size(self)-> str:
        return len(self.corpus)
    @staticmethod
    def get_soup(html_content: str) -> BeautifulSoup:
        return BeautifulSoup(html_content, 'html.parser')

    @staticmethod
    def extract_chunks(corpus_doc: List[str]) -> List[List[str]]:
        html_content = " ".join(corpus_doc)
        soup = DocumentProcessor.get_soup(html_content)
        headers = soup.find_all(['h2', 'h3'])

        chunks = []
        for i, header in enumerate(headers):
            if i == 0:
                chunk = corpus_doc[:header.sourceline - 1]
            else:
                prev_header = headers[i - 1]
                chunk = corpus_doc[prev_header.sourceline - 1:header.sourceline - 1]
            chunks.append(chunk)

        last_header = headers[-1]
        last_chunk = corpus_doc[last_header.sourceline - 1:]
        chunks.append(last_chunk)
        
        return chunks

    def process_document(self, doc_index: int) -> List[List[str]]:
        corpus_doc = self.corpus[doc_index]
        return self.extract_chunks(corpus_doc)

if __name__ == "__main__":
    processor = DocumentProcessor('./corpus')
    chunk_manager = DocumentManager()
   
    for i in range(processor.size()):
        try:
            chunks = processor.process_document(i)
            chunk_manager.process_chunks(chunks)
        except Exception as e:
            print(f"Error processing document {i}: {e}")

    chunk_manager.save_to_json("document_chunks.json")
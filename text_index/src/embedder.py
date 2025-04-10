from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import re
from bm25s.tokenization import Tokenizer
import math
from utils import date_preprocessing
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc
)
segmenter = Segmenter()
morph_vocab = MorphVocab()
news_embedding = NewsEmbedding()

class Embedder:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')
    
    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        
    def predict_batch(self, batch_texts: list[str]):
        inputs = self.tokenizer(
            batch_texts,
            max_length=512,
            padding=True,
            truncation=True, 
            return_tensors='pt'
        )

        self.model.eval()
        with torch.no_grad():
            output = self.model(**inputs)
        embeddings = self.average_pool(output.last_hidden_state, inputs['attention_mask'])
        embedding = torch.nn.functional.normalize(embeddings, p=2, dim=1).squeeze().detach().cpu()
        embedding = embedding.numpy()
        
        return embedding
    
    def predict(self, texts: list[str]):
        embeddings = []
        dataloader = DataLoader(texts, batch_size=16, num_workers=8, shuffle=False, pin_memory=True)
        for batch_texts in tqdm(dataloader):
            embeddings.extend(
                self.predict_batch(batch_texts)
            )
        embeddings = np.array(embeddings)

        return embeddings

def split_text(text: list[str], chunk_charlen_size: int=500):
    sentences = []
    text = remove_emoji(text)
    for sentence in re.split(r'(?<!\d)\.(?!\d)',text):
        sentences.append(
            " ".join([word for word in sentence.split(" ") if word])
        )

    starting_index = 0
    offset_index   = 0
    chunks = []
    while starting_index < len(sentences):
        offset_index += 1
        chunk = ". ".join(sentences[starting_index: starting_index + offset_index])
        if len(chunk) > chunk_charlen_size:
            chunks.append(chunk)
            starting_index += 1
            offset_index = 0

        if offset_index > len(sentences):
            break

    chunks.append(
        ". ".join(sentences[starting_index: starting_index + offset_index])
    )
    
    return chunks

def natasha_splitter(text: str, doc : Doc = None) -> list[str]:
    if doc is None:
        doc = Doc(text)
        doc.segment(segmenter)  # Сегментация на токены
        morph_tagger = NewsMorphTagger(news_embedding)
        doc.tag_morph(morph_tagger) 

    tokens = []
    for token in doc.tokens:
        # Пропускаем пунктуацию и символы
        if token.pos in ['PUNCT', 'SYM']:
            continue
        # Лемматизируем токен
        token.lemmatize(morph_vocab)
        tokens.append(token.lemma.lower())  # Приводим к нижнему регистру
    return tokens


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def sliding_word_chunks(
    text: str, 
    chunk_size: int = 10, 
    step: int = 5,
    tokenizer: callable = str.split
    ):

        words = tokenizer.tokenize([text], return_as='tuple')
        words = list(words.vocab.keys())
        if chunk_size > len(words):
            yield words
        else:
            for i in range(0, len(words) - chunk_size + 1, step):
                yield words[i:i + chunk_size]


def natasha_split(text):
    text = remove_emoji(text)
    doc = Doc(text)
    doc.segment(segmenter)  # Сегментация на токены
    morph_tagger = NewsMorphTagger(news_embedding)
    doc.tag_morph(morph_tagger) 
    tokens = natasha_splitter(text, doc)
    words_len = sum(list(map(len,tokens)))
    if(len(tokens) < 1):
        return []
    avg_word_len = words_len/len(tokens)
    s = 0
    for sent in doc.sents:
        s += len(sent.text) 
    avg_sent_len = s/len(doc.sents)
    chunk_size = (math.ceil(avg_sent_len / avg_word_len))
    step = math.ceil(avg_word_len)
    chunks_res = []
    tokenizer = Tokenizer(
                lower=False,  # Регистр уже обработан в splitter
                splitter=natasha_splitter,
                stopwords='ru',  # Используем русские стоп-слова
                stemmer=None  # Лемматизация уже выполнена Natasha
            )
    for i, chunk in enumerate(sliding_word_chunks(text, chunk_size=chunk_size * 5, step=step * 2, tokenizer = tokenizer)):
        chunks_res.append(" ".join(chunk))
    
    return chunks_res

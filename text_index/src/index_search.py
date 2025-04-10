from text_index.src.embedder import Embedder
from cb_ranker.src.cb_ranker import CatBoostRank
from logs.logger import main_logger
from utils import date_preprocessing
from db.chunks import get_chunks_by_ids
from db.messages import get_message_by_chunks
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from datetime import datetime, date
from usearch.index import Index
from sklearn.metrics.pairwise import cosine_similarity
from configuration.config import config_manager
from itertools import islice
from text_index.src.embedder import natasha_split, natasha_splitter
import bm25s
import statistics
from bm25s.tokenization import Tokenizer
import asyncio
import json
import requests

def take(n, iterable):
    return list(islice(iterable, n))

usearch_index_config = config_manager.get_config('usearch_index')
bm25_index_config = config_manager.get_config('bm25_index')
index_config = config_manager.get_config('index')


def get_bm25_ids_dict(savepath_index_to_db : str):
    dict_index_to_db = ''
    with open(savepath_index_to_db + "/index_to_db.json", 'r') as f:
        dict_index_to_db = json.load(f)
   
    return dict_index_to_db


class Candidate:
    def __init__(self, id, text, bm25_score, cosine_similarity, is_release, message_date, embedding, cross_score, from_index, message_link):
        self.id = id 
        self.text = text
        self.bm25_score = bm25_score
        self.cosine_similarity = cosine_similarity
        self.is_release = is_release
        self.message_date = message_date
        self.embedding = embedding
        self.from_index = from_index
        self.cross_score = cross_score
        self.is_tech_light = ""
        self.message_link = message_link
        self.reranker_score = 0


    def __str__(self):
        return  f"ID : {self.id},{self.text},Cross-Score: {self.cross_score},From: {self.from_index}"





class IndexQuery(object):


    @classmethod
    async def Create(cls, query_str : str, source : str, release_only : bool, topK : int):
        self = cls()
        self.query_loop = asyncio.get_event_loop()
        self.release_only = release_only
        self.query = query_str.lower()
        self.topK = topK
        self.source = source
        self.candidates = {}
        embedder = Embedder()
        self.embedding = embedder.predict([self.query])
        self.dates = self.retrieve_dates_from_query()
        self.cos_scores = {}
        self.bm25_scores = {}
        self.bm25_candidates = await self.calc_bm25_scores()
        self.bm25_scores_list =  [self.bm25_scores[k]['score'] for k in self.bm25_scores]
        self.usearch_candidates = await self.calc_usearch_scores()
        if len(self.dates):
            self.date_candidates = await self.get_candidates_by_dates()
            if not len(self.date_candidates):
                self.candidates = self.get_contextual_candidates()
            else:
                self.candidates = self.date_candidates
        else:
            self.candidates = self.get_contextual_candidates()

        return self


    def get_contextual_candidates(self):
        return (self.usearch_candidates + self.bm25_candidates)


    def hybrid_score(self, bm25: float, cosine: float, query: str) -> float:
        query_len = len(query.split())
        normalized_len = min(query_len / 15, 1.0)
        weight = 1.0 - 0.7 * normalized_len
        return weight * bm25 + (1 - weight) * cosine


    def add_candidate(self, candidate : Candidate):
        self.candidates[candidate.id] = candidate


    async def calc_bm25_scores(self):
        save_dir = bm25_index_config['index_file_path']
        file_name = bm25_index_config['index_file_name'].replace("#source#", self.source[-5:])
        if self.release_only:
            file_name += '_release'
        
        save_path = save_dir + file_name
        bm25_sample = bm25s.BM25()
        retriever = bm25s.BM25.load(save_path, load_corpus=True, mmap=True)
        bm25_sample.load_scores(save_dir = save_path)
        query_tokens = bm25s.tokenize([self.query])
        texts, scores = retriever.retrieve(query_tokens, k = len(retriever.corpus))
        texts = texts[0]
        scores = scores[0]
        index_to_db_dict = get_bm25_ids_dict(save_path)
        scores_dict = {}
        for i in range(len(texts)):
            scores_dict[str(index_to_db_dict[str(texts[i]['id'])])] = {"bm25_id": texts[i]['id'], "score" : scores[i]}

        self.bm25_scores = scores_dict
        chunk_ids = list(map(int,scores_dict.keys()))[:self.topK * 2]
        candidates = await self.get_candidates_by_chunk_ids(chunk_ids, "bm25")
            
        return candidates


    def prepare_candidates(self):
        median_cross_score = statistics.median([candidate.cross_score for candidate in self.candidates])
        candidates_actual = list(filter(lambda x : x.cross_score >= median_cross_score, self.candidates))
        self.candidates = candidates_actual
        if self.dates:
            self.candidates = sorted(self.candidates, key=lambda x : x.cross_score)
        else:
            self.candidates = sorted(self.candidates, key=lambda x : datetime.strptime(x.message_date, '%d.%m.%Y'), reverse=True)
        

    async def get_candidates_by_dates(self):
        if not self.dates:
            return []
        with open(index_config['dates_json_path'], 'r') as f:
            dates_json = json.load(f)
        chunk_ids = []
        for date in self.dates:
            if date in dates_json:
                chunk_ids = dates_json[date]
        return await self.get_candidates_by_chunk_ids(ids = chunk_ids, from_index ='dates')


    def retrieve_dates_from_query(self):
        return date_preprocessing.extract_dates(self.query)


    def calc_cos_scores(self, chunks_embeds : list[list[float]]):
        return cosine_similarity([self.embedding], chunks_embeds)[0]
        

    async def get_candidates_by_chunk_ids(self, ids, from_index):
        candidates = []
        message_ids = []
        existed_candidates_id = [candidate.id for candidate in self.candidates]
        if not len(ids):
            return []
        
        for id_ in ids:
            chunk = (await get_chunks_by_ids(chunks_ids = id_))

            if chunk['id'] in existed_candidates_id:
                continue

            message = await get_message_by_chunks(id_)
            if message['id'] in message_ids:
                continue
            else:
                message_ids.append(message['id'])

            message_link = message['message_link']
            date_obj = datetime.fromisoformat(str(message['timestamp']))
            formatted_date = date_obj.strftime("%d.%m.%Y")
            if from_index == 'usearch':
                cos_score = self.cos_scores[id_]
            else:
                cos_score = self.calc_cos_scores([chunk['embedding']])[0] 
            
            candidate = Candidate(
                id = chunk['id'], 
                text = chunk['text'], 
                is_release = not (chunk['release_date'] is None),
                message_date = formatted_date,
                bm25_score = self.bm25_scores[str(id_)]['score'],
                embedding  = chunk['embedding'],
                message_link = message_link,
                cosine_similarity = cos_score,
                from_index = from_index,
                cross_score = self.hybrid_score(self.bm25_scores[str(id_)]['score'], self.calc_cos_scores([chunk['embedding']])[0], self.query)
            )
            candidates.append(candidate)
            #self.add_candidate(candidate)
        return candidates


    async def calc_usearch_scores(self):
        
        ''' 
            Запрос к usearch-индексу.
            query - запрос
            topK - сколько чанков будет выдано в ответ
            source - если указано, будет поиск только по этому источнику
            release_only - поиск по только по релизам.
        '''
        
        save_dir = usearch_index_config['index_file_path']
        file_name = usearch_index_config['index_file_name'].replace("#source#", self.source[-5:])
        if self.release_only:
            file_name += '_release'
        save_path = save_dir + file_name
        index = Index(ndim = usearch_index_config['embeddings_dim'])
        index.load(save_path)
        closest_chunks = index.search(self.embedding, self.topK).to_list()
        chunks_ids = [int(match[0]) for match in closest_chunks]
        chunks_dist = [float(match[1]) for match in closest_chunks]
        self.cos_scores = dict(zip(chunks_ids, chunks_dist))
        return await self.get_candidates_by_chunk_ids(chunks_ids, 'usearch')

def cross_encoder_rank(query : IndexQuery):
    candidates = query.candidates
    payload = {
        "query" : query.query, 
        "texts" : [candidate.text for candidate in candidates]
    }
    ranker_result = json.loads(requests.post("http://192.168.1.73:8000/rank", json = payload).json())
    for i in ranker_result['index']:
        candidates[i].reranker_score = ranker_result['scores'][i]
    candidates =  sorted(candidates[:15], key= lambda x: datetime.strptime(x.message_date, '%d.%m.%Y'), reverse = True)
    print(*[(candidates[i].reranker_score, candidates[i].message_date) for i in range(len(candidates))])
    return sorted(candidates, key= lambda x: x.reranker_score, reverse = True)

# def prepare_cbranker_data(query, total_res):
#     embedder = Embedder()
#     query_embed = embedder.predict([query]).tolist()
#     cb_output = []
#     query_len = len(query)
#     for index, result in enumerate(total_res):
#         cb_row = {}
#         chunk_len = len(result['text'])
#         cb_row['cross_score'] = result['cross_score']
#         cb_date = result['release_date']
#         if not cb_date:
#             cb_date = "21.03.2025"
#         cand_date = datetime.strptime(cb_date, "%d.%m.%Y").date()
#         days_difference = abs(date.today() - cand_date)
#         days_left = days_difference.days
#         cb_row['days_left'] = days_left
#         cb_row['is_release'] = result['is_release']
#         cb_row['is_tech_light'] = "техническая молния" in result['text'].lower()
#         cb_row['query_rank'] = 1
#         cb_output.append(cb_row)
#     return cb_output

async def index_search(query: str, topK: int, source : str, release_only : bool) -> dict[list[dict]]:
    '''
        Функция поиска по индексу двумя способами: usearch и bm25s
    '''
    index_query = await IndexQuery.Create(query_str = query, source = source, release_only = release_only, topK = topK)
    index_query.prepare_candidates()
    index_query.candidates = cross_encoder_rank(index_query)
    return index_query.candidates









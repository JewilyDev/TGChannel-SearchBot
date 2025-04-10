from text_index.src.embedder import Embedder, split_text, natasha_split
from cb_ranker.src.cb_ranker import CatBoostRank
from utils.json_dates import create_json_dates
from logs.logger import main_logger
from db.messages import get_messages_by_chat, get_all_messages
from db.chunks import insert_embeddings, write_chunks, get_timestamps_by_chunk_id, get_chunks_by_chat, get_all_chunks
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from datetime import datetime
from usearch.index import Index
from sklearn.cluster import DBSCAN
from configuration.config import config_manager
import bm25s
import os
import json
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
usearch_index_config = config_manager.get_config('usearch_index')
bm25_index_config = config_manager.get_config('bm25_index')
index_config = config_manager.get_config('index')


async def create_embeddings(
        recreate_embeddings: bool, 
        chunks: list[dict], 
        chunks_ids: list[int] | None) -> tuple[list[float], int]:
    '''
    Создание / вытаскивание эмбеддингов, 
    recreate_embeddings = True -> создаем эмбеддинги для всех чанков в БД,
    recreate_embeddings = False -> вытаскиваем существующие эмбеддинги чанков из БД
    '''
    if recreate_embeddings:
        chunks = [chunk['text'].lower() for chunk in chunks]
        embedder = Embedder()
        embeddings = embedder.predict(chunks)
        embeddings_ids = [(embeddings[i], chunks_ids[i]) for i in range(len(chunks_ids))]
        await insert_embeddings(embeddings_ids)

    else:
        embeddings = [chunk['embedding'] for chunk in chunks if chunk['embedding']]
        chunks_ids = [chunk['id'] for chunk in chunks]

    return embeddings, chunks_ids


async def clusterize_embeddings(
        embeddings: list[list[float]], 
        chunks_ids: list[int]) -> tuple[int, list[float]]:
    '''
    Кластеризация эмбеддингов и выбор наиболее репрезентативных
    В каждом кластере берется документ первый по новизне, остальные отбрасываются
    '''
    dbscan = DBSCAN(eps=0.05, min_samples=1)    
    clusters = dbscan.fit_predict(embeddings)

    d = {}
    # возможно, здесь переделать
    for i, cluster in enumerate(clusters): # сопоставляем кластеры и id chunk'ов в виде d = {cluster: [id_1, ..., id_n]}
        if cluster in d:
            d[cluster].append(chunks_ids[i])
        else:
            d[cluster] = [chunks_ids[i]]

    unique_clusters = set(clusters) - {-1} 

    repr_ids, repr_embeddings = [], []
    for cluster in unique_clusters:
        cluster_chunk_ids = d[cluster]
        timestamps = await get_timestamps_by_chunk_id(chunks_ids=cluster_chunk_ids)
        max_timestamp_chunk_id = max(timestamps, key=lambda t: t['timestamp'])['id']

        repr_ids.append(max_timestamp_chunk_id)
        repr_embeddings.append(embeddings[chunks_ids.index(max_timestamp_chunk_id)])
    
    return np.array(repr_ids), np.array(repr_embeddings) 


async def create_bm25_index(chunks: list[dict], specified_source: str = '', release_only : bool = False) -> None:
    '''
        Просто фитим BM25 под наши текстики из БД
    '''
    corpus = [chunk['text'].lower() for chunk in chunks]
    chunks_ids = [chunk['id'] for chunk in chunks]
    corpus_tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25(method = "atire",corpus=corpus)
    retriever.index(corpus_tokens)
    save_dir = bm25_index_config['index_file_path']
    file_name = bm25_index_config['index_file_name'].replace("#source#", specified_source[-5:])
    
    if release_only:
            file_name += '_release'
    save_path = save_dir + file_name
    retriever.save(save_path)
    build_bm25_ids_json(save_path, chunks_ids)

    main_logger.info(f"BM25 index был построен из источника {specified_source}, только релизы: {release_only}")


async def create_search_indicies_by_sources(
        create_total_index: bool = True, 
        create_bm25: bool = True, 
        create_usearch: bool = True, 
        recalc_chunks: bool = True,
        recreate_embeddings: bool = False,  
        sources: list[str] = [],
        release_only :bool = False) -> None:
    '''
    create_total_index : Будет ли создан "общий" индекс, в котором будет вся информация из БД
    create_bm25 : создавать ли bm25 индекс
    create_usearch : создавать ли usearch индекс
    recreate_embeddings : пересчитывать ли эмбеддинги или подтянуть их из БД
    sources - все источники для которых требуется посроить специфические индексы
    release_only - строить ли индексы только по каналам или нет
    '''    
    if create_total_index:
        await create_index(create_bm25, create_usearch, recalc_chunks, recreate_embeddings, release_only = release_only)
        recreate_embeddings = False
        recalc_chunks = False

    for source in sources:
        await create_index(create_bm25, create_usearch, recalc_chunks, recreate_embeddings, source, release_only)


async def create_index(
        create_bm25: bool = True, 
        create_usearch: bool = True, 
        recalc_chunks : bool = True,
        recreate_embeddings: bool = False, 
        specified_source: str = '',
        release_only : bool = False) -> None:
    '''
    Создание индексов для чанков, лежащих в БД,
    recreate_embeddings = True -> создание эмбеддингов + индексов,
    recreate_embeddings = False -> создание только индексов, эмбеддинги уже есть в БД
    '''
    messages, chunks, chunks_ids = [], [], []

    if len(specified_source):
        messages = await get_messages_by_chat(id_chat = specified_source)
    else:
        messages = await get_all_messages()

    if release_only:
        messages = [message for message in messages if message['release_date'] is not None]

    if recalc_chunks:
        chunks_and_id = []
        chunks_raw = []
        releases = []
        message_dates = []
        for msg in messages:
            local_chunks = split_text(msg['text'])
            date_obj = datetime.fromisoformat(str(msg['timestamp']))
            formatted_date = date_obj.strftime("%d.%m.%Y")
            for ind in range(len(local_chunks)):
                text_to_db = local_chunks[ind]
                releases.append(msg['release_date'])
                message_dates.append(formatted_date)
                chunks_and_id.append((msg['id'], text_to_db, formatted_date, msg['release_date']))
                chunks_raw.append(text_to_db)

        chunks_ids = await write_chunks(chunks_and_id)
        chunks = [{'message_date' : message_dates[i], 'release_date' : releases[i],'text' : chunks_raw[i], 'id' : chunks_ids[i]} for i in range(len(chunks_ids))]
    else:
        if len(specified_source):
            chunks = await get_chunks_by_chat(id_chat = specified_source)
        else:
            chunks = await get_all_chunks()

        if release_only:
            chunks = [chunk for chunk in chunks if chunk['release_date'] is not None]

        chunks_ids = [chunk['id'] for chunk in chunks]

    if not len(specified_source) and create_bm25:
        create_json_dates(chunks, index_config['index_files_path'])



    if create_usearch:
        embeddings, chunks_ids = await create_embeddings(recreate_embeddings, chunks, chunks_ids)
        representative_indicies, representative_embeddings = await clusterize_embeddings(embeddings, chunks_ids)
        
        index = Index(ndim = usearch_index_config['embeddings_dim'])
        index.add(representative_indicies, representative_embeddings)
        save_dir = usearch_index_config['index_file_path']
        file_name = usearch_index_config['index_file_name'].replace("#source#", specified_source[-5:])
        if release_only:
            file_name += '_release'
        save_path = save_dir + file_name
        index.save(save_path)
        main_logger.info(f"Usearch index был построен из источника {specified_source}, только релизы: {release_only}")

    if create_bm25:
        await create_bm25_index(chunks, specified_source, release_only)
   


def build_bm25_ids_json(savepath_index_to_db : str, ids : list[str]):
    dict_index_to_db = {k : v for (k, v) in zip(list(range(len(ids))), ids)}
    with open(savepath_index_to_db + "/index_to_db.json",'w') as f:
        json.dump(obj = dict_index_to_db,fp = f )


def is_exists_bm25_index(specified_source : str, release_only : bool):
    save_dir = bm25_index_config['index_file_path']
    file_name = bm25_index_config['index_file_name'].replace("#source#", specified_source[-5:])
    if release_only:
            file_name += '_release'
    save_path = save_dir + file_name
    return os.path.exists(save_path)
    

def is_exists_usearch_index(specified_source : str, release_only : bool):
    save_dir = usearch_index_config['index_file_path']
    file_name = usearch_index_config['index_file_name'].replace("#source#", specified_source[-5:])
    if release_only:
        file_name += '_release'
    save_path = save_dir + file_name
    return os.path.exists(save_path)


async def update_usearch_index(chunks : list[dict], specified_source : str, release_only : bool):

    if release_only:
        chunks = [chunk for chunk in chunks if chunk['release_date']]

    chunks_ids = [chunk['id'] for chunk in chunks]
    embeddings, chunks_ids = await create_embeddings(True, chunks, chunks_ids)
    representative_indicies, representative_embeddings = await clusterize_embeddings(embeddings, chunks_ids)
    index = Index(ndim = usearch_index_config['embeddings_dim'])
    save_dir = usearch_index_config['index_file_path']
    file_name = usearch_index_config['index_file_name'].replace("#source#", specified_source[-5:])
    if release_only:
        file_name += '_release'

    save_path = save_dir + file_name
    index.load(save_path)
    index.add(representative_indicies, representative_embeddings)
    index.save(save_path)
    main_logger.info(f"Usearch index был обновлен из источника {specified_source}, только релизы: {release_only}")


async def update_bm25_index(specified_source : str, release_only : bool):

    if len(specified_source):
        chunks = await get_chunks_by_chat(id_chat = specified_source)
    else:
        chunks = await get_all_chunks()

    if release_only:
        chunks = [chunk for chunk in chunks if chunk['release_date'] is not None]

    await create_bm25_index(chunks, specified_source, release_only)

# async def insert_to_index(message_id: int, message_text : str) -> None:
#     '''
#         Вот эта штука под вопросом вообще пока, как будто в SIH не бывает дублей.
#         Поэтому возможно нужно будет делать репарсинг каждые n минут.
#     '''
#     index = Index(ndim = usearch_index_config['embeddings_dim'])
#     index.load(usearch_index_config['index_file_path'])
#     chunks = split_text(message_text)
    
#     embedder = Embedder()
#     chunk_embeddings = embedder.predict(chunks)
#     similar_vectors = index.search(chunk_embeddings, 10).to_list()
    
#     eps = 0.01
#     insert_usearch = True
#     for match in similar_vectors:
#         if float(match[1]) < eps:
#             insert_usearch = False
#             break

#     if insert_usearch:
#         chunks_to_db = [(message_id, chunk) for chunk in chunks]
#         chunks_ids = await write_chunks(chunks_to_db)
#         embeddings_ids = [(chunk_embeddings[i], chunks_ids[i]) for i in range(len(chunks_ids))]

#         await insert_embeddings(embeddings_ids)
#         index.add(np.array(chunks_ids), np.array(chunk_embeddings))

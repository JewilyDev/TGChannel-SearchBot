import json
from omegaconf import OmegaConf

config = OmegaConf.load("cb_ranker_config.yaml")
    
ranker = CatBoostRank(
    config=config,
    text_columns=['query_embedding', 'chunk_text_embedding'],
    group_id_column='query_embedding',
    target_column='feedback'
)

train_df = pd.read_parquet('train_data_exp.pq')
val_df = pd.read_parquet('val_data_exp.pq')

ranker.fit(train_df, eval_data=val_df)

test_df = pd.read_parquet('test_data.pq')
predictions = ranker.predict(test_df)

val_metrics = ranker.evaluate(val_df)
print(val_metrics)

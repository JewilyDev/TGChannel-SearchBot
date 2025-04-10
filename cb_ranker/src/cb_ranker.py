from catboost import CatBoostRanker, Pool
import pandas as pd
import numpy as np

class CatBoostRank:
    def __init__(
        self,
        config: str,
        cat_features: list[str],
        group_id_column: str,
        target_column: str
    ):
        self.config = config
        self.cat_features = cat_features
        self.group_id_column = group_id_column
        self.target_column = target_column
        self.model = CatBoostRanker(
            loss_function=config.loss_function,
            iterations=config.iterations,
            learning_rate=config.learning_rate,
            depth=config.depth,
            cat_features=self.cat_features,
            # verbose=config.verbose,
            task_type='GPU' if config.use_gpu else 'CPU'
        )
        
    def _prepare_pool(
        self,
        data: pd.DataFrame
    ) -> Pool:
        return Pool(
            data=data.drop([self.group_id_column, self.target_column], axis=1),
            label=data[self.target_column],
            group_id=data[self.group_id_column],
            cat_features=self.cat_features
        )
    
    def fit(
        self,
        train_data: pd.DataFrame,
        eval_data: pd.DataFrame = None
    ) -> None:
        train_pool = self._prepare_pool(train_data)
        
        if eval_data is not None:
            eval_pool = self._prepare_pool(eval_data)
            self.model.fit(train_pool, eval_set=eval_pool, plot=True)
        else:
            self.model.fit(train_pool, plot=True)
    
    def predict(
        self,
        data: pd.DataFrame
    ) -> np.ndarray:
        pool = Pool(
            data=data.drop([self.group_id_column], axis=1),
            cat_features=self.cat_features
        )
        return self.model.predict(pool)
    
    def evaluate(
        self,
        data: pd.DataFrame,
        metrics: list[str] = ['NDCG', 'MAP']
    ) -> dict[str, float]:
        pool = self._prepare_pool(data)
        return self.model.eval_metrics(pool, metrics=metrics)

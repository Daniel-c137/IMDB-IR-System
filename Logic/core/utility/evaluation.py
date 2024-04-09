
from typing import List
from numpy import log2
from wandb import log

class Evaluation:

    def __init__(self, name: str):
            self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        precision = 0.0

        tp = 0
        fp = 0
        for i, ranking in enumerate(predicted):
            actual_ranking = actual[i]
            for movie in ranking:
                if movie in actual_ranking:
                    tp += 1
                else:
                    fp += 1
        precision = tp / (tp + fp)
        
        return precision
    
    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        recall = 0.0

        tp = 0
        fn = 0
        for i, ranking in enumerate(actual):
            pred_ranking = predicted[i]
            for movie in ranking:
                if movie in pred_ranking:
                    tp += 1
                else:
                    fn += 1
        precision = tp / (tp + fn)

        return recall
    
    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        f1 = 0.0

        p = self.calculate_precision(actual, predicted)
        r = self.calculate_recall(actual, predicted)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        return f1
    
    def calculate_AP(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Average Precision of the predicted results

        Parameters
        ----------
        actual : List[str]
            The actual result
        predicted : List[str]
            The predicted result

        Returns
        -------
        float
            The Average Precision of the predicted result
        """
        AP = 0.0

        cnt = 0
        for j, movie in enumerate(predicted):
            if movie in actual:
                cnt += 1
                AP += self.calculate_precision([predicted[:(j + 1)]], [actual])
        if cnt == 0:
            return 0
        return AP / cnt
    
    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        MAP = 0.0

        for i, ranking in enumerate(predicted):
            MAP += self.calculate_AP(actual[i], ranking)

        return MAP / len(predicted)
    
    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        DCG = 0.0

        for i, ranking in enumerate(predicted):
            actual_ranking = actual[i]
            rank_gain = 0
            for j, movie in enumerate(ranking):
                if movie in actual_ranking:
                    rate = len(actual_ranking) - actual.index(movie)
                    if j == 0:
                        rank_gain += rate
                    else:
                        rank_gain += rate / log2(j + 1)
            DCG += rank_gain

        return DCG
    
    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        NDCG = 0.0

        for i, ranking in enumerate(predicted):
            actual_ranking = actual[i]
            rank_gain = 0
            nf = 0
            for j, movie in enumerate(ranking):
                opt_rate = len(actual_ranking) - j
                if j == 0:
                    nf += opt_rate
                else:
                    nf += opt_rate / log2(j + 1)
                if movie in actual_ranking:
                    rate = len(actual_ranking) - actual.index(movie)
                    if j == 0:
                        rank_gain += rate
                    else:
                        rank_gain += (rate / log2(j + 1))        
            NDCG += rank_gain / nf

        return NDCG
    
    def cacluate_RR(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Reciprocal Rank of the predicted result

        Parameters
        ----------
        actual : List[str]
            The actual result
        predicted : List[str]
            The predicted result

        Returns
        -------
        float
            The Reciprocal Rank of the predicted result
        """
        RR = 0.0

        for i, movie in enumerate(predicted):
            if movie in actual:
                k = actual.index(movie) + 1
                RR += 1 / k

        return RR
    
    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        MRR = 0.0

        for i, ranking in enumerate(predicted):
            MRR += self.cacluate_RR(actual[i], ranking)
        MRR /= len(predicted)

        return MRR
    

    def print_evaluation(self, precision, recall, f1, map, dcg, ndcg, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")

        print('precision:', precision)
        print('recall:', recall)
        print('F1 score:', f1)
        print('MAP:', map)
        print('DCG', dcg)
        print('NDCG', ndcg)
        print('MRR:', mrr)
      

    def log_evaluation(self, precision, recall, f1, map, dcg, ndcg, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        
        log({
            'Precision:': precision,
            'Recall:': recall,
            'F1 score:': f1,
            'MAP:': map,
            'DCG': dcg,
            'NDCG': ndcg,
            'MRR:': mrr
        })


    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        """
        According to lecture slides on Evaluation(pages 29-31), Average Precision is calculated 
        per query and thus per ranking. So, its function should be called per ranking only.
        Based on this observation, I have also changed the argument types of the AP function to receive
        1D arrays(single ranking).
        """
        # ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        """
        According to lecture slides on Evaluation(page 37), RR is calculated per query
        and thus per ranking. So, its function should be called per ranking only. 
        Based on this observation, I have also changed the argument types of the AP function to receive
        1D arrays(single ranking).
        """
        # rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        #call print and viualize functions
        self.print_evaluation(precision, recall, f1, map_score, dcg, ndcg, mrr)
        self.log_evaluation(precision, recall, f1, map_score, dcg, ndcg, mrr)




__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["Evaluator"]

import numpy as np
from reckit.dataiterator import DataIterator
from reckit.util import typeassert
from reckit.cython import eval_score_matrix
from reckit.cython import float_type, is_ndarray

metric_dict = {"Precision": 1, "Recall": 2, "MAP": 3, "NDCG": 4, "MRR": 5}
re_metric_dict = {value: key for key, value in metric_dict.items()}


class Evaluator(object):
    """Evaluator for item ranking task.

    Evaluation metrics of `Evaluator` are configurable and can
    automatically fit both leave-one-out and fold-out data splitting
    without specific indication:

    * **First**, evaluation metrics of this class are configurable via the
      argument `metric`. Now there are five configurable metrics: `Precision`,
      `Recall`, `MAP`, `NDCG` and `MRR`.

    * **Second**, this class and its evaluation metrics can automatically fit
      both leave-one-out and fold-out data splitting without specific indication.

      In **leave-one-out** evaluation:
        1) `Recall` is equal to `HitRatio`;
        2) The implementation of `NDCG` is compatible with fold-out;
        3) `MAP` and `MRR` have same numeric values;
        4) `Precision` is meaningless.
    """

    @typeassert(user_train_dict=(dict, None), user_test_dict=dict)
    def __init__(self, dataset, user_train_dict, user_test_dict,
                 metric=None, top_k=50, batch_size=1024, num_thread=8):
        """Initializes a new `Evaluator` instance.

        Args:
            user_train_dict (dict, None): Each key is user ID and the corresponding
                value is the list of **training items**.
            user_test_dict (dict): Each key is user ID and the corresponding
                value is the list of **test items**.
            metric (None or list of str): If `metric == None`, metric will
                be set to `["Precision", "Recall", "MAP", "NDCG", "MRR"]`.
                Otherwise, `metric` must be one or a sublist of metrics
                mentioned above. Defaults to `None`.
            top_k (int or list of int): `top_k` controls the Top-K item ranking
                performance. If `top_k` is an integer, K ranges from `1` to
                `top_k`; If `top_k` is a list of integers, K are only assigned
                these values. Defaults to `50`.
            batch_size (int): An integer to control the test batch size.
                Defaults to `1024`.
            num_thread (int): An integer to control the test thread number.
                Defaults to `8`.

        Raises:
             ValueError: If `metric` or one of its element is invalid.
        """
        super(Evaluator, self).__init__()
        if metric is None:
            metric = ["Precision", "Recall", "MAP", "NDCG", "MRR"]
        elif isinstance(metric, str):
            metric = [metric]
        elif isinstance(metric, (set, tuple, list)):
            pass
        else:
            raise TypeError("The type of 'metric' (%s) is invalid!" % metric.__class__.__name__)

        for m in metric:
            if m not in metric_dict:
                raise ValueError("There is not the metric named '%s'!" % metric)

        self.user_pos_train = dict()
        self.user_pos_test = dict()
        self.dataset = dataset
        self.set_train_data(user_train_dict)
        self.set_test_data(user_test_dict)

        self.metrics_num = len(metric)
        self.metrics = [metric_dict[m] for m in metric]
        self.num_thread = num_thread
        self.batch_size = batch_size

        self.max_top = top_k if isinstance(top_k, int) else max(top_k)
        if isinstance(top_k, int):
            self.top_show = np.arange(top_k) + 1
        else:
            self.top_show = np.sort(top_k)

    @typeassert(user_train_dict=(dict, None))
    def set_train_data(self, user_train_dict=None):
        self.user_pos_train = user_train_dict if user_train_dict is not None else dict()

    @typeassert(user_test_dict=dict)
    def set_test_data(self, user_test_dict):
        if len(user_test_dict) == 0:
            raise ValueError("'user_test_dict' can be empty.")
        self.user_pos_test = user_test_dict

    def metrics_info(self):
        """Get all metrics information.

        Returns:
            str: A string consist of all metrics information， such as
                `"Precision@10    Precision@20    NDCG@10    NDCG@20"`.
        """
        metrics_show = ['\t'.join([("%s@" % re_metric_dict[metric] + str(k)).ljust(12) for k in self.top_show])
                        for metric in self.metrics]
        metric = '\t'.join(metrics_show)
        return "metrics:\t%s" % metric

    def evaluate(self, model, test_users=None):
        """Evaluate `model`.

        Args:
            model: The model need to be evaluated. This model must have
                a method `predict(self, users)`, where the argument
                `users` is a list of users and the return is a 2-D array that
                contains `users` rating/ranking scores on all items.
            test_users: The users will be used to test.
                Default is None and means test all users in user_pos_test.

        Returns:
            str: A single-line string consist of all results, such as
                `"0.18663847    0.11239596    0.35824192    0.21479650"`.
        """
        # B: batch size
        # N: the number of items
        if not hasattr(model, "predict"):
            raise AttributeError("'model' must have attribute 'predict'.")
        # elif model.predict.__code__.co_argcount != 2:
        #     raise AttributeError("'model.predict' must have 1 parameter.")

        test_users = test_users if test_users is not None else list(self.user_pos_test.keys())
        if not isinstance(test_users, (list, tuple, set, np.ndarray)):
            raise TypeError("'test_user' must be a list, tuple, set or numpy array!")

        test_users = DataIterator(test_users, batch_size=self.batch_size, shuffle=False, drop_last=False)
        batch_result = []
        for batch_users in test_users:
            test_items = [self.user_pos_test[u] for u in batch_users]
            ranking_score = model.predict(batch_users)  # (B,N)
            if not is_ndarray(ranking_score, float_type):
                ranking_score = np.array(ranking_score, dtype=float_type)

            # set the ranking scores of training items to -inf,
            # then the training items will be sorted at the end of the ranking list.
            for idx, user in enumerate(batch_users):
                if user in self.user_pos_train and len(self.user_pos_train[user]) > 0:
                    train_items = self.user_pos_train[user]
                    ranking_score[idx][train_items] = -np.inf

            result = eval_score_matrix(ranking_score, test_items, self.metrics,
                                       top_k=self.max_top, thread_num=self.num_thread)  # (B,k*metric_num)
            batch_result.append(result)

        # concatenate the batch results to a matrix
        all_user_result = np.concatenate(batch_result, axis=0)  # (num_users, metrics_num*max_top)
        final_result = np.mean(all_user_result, axis=0)  # (1, metrics_num*max_top)

        final_result = np.reshape(final_result, newshape=[self.metrics_num, self.max_top])  # (metrics_num, max_top)
        final_result = final_result[:, self.top_show - 1]
        final_result = np.reshape(final_result, newshape=[-1])
        buf = '\t'.join([("%.8f" % x).ljust(12) for x in final_result])
        return final_result, buf

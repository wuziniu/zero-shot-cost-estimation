import collections
import itertools
import json
import logging
import types
import os
from json import JSONDecodeError
from time import perf_counter
import copy

import numpy as np
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.parse_run import dumper
from cross_db_benchmark.benchmark_tools.utils import load_json
from models.training.checkpoint import save_csv
from data_driven_cardinalities.cardinality_corrector.augment_plan import get_table_aliases_imdb, \
    get_act_est_card, q_err

logger = logging.getLogger(__name__)


def augment_cardinalities(schema, src, target, statistics_file, target_statistics_file,
                          hyperparameter_path, target_hyperparameter_path, tuning_scale=0.5):
    try:
        run = load_json(src, namespace=True)
    except JSONDecodeError:
        raise ValueError(f"Error reading {src}")


    # find out if this a non_inclusive workload (< previously replaced by <=)
    non_inclusive = False
    if any([b in src for b in ['job-light', 'scale', 'synthetic']]):
        non_inclusive = True
        print("Assuming NON-INCLUSIVE workload")

    q_stats = []
    est_pg = 0
    est_tuned = 0
    for q_id, p in enumerate(tqdm(run.parsed_plans)):
        p.plan_parameters.est_pg = 0
        p.plan_parameters.est_tuned = 0
        augment_bottom_up(schema, p, q_id, q_stats, non_inclusive=non_inclusive, tuning_scale=tuning_scale)
        est_pg += p.plan_parameters.est_pg
        est_tuned += p.plan_parameters.est_tuned

        def augment_prod(p):
            if len(p.children) == 0:
                p.plan_parameters.tuned_est_children_card = 1
            else:
                child_card = 1
                for c in p.children:
                    child_card *= c.plan_parameters.tuned_est_card
                    augment_prod(c)
                p.plan_parameters.tuned_est_children_card = child_card

        augment_prod(p)

    argumented_queries = types.SimpleNamespace()
    argumented_queries.database_stats = run.database_stats
    argumented_queries.run_kwargs = run.run_kwargs
    argumented_queries.parsed_plans = []
    for q_id, p in enumerate(run.parsed_plans):
        if q_id in all_MSCN_est:
            argumented_queries.parsed_plans.append(p)

    print(len(argumented_queries.parsed_plans))
    target_dir = os.path.dirname(target)
    os.makedirs(target_dir, exist_ok=True)
    with open(target, 'w') as outfile:
        json.dump(argumented_queries, outfile, default=dumper)

    feature_statistics = load_json(statistics_file, namespace=False)
    feature_statistics['est_tuned'] = {'max': 0.0, 'scale': 1.0, 'center': 1.0, 'type': 'numeric'}
    feature_statistics['tuned_est_card'] = feature_statistics['act_card']
    feature_statistics['tuned_est_children_card'] = feature_statistics['act_children_card']
    with open(target_statistics_file, "w") as f:
        json.dump(feature_statistics, f)

    hyperparameter = load_json(hyperparameter_path, namespace=False)
    hyperparameter['plan_featurization_name'] = "PostgresTunedCardDetail"
    with open(target_hyperparameter_path, "w") as f:
        json.dump(hyperparameter, f)

    return q_stats


def augment_bottom_up(schema, plan, q_id, q_stats, non_inclusive=False, tuning_scale=0.5):

    for c in plan.children:
        augment_bottom_up(schema, c, q_id, q_stats, non_inclusive=non_inclusive, tuning_scale=tuning_scale)

    # evaluate query
    act_card, pg_est_card = get_act_est_card(plan.plan_parameters)

    if act_card is None or pg_est_card is None:
        tuned_est = 1
        q_err_pg = 1
        q_err_tuned = 1
    else:
        tuned_est = act_card * tuning_scale + pg_est_card * (1 - tuning_scale)
        q_err_pg = q_err(pg_est_card, act_card)
        q_err_tuned = q_err(tuned_est, act_card)

    q_stats.append({
        'query_id': q_id,
        'q_errors_pg': q_err_pg,
        'q_errors_tuned': q_err_tuned
    })

    plan.plan_parameters.tuned_est_card = tuned_est


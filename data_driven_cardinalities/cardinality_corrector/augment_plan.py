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

logger = logging.getLogger(__name__)


def get_table_aliases_imdb():
    table_aliases = dict()
    table_aliases["title"] = "t"
    table_aliases["cast_info"] = "ci"
    table_aliases["movie_info"] = "mi"
    table_aliases["movie_info_idx"] = "mii"
    table_aliases["person_info"] = "pi"
    table_aliases["name"] = "n"
    table_aliases["aka_name"] = "an"
    table_aliases["keyword"] = "k"
    table_aliases["movie_keyword"] = "mk"
    table_aliases["movie_companies"] = "mc"
    table_aliases["movie_link"] = "ml"
    table_aliases["aka_title"] = "at"
    table_aliases["complete_cast"] = "cc"
    table_aliases["kind_type"] = "kt"
    table_aliases["role_type"] = "rt"
    table_aliases["char_name"] = "chn"
    table_aliases["info_type"] = "it"
    table_aliases["company_type"] = "ct"
    table_aliases["company_name"] = "cn"
    table_aliases["movie_link"] = "ml"
    table_aliases["link_type"] = "lt"
    table_aliases["comp_cast_type"] = "cct"
    return table_aliases


def augment_cardinalities(schema, all_MSCN_est, src, table_aliases, target, statistics_file, target_statistics_file,
                          hyperparameter_path, target_hyperparameter_path, scale=1):
    try:
        run = load_json(src, namespace=True)
    except JSONDecodeError:
        raise ValueError(f"Error reading {src}")

    q_stats = []

    # find out if this an non_inclusive workload (< previously replaced by <=)
    non_inclusive = False
    if any([b in src for b in ['job-light', 'scale', 'synthetic']]):
        non_inclusive = True
        print("Assuming NON-INCLUSIVE workload")

    est_pg = 0
    est_mscn = 0
    all_query_tables = []
    for q_id, p in enumerate(tqdm(run.parsed_plans)):
        if q_id not in all_MSCN_est:
            all_query_tables.append([])
            continue
        MSCN_est = all_MSCN_est[q_id]
        p.plan_parameters.est_pg = 0
        p.plan_parameters.est_mscn = 0
        all_tables = []
        _ = augment_bottom_up(schema, p, q_id, run.database_stats, MSCN_est, table_aliases, q_stats, p, scale,
                              non_inclusive=non_inclusive, all_tables=all_tables)
        all_query_tables.append(all_tables)
        est_pg += p.plan_parameters.est_pg
        est_mscn += p.plan_parameters.est_mscn

        def augment_prod(p):
            if len(p.children) == 0:
                p.plan_parameters.cc_est_children_card = 1
            else:
                child_card = 1
                for c in p.children:
                    child_card *= c.plan_parameters.cc_est_card
                    augment_prod(c)
                p.plan_parameters.cc_est_children_card = child_card

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
    feature_statistics['est_mscn'] = {'max': 0.0, 'scale': 1.0, 'center': 1.0, 'type': 'numeric'}
    feature_statistics['cc_est_card'] = feature_statistics['act_card']
    feature_statistics['cc_est_children_card'] = feature_statistics['act_children_card']
    with open(target_statistics_file, "w") as f:
        json.dump(feature_statistics, f)

    hyperparameter = load_json(hyperparameter_path, namespace=False)
    hyperparameter['plan_featurization_name'] = "PostgresCardCorrectorDetail"
    with open(target_hyperparameter_path, "w") as f:
        json.dump(hyperparameter, f)

    return all_query_tables, est_mscn, est_pg, q_stats


def report_stats(est_mscn, est_pg, q_stats):
    if len(q_stats) > 0:
        def report_percentiles(key):
            vals = np.array([q_s[key] for q_s in q_stats])
            print(f"{key}: p50={np.median(vals):.2f} p95={np.percentile(vals, 95):.2f} "
                  f"p99={np.percentile(vals, 99):.2f} pmax={np.max(vals):.2f}")

        report_percentiles('q_errors_pg')
        report_percentiles('q_errors_mscn')
        print(f"{est_mscn / (est_mscn + est_pg) * 100:.2f}% estimated using MSCN")


def match_sub_queries(tables, MSCN_est, table_aliases, q_id):
    aliased_tables = set()
    for table in tables:
        alias = table_aliases[table]
        aliased_tables.add(alias)
    for alias in MSCN_est:
        alias_set = set(alias)
        if alias_set == aliased_tables:
            return MSCN_est[alias]
    print(f"query {q_id}: {aliased_tables} not found in {MSCN_est.keys()}. Replacing with PG estimates")
    return None


def augment_bottom_up(schema, plan, q_id, database_statistics, MSCN_est, table_aliases,
                      q_stats, top_p, scale, all_tables, non_inclusive=False):
    workers_planned = vars(plan.plan_parameters).get('workers_planned')
    if workers_planned is None:
        workers_planned = 0
    # assert workers_planned is not None

    aggregation_below = 'Aggregate' in plan.plan_parameters.op_name

    # augment own tables
    tables = set()
    t_idx = vars(plan.plan_parameters).get('table')
    if t_idx is not None:
        table_stats = database_statistics.table_stats[t_idx]
        if hasattr(table_stats, 'relname'):
            table_name = table_stats.relname
        elif hasattr(table_stats, 'table'):
            table_name = table_stats.table
        else:
            raise NotImplementedError
        tables.add(table_name)

    for c in plan.children:
        c_aggregation_below, c_tables = augment_bottom_up(schema, c, q_id, database_statistics,
                                                          MSCN_est, table_aliases, q_stats,
                                                          top_p, scale,
                                                          non_inclusive=non_inclusive,
                                                          all_tables=all_tables
                                                          )
        aggregation_below |= c_aggregation_below
        tables.update(c_tables)

    # evaluate query
    act_card, pg_est_card = get_act_est_card(plan.plan_parameters)

    query_parsed = True
    q = None
    if len(tables) == 0:
        print("Could not parse query")
        query_parsed = False

    # query not supported
    if not query_parsed:
        plan.plan_parameters.cc_est_card = pg_est_card
        top_p.plan_parameters.est_pg += 1

    # group by not directly supported
    elif aggregation_below:
        plan.plan_parameters.cc_est_card = pg_est_card
        top_p.plan_parameters.est_pg += 1

    # we do not care about really small cardinalities
    elif (act_card is not None and pg_est_card <= 1000 and act_card <= 1000):
        plan.plan_parameters.cc_est_card = pg_est_card
        top_p.plan_parameters.est_pg += 1

    else:
        if plan.plan_parameters.op_name in {'Parallel Seq Scan', 'Hash Join', 'Nested Loop', 'Seq Scan', 'Materialize',
                                            'Hash', 'Parallel Hash', 'Merge Join', 'Gather', 'Gather Merge',
                                            'Hash Right Join', 'Hash Left Join', 'Nested Loop Left Join',
                                            'Merge Left Join', 'Merge Right Join'} \
                or plan.plan_parameters.op_name.startswith('XN ') \
                or plan.plan_parameters.op_name in {'Broadcast', 'Distribute'}:
            op_name = plan.plan_parameters.op_name

            cardinality_predict = match_sub_queries(tables, MSCN_est, table_aliases, q_id)
            if cardinality_predict is None:
                cardinality_predict = pg_est_card
            all_tables.append(copy.deepcopy(tables))
            if workers_planned > 0 and (op_name.startswith('Parallel')):
                cardinality_predict /= (workers_planned + 1)

            if act_card is not None:
                q_err_mscn = q_err(cardinality_predict, act_card)
                q_err_pg = q_err(pg_est_card, act_card)
            else:
                q_err_mscn = 1
                q_err_pg = 1

            # this was probably a bug, anyway rarely happens
            if q_err_mscn > 100 * q_err_pg:
                plan.plan_parameters.cc_est_card = pg_est_card
                top_p.plan_parameters.est_pg += 1
            else:
                plan.plan_parameters.cc_est_card = cardinality_predict
                top_p.plan_parameters.est_mscn += 1

                q_stats.append({
                    'query_id': q_id,
                    'q_errors_pg': q_err_pg,
                    'q_errors_mscn': q_err_mscn
                })

        # ignore this in the stats since pg semantics for cardinalities are different for this operator
        elif plan.plan_parameters.op_name in {'Index Only Scan', 'Index Scan', 'Parallel Index Only Scan',
                                              'Bitmap Index Scan', 'Parallel Bitmap Heap Scan', 'Bitmap Heap Scan',
                                              'Sort', 'Parallel Index Scan', 'BitmapAnd'}:
            plan.plan_parameters.cc_est_card = pg_est_card
            top_p.plan_parameters.est_pg += 1
        else:
            raise NotImplementedError(plan.plan_parameters.op_name)

    return aggregation_below, tables


def get_act_est_card(params):
    if hasattr(params, 'act_card'):
        act_card = params.act_card
        pg_est_card = params.est_card
    elif hasattr(params, 'est_rows'):
        act_card = params.act_avg_rows
        pg_est_card = params.est_rows
    # only estimated available
    elif hasattr(params, 'est_card'):
        # pretend that postgres is true
        act_card = None
        pg_est_card = params.est_card
    else:
        print(params)
        raise NotImplementedError
    return act_card, pg_est_card


def q_err(cardinality_predict, cardinality_true):
    if cardinality_predict == 0 and cardinality_true == 0:
        q_error = 1.
    elif cardinality_true == 0:
        q_error = 1.
    elif cardinality_predict == 0:
        q_error = cardinality_true
    else:
        q_error = max(cardinality_predict / cardinality_true, cardinality_true / cardinality_predict)
    return q_error
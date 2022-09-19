import argparse
import glob

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from models.preprocessing.feature_statistics import gather_feature_statistics
from models.training.train import train_default, train_readout_hyperparams

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workload_runs', default=None, nargs='+')
    parser.add_argument('--test_workload_runs', default=None, nargs='+')
    parser.add_argument('--statistics_file', default="../zero-shot-data/runs/parsed_plans/statistics_workload_combined.json")
    parser.add_argument('--raw_dir', default=None)
    parser.add_argument('--target', default="../zero-shot-data/evaluation/job_full_tune/")
    parser.add_argument('--loss_class_name', default='QLoss')
    parser.add_argument('--filename_model', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--max_epoch_tuples', type=int, default=100000)
    parser.add_argument('--max_no_epochs', type=int, default=None)
    parser.add_argument('--limit_queries', type=int, default=None)
    parser.add_argument('--limit_queries_affected_wl', type=int, default=None)
    parser.add_argument('--limit_num_tables', type=int, default=None)
    parser.add_argument('--limit_runtime', type=int, default=None)
    parser.add_argument('--lower_bound_num_tables', type=int, default=None)
    parser.add_argument('--lower_bound_runtime', type=int, default=None)
    parser.add_argument('--database', default=DatabaseSystem.POSTGRES, type=DatabaseSystem,
                        choices=list(DatabaseSystem))
    parser.add_argument('--gather_feature_statistics', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument('--train_model', action='store_true')
    parser.add_argument('--plan_featurization', default='PostgresTrueCardDetail')
    parser.add_argument('--hyperparameter_path', default="setup/tuned_hyperparameters/tune_best_config.json")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    
    workload_runs = ["../zero-shot-data/runs/parsed_plans/airline/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/airline/workload_100k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/ssb/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/ssb/workload_100k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/tpc_h/complex_workload_200k_s1_c8220.json",  
                      "../zero-shot-data/runs/parsed_plans/walmart/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/walmart/workload_100k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/financial/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/financial/workload_100k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/basketball/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/basketball/workload_100k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/accidents/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/accidents/workload_100k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/movielens/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/movielens/workload_100k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/baseball/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/baseball/workload_100k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/hepatitis/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/hepatitis/workload_100k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/tournament/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/tournament/workload_100k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/credit/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/credit/workload_100k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/employee/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/employee/workload_100k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/consumer/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/consumer/workload_100k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/geneea/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/geneea/workload_100k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/genome/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/genome/workload_100k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/carcinogenesis/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/carcinogenesis/workload_100k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/seznam/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/seznam/workload_100k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/fhnk/complex_workload_200k_s1_c8220.json", 
                      "../zero-shot-data/runs/parsed_plans/fhnk/workload_100k_s1_c8220.json"]
    
    if args.workload_runs is None:
        args.workload_runs = workload_runs
    else:
        args.workload_runs = workload_runs + args.workload_runs       
        
    if args.test_workload_runs is None:
        args.test_workload_runs = ["../zero-shot-data/runs/parsed_plans/imdb_full/job_full_c8220.json"]
        
    
    if args.gather_feature_statistics:
        # gather_feature_statistics
        workload_runs = []

        for wl in args.workload_runs:
            workload_runs += glob.glob(f'{args.raw_dir}/*/{wl}')
        
        broken_files = ["../zero-shot-data/runs/parsed_plans/tpc_h/workload_100k_s1_c8220.json"]
        for file in broken_files:
            if file in workload_runs:
                workload_runs.remove(file)
        gather_feature_statistics(workload_runs, args.target)

    if args.train_model:
        if args.hyperparameter_path is None:
            # for testing
            train_default(args.workload_runs, args.test_workload_runs, args.statistics_file, args.target,
                          args.filename_model, plan_featurization=args.plan_featurization, device=args.device,
                          num_workers=args.num_workers, max_epoch_tuples=args.max_epoch_tuples,
                          seed=args.seed, database=args.database, limit_queries=args.limit_queries,
                          limit_queries_affected_wl=args.limit_queries_affected_wl, max_no_epochs=args.max_no_epochs,
                          skip_train=args.skip_train, loss_class_name=args.loss_class_name, save_best=args.save_best)
        else:
            model = train_readout_hyperparams(args.workload_runs, args.test_workload_runs, args.statistics_file, args.target,
                                      args.filename_model, args.hyperparameter_path, device=args.device,
                                      num_workers=args.num_workers, max_epoch_tuples=args.max_epoch_tuples,
                                      seed=args.seed, database=args.database, limit_queries=args.limit_queries,
                                      limit_queries_affected_wl=args.limit_queries_affected_wl,
                                      limit_num_tables=args.limit_num_tables,
                                      limit_runtime=args.limit_runtime,
                                      lower_bound_num_tables=args.lower_bound_num_tables,
                                      lower_bound_runtime=args.lower_bound_runtime,
                                      max_no_epochs=args.max_no_epochs, skip_train=args.skip_train,
                                      loss_class_name=args.loss_class_name, save_best=args.save_best)

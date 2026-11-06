"""Independent CIC study: prepare -> tune weighted FedAvg -> compare both weightings."""
import argparse
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','audit','doctor','tune','train','results'])
    parser.add_argument('--config',default=str(Path(__file__).resolve().parent/'configs/tuning.json'))
    parser.add_argument('--model',choices=['all','dnn','cnn'],default='all')
    parser.add_argument('--task',choices=['binary'],default='binary')
    parser.add_argument('--max-new',type=int)
    parser.add_argument('--scenarios',help='Optional comma-separated final scenarios.')
    args = parser.parse_args()
    if args.command == 'prepare':
        from prepare import prepare
        prepare(args.config)
    elif args.command == 'doctor':
        from runtime import configure
        print(configure())
    else:
        import notebook_workflow as workflow
        c = workflow.settings(args.config)
        if args.command == 'audit':
            from prepare import verify_prepared
            m = workflow.data_manifest(c)
            verify_prepared(m,workflow.run_root(c)/'prepared',deep=True)
            print(m['audit'])
            print(workflow.split_table(args.config).to_string(index=False))
        elif args.command == 'results': print(workflow.report(args.config).to_string(index=False))
        else:
            models = c['models'] if args.model == 'all' else [args.model]
            tasks = (workflow.tuning_tasks(c,args.task) if args.command == 'tune'
                     else [args.task])
            for task in tasks:
                for model in models:
                    if args.command == 'tune':
                        print(workflow.tune(model,args.config,args.max_new,task))
                    else:
                        scenarios = args.scenarios.split(',') if args.scenarios else None
                        print(workflow.run_experiments(model,args.config,scenarios,task).to_string(index=False))

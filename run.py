"""Entry point for the CIC Log+Z-score project (WSL)."""
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'doctor', 'tune', 'train', 'results'])
    parser.add_argument('--model', choices=['dnn', 'lstm', 'cnn'], default='dnn')
    args = parser.parse_args()
    if args.command == 'prepare':
        from prepare import prepare
        prepare()
    elif args.command == 'doctor':
        from runtime import configure
        print(configure())
    else:
        import notebook_workflow as workflow
        if args.command == 'tune':
            print(workflow.tune(args.model))
        elif args.command == 'train':
            print(workflow.run_experiments(args.model).to_string(index=False))
        else:
            print(workflow.report().to_string(index=False))

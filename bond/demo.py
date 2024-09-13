from training.autotrain_bond import BONDTrainer
from training.autotrain_bond_ensemble import ESBTrainer
from dataset.preprocess_SND import dump_name_pubs, dump_features_relations_to_file, build_graph
from params import set_params



def pipeline(args):
    model = args.model
    
    # Module-1: Data Loading
    if args.dump_data:
        dump_name_pubs(args)
        dump_features_relations_to_file(args)
        build_graph(args)

    # Modules-2: Feature Creation & Module-3: Model Construction
    if model == 'bond':
        trainer = BONDTrainer()
        trainer.fit(datatype=args.mode)
    elif model == 'bond+':
        trainer = ESBTrainer()
        trainer.fit(datatype=args.mode)

    # Modules-4: Evaluation
    # Please uppload your result to http://whoiswho.biendata.xyz/#/

if __name__ == "__main__":
    args = set_params()
    pipeline(args)
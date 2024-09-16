# import faulthandler
# # 在import之后直接添加以下启用代码即可
# faulthandler.enable()

from training.autotrain_bond import BONDTrainer
from training.autotrain_bond_ensemble import ESBTrainer
from dataset.preprocess_SND import dump_name_pubs, dump_features_relations_to_file, build_graph
from params import set_params


def pipeline(args):
    model = args.model
    
    # Module-1: Data Loading
    if args.dump_data:
        print("point1")
        dump_name_pubs(args)
        print("point2")
        dump_features_relations_to_file(args)
        print("point3")
        build_graph(args)
        print("point4")

    # Modules-2: Feature Creation & Module-3: Model Construction
    if model == 'bond':
        trainer = BONDTrainer()
        print("point5")
        trainer.fit(datatype=args.mode)
        print("point6")
    elif model == 'bond+':
        trainer = ESBTrainer()
        print("point7")
        trainer.fit(datatype=args.mode)
        print("point8")

    # Modules-4: Evaluation
    # Please uppload your result to http://whoiswho.biendata.xyz/#/

if __name__ == "__main__":
    print("point9")
    args = set_params()
    print("point10")
    pipeline(args)
    print("point11")

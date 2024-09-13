# BOND
This repository contains the source code of paper "BOND: Bootstrapping From-Scratch Name Disambiguation with Multi-task Promoting".

![Overview](/Bond.png)

## Usage

### Data preparation

The datasets WhoIsWho can be downloaded from: [Dataset](https://www.aminer.cn/whoiswho)

Paper embedding can be downloaded from: [Embedding](https://pan.baidu.com/s/1A5XA9SCxvENM2kKPUv6X4Q?pwd=c9kk )
Password: c9kk

Download dataset "src" and embedding "paper_emb", organize in directory "dataset" as follows:

```
    .
    ├── dataset
        ├── data
            ├── paper_emb
            └── src
```

### One-line Command
Execute the following command. You can freely choose the data type(train/valid/test) by '--mode train/valid/test'. Post-matching is performed by default, you can control this operation by '--post_match True/False':
```
stdbuf -oL -eL python demo.py --model bond --post_match True --mode train --save_path dataset/data/v2 --dump_data 2>&1 | tee reproduce.log
stdbuf -oL -eL python demo.py --model bond --post_match True --mode valid --save_path dataset/data/v2 2>&1 | tee reproduce_valid.log
stdbuf -oL -eL python demo.py --model bond --post_match True --mode test --save_path dataset/data/v2 2>&1 | tee reproduce_test.log
```

It will do name disambiguation with clearly designed pipeline, you can choose model BOND or BOND+ by the parameter 'model':
```
def pipeline(model):
    # Module-1: Data Loading
    dump_name_pubs()
    dump_features_relations_to_file()
    build_graph()

    # Modules-2: Feature Creation & Module-3: ModeConstruction
    if model == 'bond':
        trainer = BONDTrainer()
        trainer.fit(datatype=args.mode)
    elif model == 'bond+':
        trainer = ESBTrainer()
        trainer.fit(datatype=args.mode)

    # Modules-4: Evaluation
    # Please uppload your result to http://whoiswho.biendata.xyz/#/

```

The output will be stored in 'bond/out'. To evaluate, please upload your result to http://whoiswho.biendata.xyz/#/.

## Implementation requirements

```
    gensim==4.3.0
    matplotlib==3.7.1
    numpy==1.24.3
    pandas==1.5.3
    pinyin==0.4.0
    scikit-learn==1.2.2
    scipy==1.10.1
    torch==1.12.1
    torch-geometric==2.2.0
    tqdm==4.65.0
```

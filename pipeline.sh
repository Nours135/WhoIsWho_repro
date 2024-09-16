#! /bin/bash

## step 1, store data into path dataset data

```
    .
    ├── dataset
        ├── data
            ├── {data_version}
                ├── src
                    ├── train
                    ├── sna-valid
                    ├── sna-test
```
# in each folder of src, there are 3 files, pubs.json, atuhor_pubs.json, ground_truth.json

## step 2, preprocess data
# NOTE: mv data to "./whoiswho/dataset" for to run preprocessing
# step 2.1, file: whoiswho/dataset/data_process.py
    # call the fowlling function
    # processdata_SND(version=version)
    # 主要是为了 processed_data/extrace_text/plain_text.txt，为了拿这个训练word2vec，别的提取对不对不重要，也没有可能会删掉
    # plain_text.txt 包含了train valid test所有的内容
    # 直接改好代码，调出环境，移动到目标文件夹： python data_process.py

## step 3, train Word2Vec and dump paper_emb.pkl
# whoiswho/featureGenerator/sndFeature/semantic_features.py
    # data stored in   processed_data/snd-embs


## step 4, mv data to bond
# cd {proj_dir}/whoiswho/dataset/data/v2/SND/src  # 移动到data的文件，目前是 v2 是 bond 的数据，后面用我们的数据应该对应的是 openalex
# mv train {proj_dir}/bond/dataset/data/v2/src
# mv valid {proj_dir}/bond/dataset/data/v2/src
# mv test {proj_dir}/bond/dataset/data/v2/src
# mv test /home/zrli/WhoIsWho_repro-1/bond/dataset/data/v2/src
# /home/zrli/WhoIsWho_repro-1/

# 移动paper-embs
# mv ./processed_data/snd-embs {proj_dir}/bond/dataset/data/v2/
# mv ./processed_data/snd-embs /home/zrli/WhoIsWho_repro-1/bond/dataset/data/v2/

# cd {proj_dir}/bond/dataset/data/v2
# cd /home/zrli/WhoIsWho_repro-1/bond/dataset/data/v2
# 下面是在给文件夹改名
# mv snd-embs paper_emb 
# cd src
# mv valid sna-valid  
# mv test sna-test 


## step 5, run model on train, valid, test set
# NOTE: remember to change file name of ./out/res.json -> ./out/res_bond_{mode}.json 
# only need to run the following command
# only the first run needs to dump data to file system
# use "&&" to connect commands   这些被连接的命令会自动串行
# 加入 --cuda 这个参数可以使用显卡，但是我的设备4G显存不够！！只能在cpu训练了
stdbuf -oL -eL python demo.py --model bond --post_match --mode train --save_path dataset/data/v2 --dump_data 2>&1 | tee reproduce.log &&
mv ./out/res.json ./out/res_bond_train.json && 
stdbuf -oL -eL python demo.py --model bond --post_match --mode valid --save_path dataset/data/v2 2>&1 | tee reproduce_valid.log &&
mv ./out/res.json ./out/res_bond_valid.json && 
stdbuf -oL -eL python demo.py --model bond --post_match --mode test --save_path dataset/data/v2 2>&1 | tee reproduce_test.log &&
mv ./out/res.json ./out/res_bond_test.json

## step 6, run evaluation
python run_evaluate.py
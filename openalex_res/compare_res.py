
# 1. compare distinction between share and not share weights

import json
from pic_utils import plot_histogram

def jaccard(a, b):
    return len(a & b) / len(a | b)

def compare_share_weights_ornot():
    with open('./share_weights/res_bond_train.json', 'r') as fp:
        share_res = json.load(fp)
        
    with open('./no_share_weights/noshare_res_bond_train.json', 'r') as fp:
        no_share_res = json.load(fp)
        
    # print(f'samples of share_res: {list(share_res.items())[:3]}')
    # print(f'samples of no_share_res: {list(no_share_res.items())[:3]}')
    
    exam_res = dict()
    for name in share_res:
        res = share_res[name]
        name_jaccard_sim_l = []
        for disambiguated_author_pubs in res:
            j_sim_l = [jaccard(set(pub_l), set(disambiguated_author_pubs)) for pub_l in no_share_res[name]]
            name_jaccard_sim_l.append(max(j_sim_l)) 
        exam_res[name] = name_jaccard_sim_l
    
    # write to json
    with open('./share_no_share_res.json', 'w') as fp:
        json.dump(exam_res, fp)


    
def compare_pred_with_gt():
    # step 1 load res
    with open('./no_share_weights/noshare_res_bond_train.json', 'r') as fp:
        no_share_res = json.load(fp)
    
    # step 2 load gt
    with open('/share/whoiswho_data/openalex/src/train/train_author.json') as fp:
        gt_json_data = json.load(fp)

    print(f'samples of share_res: {list(no_share_res.keys())[:3]}')
    print(f'samples of gt_json_data: {list(gt_json_data.keys())[:3]}')

    exam_res = dict()
    for name in gt_json_data:
        res = gt_json_data[name]
        name_jaccard_sim_l = []
        for disambiguated_author_pubs in res.values():
            j_sim_l = [jaccard(set(pub_l), set(disambiguated_author_pubs)) for pub_l in no_share_res[name]]
            name_jaccard_sim_l.append(max(j_sim_l)) 
        exam_res[name] = name_jaccard_sim_l
        
    with open('./noshare_and_gt_res.json', 'w') as fp:
        json.dump(exam_res, fp)
        
        
    # hist show jaccard
    total_jaccard = []
    for name, j_l in exam_res.items():
        total_jaccard += j_l
    plot_histogram(total_jaccard, title='Jaccard sim distribution', xlabel='Jaccard sim', savefile='./temp.png')
    

def case_study(case_name='jiawen_wang'):
    '''
    case study of author
    '''
    # step 1 load res
    with open('./no_share_weights/noshare_res_bond_train.json', 'r') as fp:
        no_share_res = json.load(fp)
    
    # step 2 load gt
    with open('/share/whoiswho_data/openalex/src/train/train_author.json') as fp:
        gt_json_data = json.load(fp)

    pred = no_share_res[case_name]
    gt = gt_json_data[case_name]
    print(pred)
    print(gt)
    print(len(pred))
    print(len(gt))
     
if __name__ == '__main__':
    # compare_share_weights_ornot()
    # compare_pred_with_gt()
    
    case_study('william_weiner')
    # case 1 jiawen_wang   都预测到一个author里了
    # [['W4362677437', 'W3209226207', 'W4286571727', 'W4220961777', 'W4286571780', 'W3134586026', 'W3164418477', 'W3159633994', 'W3161421427', 'W3163167024', 'W3094041807', 'W3091852081', 'W3092565541']]
    # {'A5045058519_1': ['W4362677437'], 'A5045058519_6': ['W3209226207', 'W4286571727', 'W4220961777', 'W4286571780', 'W3134586026', 'W3164418477', 'W3159633994', 'W3161421427', 'W3163167024', 'W3094041807', 'W3091852081', 'W3092565541']}
    
    # case 2 jonathan_casey
    # 省略，只有一个author，但是被拆成了两个，最接近的那个结果，占了 87% 
    
    # case 3 b._jeanrenaud  [0.9917355371900827, 0.004149377593360996]
    # 分的比较差
    
    # case 4 sang_kim [0.1643192488262911, 0.29577464788732394, 0.3333333333333333, 0.15492957746478872, 0.05164319248826291]
    # 5个gt author，只预测出来了两个 author
    
    
    # case 5  william_weiner  [0.6484018264840182]
    # 只有一个gt author，预测出来了 9 个，最接近的这个match 了 64% 的 pubs
    
    
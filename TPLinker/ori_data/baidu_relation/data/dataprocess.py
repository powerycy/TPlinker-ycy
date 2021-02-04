import re
import codecs
import json
def process(data_path,target_path):
    target_file = open(target_path,'w',encoding='utf8')
    with codecs.open(data_path) as read_data:
        data_list = []
        for num,line in enumerate(read_data):
            line = re.sub('\n','',line)
            data_dict = {}
            relation_list = []
            entity_list = []
            line_json = json.loads(line)
            data_dict["id"] = num
            data_dict["text"] = line_json['text']
            # c = line_json['spo_list']
            for rel_num in range(len(line_json['spo_list'])):
                relation = {}
                entity_obj = {}
                entity_sub = {}
                relation['object'] = line_json['spo_list'][rel_num]['object']['@value']
                relation['subject'] = line_json['spo_list'][rel_num]['subject']
                relation['predicate'] = line_json['spo_list'][rel_num]['predicate']
                entity_obj['text'] = line_json['spo_list'][rel_num]['object']['@value']
                entity_obj['type'] = line_json['spo_list'][rel_num]['object_type']['@value']
                entity_sub['text'] = line_json['spo_list'][rel_num]['subject']
                entity_sub['type'] = line_json['spo_list'][rel_num]['subject_type']
                relation_list.append(relation)
                entity_list.append(entity_obj)
                entity_list.append(entity_sub)
                data_dict['relation_list'] = relation_list
                data_dict['entity_list'] = entity_list
            data_list.append(data_dict)
    target_file.write(json.dumps(data_list,ensure_ascii=False))
    target_file.close()
process('ori_data/baidu_relation/data/dev.json','data4bert/baidu_relation/train_dev.json')


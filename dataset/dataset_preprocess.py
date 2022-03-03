import numpy as np
import json

def process_entity():
    id = 0
    f = open("entity/es.tsv", "r")
    fw = open("entity/es_id.tsv", "w")
    for line in f:
        print(line)
        fw.write(str(id)+"\t"+line)
        id += 1

def process_relation():
    id = 0
    f = open("relations.txt", "r")
    fw = open("relations_id.txt", "w")
    for line in f:
        print(line)
        fw.write(str(id)+"\t"+line)
        id += 1

def process_kg():
    fw = open("kg/es-kg.tsv", "w")
    ftr = open("kg/es-train.tsv", "r")
    fva = open("kg/es-val.tsv", "r")
    fte = open("kg/es-test.tsv", "r")

    for line in ftr:
        print(line)
        fw.write(line)

    for line in fva:
        print(line)
        fw.write(line)

    for line in fte:
        print(line)
        fw.write(line)

def process_align():
    fr = open("seed_alignlinks/es-en.tsv", "r")
    fw = open("seed_alignlinks/en-es.tsv", "w")
    for line in fr:
        l = line.strip().split()
        # print(l[0].split('.')[0])
        cont = l[0].split('.')[0] + "\t" + l[1].split('.')[0] + "\n"
        print(cont)
        fw.write(cont)

def replace_non_en_id():
    fen = open("data/en_es/ent_ids_2.tsv", "r")
    fenw = open("data/en_es/ent_ids_2_newid.tsv", "w")
    for line in fen:
        l = line.strip().split()
        # print(str(int(l[0])+1000000))
        cont = str(int(l[0])+13996) + "\t" + l[1] + "\n"
        print(cont)
        fenw.write(cont)

    fref = open("data/en_es/ref_ent_ids.tsv", "r")
    frefw = open("data/en_es/ref_ent_ids_newid.tsv", "w")
    for line in fref:
        l = line.strip().split()
        # print(str(int(l[0])+1000000))
        cont = l[0] + "\t" + str(int(l[1])+13996) + "\n"
        print(cont)
        frefw.write(cont)

    ftri = open("data/en_es/triples_2.tsv", "r")
    ftriw = open("data/en_es/triples_2_newid.tsv", "w")
    for line in ftri:
        l = line.strip().split()
        # print(str(int(l[0])+1000000))
        cont = str(int(l[0])+13996) + "\t" + l[1] + "\t" + str(int(l[2])+13996) + "\n"
        print(cont)
        ftriw.write(cont)

def check_dataset_overlapping():
    f = open('data/en_es/triples_1_basic.tsv', 'r')
    entity = {}
    for line in f:
        l = line.strip().split()
        if l[0] not in entity.keys():
            entity[l[0]] = 1
        else:
            entity[l[0]] += 1

        if l[2] not in entity.keys():
            entity[l[2]] = 1
        else:
            entity[l[0]] += 1

    print(len(entity.keys()))

    count_trip = 0
    f1 = open('data/en_es/triples_1_meta_valid.tsv', 'r')
    entity_meta = {}
    for line in f1:
        l = line.strip().split()
        if l[0] not in entity_meta.keys():
            entity_meta[l[0]] = 1
        else:
            entity_meta[l[0]] += 1

        if l[2] not in entity_meta.keys():
            entity_meta[l[2]] = 1
        else:
            entity_meta[l[0]] += 1

        if l[0] not in entity.keys() or l[2] not in entity.keys():
            count_trip += 1

    print(len(entity_meta))

    count = 0
    for k in entity_meta.keys():
        if k not in entity.keys():
            count += 1

    print(count)
    print(count_trip)

def remove_unseen_entity_pairs():
    f = open('data/en_es/triples_2_basic_train.tsv', 'r')
    entity = {}
    for line in f:
        l = line.strip().split()
        if l[0] not in entity.keys():
            entity[l[0]] = 1
        else:
            entity[l[0]] += 1

        if l[2] not in entity.keys():
            entity[l[2]] = 1
        else:
            entity[l[0]] += 1

    print(len(entity.keys()))

    f1 = open('data/en_es/triples_2_meta_test.tsv', 'r')
    fw = open('data/en_es/triples_2_meta_test_.tsv', 'w')
    for line in f1:
        l = line.strip().split()
        if l[0] in entity.keys() and l[2] in entity.keys():
            fw.write(line)



def load_relations():
    file = 'en'
    ftr = open('relations/' + file + '-train.tsv', 'r')
    fval = open('relations/' + file + '-val.tsv', 'r')
    fte = open('relations/' + file + '-test.tsv', 'r')

    train = []
    val = []
    test = []

    for line in ftr:
        l = line.strip().split()
        train.append(l[0])

    for line in fval:
        l = line.strip().split()
        val.append(l[0])

    for line in fte:
        l = line.strip().split()
        test.append(l[0])

    all_relation = train + val + test

    return all_relation, train, val, test

def load_triplets():
    file = 'es'
    ft = open('data/en_'+file+'/triples_2_newid.tsv', 'r')
    relation2trip = {}
    for line in ft:
        l = line.strip().split()
        if l[1] not in relation2trip.keys():
            relation2trip[l[1]] = [(l[0], l[1], l[2])]
        else:
            relation2trip[l[1]].append((l[0], l[1], l[2]))

    all_meta_rels, train, val, test = load_relations()

    print(len(all_meta_rels))

    ####### generate basic triplets
    fb = open('data/en_'+file+'/triples_2_basic.tsv', 'w')
    count = 0
    for k in relation2trip.keys():
        if k not in all_meta_rels:
            for item in relation2trip[k]:
                cont = item[0] + '\t' + item[1] + '\t' + item[2] + '\n'
                fb.write(cont)
        else:
            count += 1
    print(count)
    #######

    ####### generate meta train triplets
    print(len(train))
    fmtr = open('data/en_'+file+'/triples_2_meta_train.tsv', 'w')
    count = 0
    for k in relation2trip.keys():
        if k in train:
            for item in relation2trip[k]:
                cont = item[0] + '\t' + item[1] + '\t' + item[2] + '\n'
                fmtr.write(cont)
            count += 1
    print(count)

    ######### generate meta valid triplets
    print(len(val))
    fmtr = open('data/en_'+file+'/triples_2_meta_valid.tsv', 'w')
    count = 0
    for k in relation2trip.keys():
        if k in val:
            for item in relation2trip[k]:
                cont = item[0] + '\t' + item[1] + '\t' + item[2] + '\n'
                fmtr.write(cont)
            count += 1
    print(count)

    ######### generate meta test triplets
    print(len(test))
    fmtr = open('data/en_'+file+'/triples_2_meta_test.tsv', 'w')
    count = 0
    for k in relation2trip.keys():
        if k in test:
            for item in relation2trip[k]:
                cont = item[0] + '\t' + item[1] + '\t' + item[2] + '\n'
                fmtr.write(cont)
            count += 1
    print(count)

def load_en_triples():
    ft = open('data/en_es/triples_1.tsv', 'r')
    relation2trip = {}
    for line in ft:
        l = line.strip().split()
        if l[1] not in relation2trip.keys():
            relation2trip[l[1]] = [(l[0], l[1], l[2])]
        else:
            relation2trip[l[1]].append((l[0], l[1], l[2]))

    all_meta_rels, train, val, test = load_relations()

    print(len(all_meta_rels))

    tra_val = train + val

    ####### generate basic triplets
    fb = open('data/en_es/triples_1_basic.tsv', 'w')
    count = 0
    for k in relation2trip.keys():
        if k not in tra_val:
            for item in relation2trip[k]:
                cont = item[0] + '\t' + item[1] + '\t' + item[2] + '\n'
                fb.write(cont)
        else:
            count += 1
    print(count)
    #######

    ####### generate meta train triplets
    print(len(train))
    fmtr = open('data/en_es/triples_1_meta_train.tsv', 'w')
    count = 0
    for k in relation2trip.keys():
        if k in train:
            for item in relation2trip[k]:
                cont = item[0] + '\t' + item[1] + '\t' + item[2] + '\n'
                fmtr.write(cont)
            count += 1
    print(count)

    ######### generate meta valid triplets
    print(len(val))
    fmtr = open('data/en_es/triples_1_meta_valid.tsv', 'w')
    count = 0
    for k in relation2trip.keys():
        if k in val:
            for item in relation2trip[k]:
                cont = item[0] + '\t' + item[1] + '\t' + item[2] + '\n'
                fmtr.write(cont)
            count += 1
    print(count)



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def load(filename, output):
    f = open(filename, 'r')
    dic = {}
    for line in f:
        l = line.strip().split()
        if l[1] not in dic.keys():
            dic[l[1]] = [l]
        else:
            dic[l[1]].append(l)
    print(len(dic.keys()))

    with open(output, 'w') as fp:
        json.dump(dic, fp, cls=NpEncoder)

def prepare_candidates(filename):
    size = 13996
    rel2ent = {}
    rel2cand = {}
    f = open(filename, 'r')
    for line in f:
        l = line.strip().split()
        if int(l[1]) not in rel2ent.keys():
            rel2ent[int(l[1])] = [int(l[0]), int(l[2])]
        else:
            rel2ent[int(l[1])].append(int(l[0]))
            rel2ent[int(l[1])].append(int(l[2]))

    for rel in rel2ent.keys():
        false_candidates = np.array(list(set(np.arange(size, 25801)) - set(rel2ent[rel])))
        rel2cand[rel] = list(np.random.choice(false_candidates, size=len(false_candidates)))

    return rel2cand


def gen_rel2cand(output):
    rel2cand_train = prepare_candidates('data/en_es/triples_2_meta_train_.tsv')
    rel2cand_valid = prepare_candidates('data/en_es/triples_2_meta_valid_.tsv')
    rel2cand_test = prepare_candidates('data/en_es/triples_2_meta_test_.tsv')

    rel2cand = {**rel2cand_train, **rel2cand_valid, **rel2cand_test}

    print(len(rel2cand.keys()))

    with open(output, 'w') as fp:
        json.dump(rel2cand, fp, cls=NpEncoder)

from trainer import *
from params import *
from data_loader import *
import json

if __name__ == '__main__':
    params = get_params()

    print("---------Parameters---------")
    for k, v in params.items():
        print(k + ': ' + str(v))
    print("----------------------------")

    # control random seed
    if params['seed'] is not None:
        SEED = params['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)

    # select the dataset
    for k, v in data_dir.items():
        data_dir[k] = params['data_path']+v

    tail = ''
    if params['data_form'] == 'In-Train':
        tail = '_in_train'

    dataset_1 = dict()
    dataset_2 = dict()
    print("loading train_tasks_1{} ... ...".format(tail))
    dataset_1['train_tasks'] = json.load(open(data_dir['train_tasks_1'+tail]))
    print("loading train_tasks_2{} ... ...".format(tail))
    dataset_2['train_tasks'] = json.load(open(data_dir['train_tasks_2'+tail]))
    print("loading test_tasks ... ...")
    dataset_2['test_tasks'] = json.load(open(data_dir['test_tasks_2']))
    print("loading dev_tasks ... ...")
    dataset_1['dev_tasks'] = json.load(open(data_dir['dev_tasks_1']))
    print("loading dev_tasks ... ...")
    dataset_2['dev_tasks'] = json.load(open(data_dir['dev_tasks_2']))
    print("loading rel2candidates_1{} ... ...".format(tail))
    dataset_1['rel2candidates'] = json.load(open(data_dir['rel2candidates_1'+tail]))
    print("loading rel2candidates{} ... ...".format(tail))
    dataset_2['rel2candidates'] = json.load(open(data_dir['rel2candidates_2'+tail]))

    if params['data_form'] == 'Pre-Train':
        print('loading embedding ... ...')
        dataset_1['ent2emb'] = np.load(data_dir['emb'])
        dataset_2['ent2emb'] = np.load(data_dir['emb'])

    print("----------------------------")

    # data_loader
    train_data_loader_source = DataLoader(dataset_1, params, step='train')
    train_data_loader_target = DataLoader(dataset_2, params, step='train')
    dev_data_loader_source = DataLoader(dataset_1, params, step='dev')
    dev_data_loader_target = DataLoader(dataset_2, params, step='dev')
    test_data_loader = DataLoader(dataset_2, params, step='test')
    data_loaders = [train_data_loader_source, dev_data_loader_source, test_data_loader, train_data_loader_target, dev_data_loader_target]

    # trainer
    trainer = Trainer(data_loaders, dataset_2, params)

    if params['step'] == 'train':
        trainer.train()
        trainer.train_target()
        print("test")
        print(params['prefix'])
        trainer.reload()
        trainer.eval(istest=True, istarget=True)
    elif params['step'] == 'test':
        print(params['prefix'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(istest=True)
        else:
            trainer.eval(istest=True, istarget=True)
    elif params['step'] == 'dev':
        print(params['prefix'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(istest=False)
        else:
            trainer.eval(istest=False, istarget=True)


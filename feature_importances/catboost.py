from datasets import load_from_disk
from catboost import CatBoostClassifier

def get_feature_importances(task):
    dataset = load_from_disk(f'datasets/{task}.hf')

    if task == 'WordContent':
        num_iter = 1000
        params = {'l2_leaf_reg': 10}
    else:
        num_iter = 5000
        model = CatBoostClassifier(iterations=num_iter, task_type='GPU', devices='1-6', random_seed=42)
        params = {'l2_leaf_reg': [1, 5, 10, 15]}
        params = model.grid_search(params, dataset['train']['X'], dataset['train']['y'])['params']

    model = CatBoostClassifier(iterations=num_iter, task_type='GPU', devices='1-6', random_seed=42, **params)
    model.fit(dataset['train']['X'], dataset['train']['y'])
    
    with open(f'catboost/{task}.npy', 'wb') as f:
        np.save(f, model.feature_importances_)

if __name__ == '__main__':
    tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents', 
         'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 
         'OddManOut', 'CoordinationInversion']

    for task in tasks:
        get_feature_importances(task)


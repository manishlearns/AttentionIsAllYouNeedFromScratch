import os
def get_config():
    return {

        'batch_size': 8,
        'num_epochs': 20,
        'lr': 10**-4,
        'seq_len': 350,
        'd_model': 512,
        'lang_src': 'en',
        'lang_tgt': 'es',
        'tokenizer_file': 'tokenizer_{0}.json',
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': None,
        'experiment_name': 'runs/tmodel'
    }

def get_weights_file_path(config, epoch):

    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return os.path.join(model_folder, model_filename)
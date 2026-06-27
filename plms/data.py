import os
from os.path import join as pj
from datasets import load_dataset


__all__ = ('get_dataset', 'DEFAULT_CACHE_DIR')

DEFAULT_CACHE_DIR = pj(os.path.expanduser('~'), '.cache', 'plms')
# dataset requires custom reference file
DATA_NEED_CUSTOM_REFERENCE = ['shnl/qg-example']


def get_dataset(path: str = 'shnl/qg-example',
                name: str = 'default',
                split: str = 'train',
                input_type: str = 'paragraph',
                output_type: str = 'questions_answers',
                use_auth_token: bool = False):
    """ Get question generation input/output list of texts. """
    
    # Check if path is a local directory
    if os.path.isdir(path):
        print(f"Loading dataset from local directory: {path}, split: {split}")
        # Load from local JSONL files
        data_files = {}
        for s in ['train', 'validation', 'test']:
            jsonl = os.path.join(path, f'{s}.jsonl')
            if os.path.exists(jsonl):
                data_files[s] = jsonl
        # Fallback: dev.jsonl → validation
        if 'validation' not in data_files:
            dev_path = os.path.join(path, 'dev.jsonl')
            if os.path.exists(dev_path):
                data_files['validation'] = dev_path
        # Filter out missing files
        dataset_dict = load_dataset('json', data_files=data_files)
        # Get specific split
        if split in dataset_dict:
            dataset = dataset_dict[split]
        else:
            print(f"Split '{split}' not found, using 'train'")
            dataset = dataset_dict['train']
    else:
        # Load from Hugging Face Hub
        print(f"Loading dataset from Hub: {path}, split: {split}")
        name = None if name == 'default' else name
        kwargs = {'split': split}
        if use_auth_token:
            kwargs['token'] = use_auth_token if isinstance(use_auth_token, str) else True
        dataset = load_dataset(path, name, **kwargs)
    
    # Use column name directly if it exists, no magic mapping
    input_col = input_type if input_type in dataset.column_names else None
    output_col = output_type if output_type in dataset.column_names else None
    
    # Fallback mapping for backward compatibility
    FALLBACK = {
        'paragraph': 'context',
        'questions_answers': 'text_output',
    }
    if input_col is None:
        input_col = FALLBACK.get(input_type, input_type)
    if output_col is None:
        output_col = FALLBACK.get(output_type, output_type)
    
    print(f"  Using columns: input='{input_col}', output='{output_col}' (from {list(dataset.column_names)})")
    return dataset[input_col], dataset[output_col]

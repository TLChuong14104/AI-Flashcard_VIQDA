import os
from os.path import join as pj
from datasets import load_dataset


__all__ = ('get_dataset', 'DEFAULT_CACHE_DIR')

DEFAULT_CACHE_DIR = pj(os.path.expanduser('~'), '.cache', 'plms')
# dataset requires custom reference file
DATA_NEED_CUSTOM_REFERENCE = ['shnl/qg-example']

# Map input/output type names to actual column names for our dataset
COLUMN_MAPPING = {
    'paragraph': 'context',
    'paragraph_sentence': 'context',
    'paragraph_answer': 'context',  # Fallback to context for QA datasets
    'question': 'question',
    'answer': 'answer',
    'questions_answers': 'question',
}


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
        data_files = {
            'train': os.path.join(path, 'train.jsonl'),
            'validation': os.path.join(path, 'validation.jsonl'),
            'test': os.path.join(path, 'test.jsonl'),
        }
        # Filter out missing files
        data_files = {k: v for k, v in data_files.items() if os.path.exists(v)}
        dataset_dict = load_dataset('json', data_files=data_files)
        # Get specific split
        if split in dataset_dict:
            dataset = dataset_dict[split]
        else:
            # Fallback to train if split not found
            print(f"Split '{split}' not found, using 'train'")
            dataset = dataset_dict['train']
    else:
        # Load from Hugging Face Hub
        print(f"Loading dataset from Hub: {path}, split: {split}")
        name = None if name == 'default' else name
        # Support both use_auth_token (deprecated) and token (new parameter)
        kwargs = {'split': split}
        if use_auth_token:
            kwargs['token'] = use_auth_token if isinstance(use_auth_token, str) else True
        dataset = load_dataset(path, name, **kwargs)
    
    # Map logical column names to actual column names in dataset
    input_col = COLUMN_MAPPING.get(input_type, input_type)
    output_col = COLUMN_MAPPING.get(output_type, output_type)
    
    # Verify columns exist, if not try original names
    if input_col not in dataset.column_names:
        input_col = input_type
    if output_col not in dataset.column_names:
        output_col = output_type
    
    return dataset[input_col], dataset[output_col]

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

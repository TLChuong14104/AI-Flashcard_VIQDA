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
    name = None if name == 'default' else name
    # Support both use_auth_token (deprecated) and token (new parameter)
    kwargs = {'split': split}
    if use_auth_token:
        kwargs['token'] = use_auth_token if isinstance(use_auth_token, str) else True
    dataset = load_dataset(path, name, **kwargs)
    return dataset[input_type], dataset[output_type]

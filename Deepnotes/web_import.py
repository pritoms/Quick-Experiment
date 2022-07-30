import requests
import time
import pathlib
import importlib

def fetch_text(url):
    response = requests.get(url)
    return response.text

def import_from_web(url, module_name):
    text = fetch_text(url)
    path = pathlib.Path(module_name + '.py')
    path.write_text(text)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if __name__ == '__main__':
    url= 'https://raw.githubusercontent.com/pritoms/Quick-Experiment/main/bytes_tokenizer.py'
    module_name = 'bytes_tokenizer'
    module = import_from_web(url, module_name)
    print(module.__name__)
    print(module.__file__)
    print(module.__package__)
    print(module.__spec__)
    from bytes_tokenizer import *
    print(encode("This is a test"))

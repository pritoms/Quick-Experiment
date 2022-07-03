import numpy as np
import os
import time
import pickle
import sys
import math
import random
import copy
import argparse
import json
import subprocess
import threading

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_dir', type=str, default='experiments/base_model')
    parser.add_argument('--restore_file', type=str, default=None)
    parser.add_argument('--train_file', type=str, default='train.txt')
    parser.add_argument('--test_file', type=str, default='test.txt')
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_hid', type=int, default=2048)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--nlayers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--log_file_level', type=str, default='info')
    parser.add_argument('--log_stdout_level', type=str, default='info')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_params', action='store_true')
    parser.add_argument('--log_metrics', action='store_true')
    parser.add_argument('--log_code', action='store_true')
    parser.add_argument('--log_graph', action='store_true')
    parser.add_argument('--log_output', action='store_true')
    parser.add_argument('--log_profiling', action='store_true')
    parser.add_argument('--log_git', action='store_true')
    parser.add_argument('--log_env', action='store_true')
    parser.add_argument('--log_requirements', action='store_true')
    parser.add_argument('--log_docker', action='store_true')
    parser.add_argument('--log_nvidia', action='store_true')
    parser.add_argument('--log_hostname', action='store_true')
    parser.add_argument('--log_gpu', action='store_true')
    parser.add_argument('--log_cpu', action='store_true')
    parser.add_argument('--log_memory', action='store_true')
    parser.add_argument('--logging', type=str, default=None)

    args = parser.parse_args()

    return args
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_git_hash():
    try:
        git_log = subprocess.Popen('git log --pretty="%h" -n 1'.split(' '), stdout=subprocess.PIPE)
        (git_hash, _) = git_log.communicate()
        git_hash = git_hash.decode('ascii').strip()
        return git_hash
    except:
        return None


def get_git_diff():
    try:
        git_diff = subprocess.Popen('git diff'.split(' '), stdout=subprocess.PIPE)
        (diff, _) = git_diff.communicate()
        diff = diff.decode('ascii').strip()
        return diff
    except:
        return None


def get_requirements():
    try:
        requirements = subprocess.Popen('pip freeze'.split(' '), stdout=subprocess.PIPE)
        (requirements, _) = requirements.communicate()
        requirements = requirements.decode('ascii').strip()
        return requirements
    except:
        return None


def get_docker():
    try:
        docker = subprocess.Popen('cat /etc/os-release'.split(' '), stdout=subprocess.PIPE)
        (docker, _) = docker.communicate()
        docker = docker.decode('ascii').strip()
        return docker
    except:
        return None


def get_nvidia_smi():
    try:
        nvidia_smi = subprocess.Popen('nvidia-smi'.split(' '), stdout=subprocess.PIPE)
        (nvidia_smi, _) = nvidia_smi.communicate()
        nvidia_smi = nvidia_smi.decode('ascii').strip()
        return nvidia_smi
    except:
        return None


def get_hostname():
    try:
        hostname = subprocess.Popen('hostname'.split(' '), stdout=subprocess.PIPE)
        (hostname, _) = hostname.communicate()
        hostname = hostname.decode('ascii').strip()
        return hostname
    except:
        return None


def get_gpu():
    try:
        gpu = subprocess.Popen('nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free --format=csv'.split(' '), stdout=subprocess.PIPE)
        (gpu, _) = gpu.communicate()
        gpu = gpu.decode('ascii').strip()
        return gpu
    except:
        return None


def get_cpu():
    try:
        cpu = subprocess.Popen('cat /proc/cpuinfo'.split(' '), stdout=subprocess.PIPE)
        (cpu, _) = cpu.communicate()
        cpu = cpu.decode('ascii').strip()
        return cpu
    except:
        return None


def get_memory():
    try:
        memory = subprocess.Popen('cat /proc/meminfo'.split(' '), stdout=subprocess.PIPE)
        (memory, _) = memory.communicate()
        memory = memory.decode('ascii').strip()
        return memory
    except:
        return None


def get_logger(file_level='info', stdout_level='info'):
    logger = logging.getLogger()

    # Set file handler for logging to file
    log_file = os.path.join(args.log_dir, args.log_file)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))

    # Set stream handler for logging to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(message)s'))

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Set level for file and stream handlers
    file_handler.setLevel(file_level.upper())
    stream_handler.setLevel(stdout_level.upper())

    # Set level for the logger
    logger.setLevel(min(file_level.upper(), stdout_level.upper()))

    return logger


def log_params():
    logger = logging.getLogger()
    for key, value in vars(args).items():
        logger.info('{}: {}'.format(key, value))


def log_metrics():
    logger = logging.getLogger()
    for key, value in metrics.items():
        logger.info('{}: {}'.format(key, value))


def log_code():
    logger = logging.getLogger()
    logger.info('Code:')
    for filename in os.listdir('.'):
        if filename.endswith('.py'):
            with open(filename, 'r') as f:
                logger.info('Filename: {}'.format(filename))
                logger.info(f.read())


def log_graph():
    logger = logging.getLogger()
    logger.info('Graph:')
    logger.info(str(model))


def log_output():
    logger = logging.getLogger()
    logger.info('Output:')

    # Get a batch of data from the test set
    x, y = next(iter(test_loader))

    # Make predictions on the batch of data and get the loss
    y_pred = model(x, create_mask(x.size(1)))
    loss = criterion(y_pred, y)

    # Log the loss and predictions to the console
    logger.info('Loss: {}'.format(loss))
    for i in range(len(y)):
        logger.info('Input: {}'.format(decode(x[i])))
        logger.info('Target: {}'.format(decode(y[i])))
        logger.info('Prediction: {}'.format(decode(y_pred[i])))


def log_profiling():
    logger = logging.getLogger()
    logger.info('Profiling:')

    # Get a batch of data from the test set
    x, y = next(iter(test_loader))

    # Make predictions on the batch of data and get the loss
    y_pred = model(x, create_mask(x.size(1)))
    loss = criterion(y_pred, y)

    # Log the loss and predictions to the console
    logger.info('Loss: {}'.format(loss))

    # Start profiling
    pr = cProfile.Profile()
    pr.enable()

    # Make predictions on the batch of data and get the loss
    y_pred = model(x, create_mask(x.size(1)))
    loss = criterion(y_pred, y)

    # Stop profiling and print results to file
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    logger.info(s.getvalue())


def log_git():
    logger = logging.getLogger()
    logger.info('Git:')

    # Get the current git hash and diff
    git_hash = get_git_hash()
    git_diff = get_git_diff()

    # Log the current git hash and diff to the console
    logger.info('Git Hash: {}'.format(git_hash))
    logger.info('Git Diff: {}'.format(git_diff))


def log_env():
    logger = logging.getLogger()
    logger.info('Environment:')

    # Get the current environment variables and log them to the console
    for key, value in os.environ.items():
        logger.info('{}: {}'.format(key, value))


def log_requirements():
    logger = logging.getLogger()
    logger.info('Requirements:')

    # Get the current requirements and log them to the console
    requirements = get_requirements()
    logger.info(requirements)


def log_docker():
    logger = logging.getLogger()
    logger.info('Docker:')

    # Get the current docker information and log it to the console
    docker = get_docker()
    logger.info(docker)


def log_nvidia():
    logger = logging.getLogger()
    logger.info('Nvidia:')

    # Get the current nvidia-smi information and log it to the console
    nvidia_smi = get_nvidia_smi()
    logger.info(nvidia_smi)


def log_hostname():
    logger = logging.getLogger()
    logger.info('Hostname:')

    # Get the current hostname and log it to the console
    hostname = get_hostname()
    logger.info(hostname)


def log_gpu():
    logger = logging.getLogger()
    logger.info('GPU:')

    # Get the current GPU information and log it to the console
    gpu = get_gpu()
    logger.info(gpu)


def log_cpu():
    logger = logging.getLogger()
    logger.info('CPU:')

    # Get the current CPU information and log it to the console
    cpu = get_cpu()
    logger.info(cpu)


def log_memory():
    logger = logging.getLogger()
    logger.info('Memory:')

    # Get the current memory information and log it to the console
    memory = get_memory()
    logger.info(memory)

def get_logging_level(logging_level):
    if logging_level == 'debug':
        return logging.DEBUG
    elif logging_level == 'info':
        return logging.INFO
    elif logging_level == 'warning':
        return logging.WARNING
    elif logging_level == 'error':
        return logging.ERROR
    elif logging_level == 'critical':
        return logging.CRITICAL

def get_logging_config():
    config = {'params': args.log_params, 'metrics': args.log_metrics, 'code': args.log_code, 'graph': args.log_graph, 'output': args.log_output, 'profiling': args.log_profiling, 'git': args.log_git, 'env': args.log_env, 'requirements': args.log_requirements, 'docker': args.log_docker, 'nvidia': args.log_nvidia, 'hostname': args.log_hostname, 'gpu': args.log_gpu, 'cpu': args.log_cpu, 'memory': args.log_memory}
    return config

def get_metrics(config):
    metrics = {}
    if config['params']:
        metrics['params'] = vars(args)
    if config['metrics']:
        metrics['metrics'] = {}
    if config['code']:
        metrics['code'] = {}
    if config['graph']:
        metrics['graph'] = str(model)
    if config['output']:
        metrics['output'] = {}
    if config['profiling']:
        metrics['profiling'] = {}
    if config['git']:
        metrics['git'] = {'hash': get_git_hash(), 'diff': get_git_diff()}
    if config['env']:
        metrics['env'] = dict(os.environ)
    if config['requirements']:
        metrics['requirements'] = get_requirements()
    if config['docker']:
        metrics['docker'] = get_docker()
    if config['nvidia']:
        metrics['nvidia'] = get_nvidia_smi()
    if config['hostname']:
        metrics['hostname'] = get_hostname()
    if config['gpu']:
        metrics['gpu'] = get_gpu()
    if config['cpu']:
        metrics['cpu'] = get_cpu()
    if config['memory']:
        metrics['memory'] = get_memory()

    return metrics
    
def log(config, metrics):
    logger = logging.getLogger()

    # Log parameters to the console and file
    if config['params']:
        logger.info('Parameters:')
        for key, value in vars(args).items():
            logger.info('{}: {}'.format(key, value))

    # Log metrics to the console and file
    if config['metrics']:
        logger.info('Metrics:')
        for key, value in metrics.items():
            logger.info('{}: {}'.format(key, value))

    # Log code to the console and file
    if config['code']:
        logger.info('Code:')
        for filename in os.listdir('.'):
            if filename.endswith('.py'):
                with open(filename, 'r') as f:
                    logger.info('Filename: {}'.format(filename))
                    logger.info(f.read())

    # Log graph to the console and file
    if config['graph']:
        logger.info('Graph:')
        logger.info(str(model))

    # Log output to the console and file
    if config['output']:
        logger.info('Output:')

        # Get a batch of data from the test set
        x, y = next(iter(test_loader))

        # Make predictions on the batch of data and get the loss
        y_pred = model(x, create_mask(x.size(1)))
        loss = criterion(y_pred, y)

        # Log the loss and predictions to the console
        logger.info('Loss: {}'.format(loss))
        for i in range(len(y)):
            logger.info('Input: {}'.format(decode(x[i])))
            logger.info('Target: {}'.format(decode(y[i])))
            logger.info('Prediction: {}'.format(decode(y_pred[i])))

    # Log profiling to the console and file
    if config['profiling']:
        logger.info('Profiling:')

        # Get a batch of data from the test set
        x, y = next(iter(test_loader))

        # Make predictions on the batch of data and get the loss
        y_pred = model(x, create_mask(x.size(1)))
        loss = criterion(y_pred, y)

        # Log the loss and predictions to the console
        logger.info('Loss: {}'.format(loss))

        # Start profiling
        pr = cProfile.Profile()
        pr.enable()

        # Make predictions on the batch of data and get the loss
        y_pred = model(x, create_mask(x.size(1)))
        loss = criterion(y_pred, y)

        # Stop profiling and print results to file
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        logger.info(s.getvalue())

    # Log git information to the console and file
    if config['git']:
        logger.info('Git:')

        # Get the current git hash and diff
        git_hash = get_git_hash()
        git_diff = get_git_diff()

        # Log the current git hash and diff to the console
        logger.info('Git Hash: {}'.format(git_hash))
        logger.info('Git Diff: {}'.format(git_diff))

    # Log environment variables to the console and file
    if config['env']:
        logger.info('Environment:')

        # Get the current environment variables and log them to the console
        for key, value in os.environ.items():
            logger.info('{}: {}'.format(key, value))

    # Log requirements to the console and file
    if config['requirements']:
        logger.info('Requirements:')

        # Get the current requirements and log them to the console
        requirements = get_requirements()
        logger.info(requirements)

    # Log docker information to the console and file
    if config['docker']:
        logger.info('Docker:')

        # Get the current docker information and log it to the console
        docker = get_docker()
        logger.info(docker)

    # Log nvidia-smi information to the console and file
    if config['nvidia']:
        logger.info('Nvidia:')

        # Get the current nvidia-smi information and log it to the console
        nvidia_smi = get_nvidia_smi()
        logger.info(nvidia_smi)

    # Log hostname to the console and file
    if config['hostname']:
        logger.info('Hostname:')

        # Get the current hostname and log it to the console
        hostname = get_hostname()
        logger.info(hostname)

    # Log GPU information to the console and file
    if config['gpu']:
        logger.info('GPU:')

        # Get the current GPU information and log it to the console
        gpu = get_gpu()
        logger.info(gpu)

    # Log CPU information to the console and file
    if config['cpu']:
        logger.info('CPU:')

        # Get the current CPU information and log it to the console
        cpu = get_cpu()
        logger.info(cpu)

    # Log memory information to the console and file
    if config['memory']:
        logger.info('Memory:')

        # Get the current memory information and log it to the console
        memory = get_memory()
        logger.info(memory)
import subprocess
import time
import threading
import sys 

class Process(object):
    def __init__(self, cmd, env=None, cwd=None):
        self._cmd = cmd
        self._env = env
        self._cwd = cwd
        self._proc = None

    def start(self):
        if self._proc is None:
            self._proc = subprocess.Popen(self._cmd, env=self._env, cwd=self._cwd)

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait()
            self._proc = None

    def wait(self):
        if self._proc is not None:
            self._proc.wait()

    @property
    def pid(self):
        return self._proc.pid if self._proc else None

    @property
    def returncode(self):
        return self._proc.returncode if self._proc else None

    @property
    def is_alive(self):
        return self.returncode is None

class ProcessManager(object):
    def __init__(self, processes):
        self._processes = processes

    def __enter__(self):
        start_processes(self._processes)

    def __exit__(self, type, value, traceback):
        stop_processes(self._processes)
        wait_processes(self._processes)

class ProcessResult(object):
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    @property
    def success(self):
        return self.returncode == 0

    @property
    def failure(self):
        return not self.success

    def check_success(self):
        if not self.success:
            raise Exception('Process failed with return code %d' % self.returncode)

    def check_failure(self):
        if not self.failure:
            raise Exception('Process succeeded with return code %d' % self.returncode)
            
if __name__ == '__main__':
    cmd = ['python', '-c', 'import numpy as np; print("Multiplying matrices..."); a = np.random.rand(1000, 1000); b = np.random.rand(1000, 1000); c = a.dot(b); print("Done!"); print(c)']
    p = Process(cmd)
    with ProcessManager([p]):
        print('Process pid: %d' % p.pid)
        while p.is_alive:
            time.sleep(.1)
            print('.', end='')
            sys.stdout.flush()
        print('\nProcess finished with return code %d' % p.returncode)

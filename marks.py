#!/usr/bin/env pythos2

"""
Shim marks.py for executing grum.py tests independent of moss.

Features:
 - Wildcard matching of test names (using Bash's *, ?, []).
 - More verbose output (up to -vvv), including full diffs.
 - Delay time multiplier to adjust delays by a factor between 0 and 1.
 - Displays all failure conditions for a test, not just the first failure.
 - Compatible with current grum.py test scripts without modification.

Known issues:
 - No timeouts implemented, programs can block the tester.
"""

# WARNING: grum.py is python 2.7
from __future__ import print_function, generators, with_statement, generators
import fnmatch
import random
import os
import sys
import time
import logging
import signal
import datetime

from pprint import pprint
import warnings
from warnings import warn
import subprocess
import difflib
from collections import defaultdict
from pipes import quote
import argparse

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

delay = 1 # delay time coefficient

class Colour:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    REVERSE = '\033[7m'
    GREY = '\033[38;5;8m'
    CYAN = '\033[38;5;14m'
    MAGENTA = '\033[38;5;13m'
    ORANGE = '\033[38;5;208m'
    YELLOW = '\033[38;5;11m'

def _diff_files(f1, f2, s1, s2=None, t1=None, t2=None):
    if t1 is None:
        t1 = os.path.getmtime(s1)
        t1 = str(datetime.datetime.fromtimestamp(t1))
    if t2 is None:
        t2 = os.path.getmtime(s2) if s2 else time.time()
        t2 = str(datetime.datetime.fromtimestamp(t2))
    if s2 is None:
        s2 = '(actual)'
    l = max(len(s1), len(s2))
    s1 = s1.ljust(l)
    s2 = s2.ljust(l)
    diff = difflib.unified_diff(f1, f2, s1, s2, t1, t2)
    return list(diff)

class TestProcess(object):

    @staticmethod
    def _tee_stream(stream, log):
        while True:
            line = next(stream)
            yield line
            log.append(line)

    def __init__(self, i, args, stdin):
        logger.debug('starting process {} with stdin {}'
                .format(i, repr(stdin)))
        logger.debug('    cmd: {}'.format(args))

        if stdin:
            stdin = open(stdin, 'r')
            self.stdin_log = list(stdin.readlines())
            stdin.seek(0)
        else:
            stdin = subprocess.PIPE
            self.stdin_log = []

        self.i = i
        self.args = args
        self.process = subprocess.Popen(args, stdout=subprocess.PIPE,
            stdin=stdin, stderr=subprocess.PIPE)

        self.stdout_log = []
        self.stderr_log = []
        self.stdout_diff_log = []
        self.stderr_diff_log = []

    def save_streams(self, prefix=''):
        # read remaining data from streams
        self.stdout_log.extend(self.process.stdout.readlines())
        self.stderr_log.extend(self.process.stderr.readlines())

        stream_names = ('in', 'out', 'err', 'diff.out', 'diff.err')
        streams = (self.stdin_log, self.stdout_log, self.stderr_log,
                self.stdout_diff_log, self.stderr_diff_log)
        for name, log in zip(stream_names, streams):
            fname = '{}_{}.{}'.format(self.i, self.args[0].replace('./', ''),
                    name)
            path = prefix + fname
            try:
                os.makedirs(os.path.dirname(path))
            except OSError:
                pass # directory could exist
            with open(path, 'w') as f:
                f.writelines(log)

    def send(self, msg):
        logger.debug('writing to stdin of process {}: {}'.format(self.i, repr(msg)))
        self.stdin_log.append(msg)
        self.process.stdin.write(msg)
        self.process.stdin.flush()

    def finish_input(self):
        logger.debug('closing stdin of process {}'.format(self.i))
        if self.process.stdin:
            self.process.stdin.close()

    def readline_stdout(self):
        line = self.process.stdout.readline()
        self.stdout_log.append(line)
        logger.debug('got line {} from process {}'.format(repr(line), self.i))
        #print('read', line)
        return line

    def send_signal(self, sig):
        logger.debug('sending signal {} to process {}'.format(sig, self.i))
        self.process.send_signal(sig)

    def diff_fd(self, fd, comparison):
        if fd == 1:
            f = self.process.stdout
            log = self.stdout_log
            dlog = self.stdout_diff_log
        elif fd == 2:
            f = self.process.stderr
            log = self.stderr_log
            dlog = self.stderr_diff_log
        fname = ['stdin', 'stdout', 'stderr'][fd]
        out = f.readlines()
        log.extend(out)
        dlog.extend(out)
        with open(comparison, 'r') as compare:
            expected = compare.readlines()

            diff = _diff_files(expected, out, comparison)
            if diff:
                logger.error('{} mismatch on process {} {}'.format(fname, self.i,
                    self.args))
                logger.warning('\n'+''.join(diff).rstrip())
            else:
                logger.info('{} matched on process {} to file {}'
                        .format(fname, self.i, repr(comparison)))

    _signal_names = {getattr(signal, s): s for s in dir(signal)
            if s.startswith('SIG') and not s.startswith('SIG_')}

    @classmethod
    def _sig_name(cls, code):
        pcode = code if code >= 0 else -code
        s = 'signal ' if code < 0 else 'exit code '
        s += str(pcode)
        if code < 0 and pcode in cls._signal_names:
            s += ' ('+cls._signal_names[pcode]+')'
        return s


    def expect_exit(self, code):
        ret = self.process.wait()
        if ret != code:
            logger.error('process {} expected {} but got {}'
                    .format(self.i, self._sig_name(code), self._sig_name(ret)))
        else:
            logger.info('process {} exited correctly with {}'
                    .format(self.i, self._sig_name(code)))

    def assert_signalled(self, sig):
        self.expect_exit(-sig)

    def kill(self):
        logger.debug('killing process {}'.format(self.i))
        self.process.kill()


class TestCase(object):
    def __init__(self):
        self.i = 1
        self.processes = []

    def process(self, args, stdin=None):
        p =  TestProcess(self.i, args, stdin)
        self.i += 1
        self.processes.append(p)
        return p

    def _save_process_streams(self, prefix):
        for p in self.processes:
            try: p.kill()
            except OSError: pass # process already dead
            p.save_streams(prefix)
    
    def delay(self, t):
        scale = '' if delay == 1 else ' (scaled to '+str(t*delay)+')'
        logger.debug('sleeping for {} seconds{}'.format(t, scale))
        time.sleep(t*delay)

    def assert_stdout_matches_file(self, proc, f):
        proc.diff_fd(1, f)
    def assert_stderr_matches_file(self, proc, f):
        proc.diff_fd(2, f)
    def assert_exit_status(self, proc, status):
        proc.expect_exit(status)

    def assert_files_equal(self, s1, s2):
        with open(s1, 'r') as f1, open(s2, 'r') as f2:
            diff = _diff_files(f2.readlines(), f1.readlines(), s2, s1)
            if diff:
                logger.error('actual {} does not match expected {}'
                        .format(repr(s1), repr(s2)))
                logger.warning(''.join(diff))
            else:
                logger.info('file {} matches expected {}'
                        .format(repr(s1), repr(s2)))

def marks(category=None, category_marks=0):
    def wrapper(f):
        f.marks = {'category': category,
            'marks': category_marks}
        return f;
    return wrapper


def eprint(*args, **kwargs):
    kwargs['file'] = sys.stderr
    print(*args, **kwargs)

def _catch_warnings(display):
    caught = []
    original = warnings.showwarning

    def showwarning(*args, **kwargs):
        caught.append(args[0])
        if display:
            return original(*args, **kwargs)
    warnings.showwarning = showwarning
    return caught

class CaptureHandler(logging.NullHandler):
    def __init__(self):
        super(logging.NullHandler, self).__init__()
        self.records = []

    def handle(self, record):
        self.records.append(record)

def nonneg_float(s):
    x = float(s)
    if x < 0:
        raise argparse.ArgumentTypeError('value cannot be negative: ' + s)
    return x

def parse_args(args):
    parser = argparse.ArgumentParser(description='Test runner for CSSE2310. '
            +'Shim marks.py by Kenton Lam.')
    parser.add_argument('-l', '--list', action='store_true',
            help='list matching tests without running them.')
    parser.add_argument('-v', '--verbose', action='count', default=0,
            help='verbosity, can be repeated up to 3 times. '
            +'once prints diffs, twice prints successes, thrice prints all actions.')
    parser.add_argument('-d', '--delay', action='store', type=nonneg_float, default=1,
            help='delay time multiplier.')
    parser.add_argument('-s', '--save', action='store_true',
            help='save test input and output streams to testres/.')
    parser.add_argument('--debug', action='store_true',
            help='exit with full stack trace on exceptions.')
    parser.add_argument('test', metavar='TEST', type=str, nargs='*', default=('*',),
            help='tests to run. can contain Bash-style wildcards. default: *')
    return parser.parse_args(args)

def main():
    global delay
    tests = {}
    test_names = []
    for cls in TestCase.__subclasses__():
        for fn in dir(cls):
            fn = getattr(cls, fn)
            if not hasattr(fn, 'marks'): continue
            name = cls.__name__+'.'+fn.__name__
            tests[name] = (cls, fn)
            test_names.append(name)
    
    args = (parse_args(sys.argv[1:]))
    list_ = args.list
    delay = args.delay
    verbose = min(args.verbose, 3)
    debug = args.debug
    save = args.save and not list_

    failure_level = logging.ERROR
    diff_level = logging.WARNING
    success_level = logging.INFO
    display_level = [logging.ERROR, logging.WARNING, logging.INFO,
            logging.DEBUG][verbose]

    level_names = {
        failure_level: Colour.WARNING+'failure',
        success_level: Colour.OKBLUE+'success',
        diff_level: 'diff',
        logging.CRITICAL: Colour.WARNING+Colour.REVERSE+'error'
    }

    handler = CaptureHandler()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    matched = []
    seen = set()
    for t in args.test:
        new_matches = [x for x in fnmatch.filter(test_names, '*'+t+'*')
                if x not in seen]
        matched.extend(new_matches)
        seen.update(new_matches)

    print('matched', len(matched), 'tests.')
    print()

    max_len = max(len(x) for x in matched) if matched else 0
    num_len = len(str(len(matched)))

    passed = 0
    failed = 0
    start = time.time()
    n = 0
    for name in matched:
        n += 1
        test_num = '[{}/{}]'.format(n, len(matched))
        test_num = test_num.rjust(2*num_len+3)
        print(Colour.MAGENTA+test_num+Colour.ENDC,
                name.ljust(max_len), '...', end=' ')
        sys.stdout.flush()
        del handler.records[:]

        cls, fn = tests[name]
        cls.__marks_options__ = {'working_dir': ''}
        try:
            cls.setup_class() # needed to set cls.prog
        except (NameError, OSError) as e:
            pass
            # throws due to missing TEST_LOCATION or invalid working dir

        interrupted = False
        c = cls()
        try:
            if not list_:
                fn(c) # run the test
            if save: c._save_process_streams('testres/'+name+'/')
        except Exception as e:
            if debug: raise # throw the exception again
            logger.critical('test could not continue due to exception:\n'+e.__class__.__name__+': '+str(e))
        except KeyboardInterrupt as e:
            if debug: raise
            interrupted = True
            #print(Colour.YELLOW+'INTERRUPTED', end=' ')
            logger.critical(e.__class__.__name__)
        else:
            if list_:
                pass # don't count pass/fail if -l was passed
            elif any(r for r in handler.records if r.levelno >= failure_level):
                failed += 1
                print(Colour.FAIL+'FAIL', end='')
            else:
                passed += 1
                print(Colour.OKGREEN+'OK', end='')
        print(Colour.ENDC)

        for i, r in enumerate(handler.records):
            if r.levelno < display_level: continue
            name = r.levelname.lower()
            if r.levelno in level_names: name = level_names[r.levelno]
            print(name+':'+Colour.ENDC, r.msg)

        if interrupted:
            print('\nterminating due to interrupt.')
            break
    print()
    if save:
        print(Colour.OKGREEN+'output saved:'+Colour.ENDC, 'test outputs are saved to testres folder.\n')
    if list_:
        print(Colour.OKBLUE+'list tests:'+Colour.ENDC, 'tests displayed. no tests were run.\n')
    print('ran', passed+failed, 'tests in', round(time.time()-start, 2), 'seconds.',
            'passed:', str(passed)+',', 'failed:', str(failed)+'.')
    print(Colour.ORANGE+'warning:',
            'this is not an official test and offers no guarantee of correctness.'+Colour.ENDC)

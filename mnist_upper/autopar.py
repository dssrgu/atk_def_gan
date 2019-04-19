#!/usr/bin/python3

from __future__ import print_function

import base64
import datetime
import os
import shutil
import subprocess
import sys

import argparse
import dateutil.tz

CONTENT_WORKSPACE = r'''
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
git_repository(
    name = "subpar",
    remote = "https://github.com/google/subpar",
    tag = "1.0.0",
)
'''
def _get_workspace_file_content():
    return CONTENT_WORKSPACE

TEMPLATE_BUILD = r'''
load("@subpar//:subpar.bzl", "par_binary")
par_binary(
    name = "{target_name}",
    python_version = "{python_version}",
    srcs = glob(["**/*.py"]),
    main = "{main_file}",
    imports = {import_paths}
)
'''
def _get_build_file_content(import_paths=['.'], **kwargs):
    kwargs['import_paths'] = '[' + ', '.join('"{}"'.format(p) for p in import_paths) + ']'
    return TEMPLATE_BUILD.format(**kwargs)

TEMPLATE_MAIN_FILE_WRAPPER = r'''
import base64
import os
import sys
import zipimport

TOP_PATH = r'{top_path}'
MAIN_FILE = r'{main_file}'
MAIN_FILE_CONTENT = r'{main_file_content}'

vars_to_decode = ['TOP_PATH', 'MAIN_FILE', 'MAIN_FILE_CONTENT']
for vk in vars_to_decode:
    globals()[vk] = base64.b64decode(globals()[vk])
    if sys.version_info >= (3, 0):
        globals()[vk] = globals()[vk].decode('utf-8')

par_virtual_main_path = os.path.join(os.path.dirname(__file__), '__main__')
par_virtual_main_path_s = os.path.join(par_virtual_main_path, '')
__file__ = os.path.join(TOP_PATH, MAIN_FILE)

original_finder_creator_type = zipimport.zipimporter

class MockFinder(object):
    def __init__(self, finder):
        for k, v in finder._files.items():
            if finder._files[k][0].startswith(par_virtual_main_path_s):
                new_path = os.path.join(TOP_PATH, finder._files[k][0].replace(par_virtual_main_path_s, ''))
                finder._files[k] = (new_path,) + finder._files[k][1:]
        self._finder = finder

    def __getattr__(self, attr):
        return getattr(self._finder, attr)

class MockFinderCreater(object):
    def __init__(self, *args, **kwargs):
        self._finder_creator = original_finder_creator_type(*args, **kwargs)

    def __getattr__(self, attr):
        if attr == 'find_loader':
            return getattr(self, attr)
        return getattr(self._finder_creator, attr)

    def find_loader(self, *args, **kwargs):
        return_value = self._finder_creator.find_loader(*args, **kwargs)
        return (MockFinder(return_value[0]), return_value[1])

for k, v in sys.path_importer_cache.items():
    if k == par_virtual_main_path or k.startswith(par_virtual_main_path_s):
        sys.path_importer_cache[k] = MockFinder(v)

for i, v in enumerate(sys.path_hooks):
    if v == original_finder_creator_type:
        sys.path_hooks[i] = MockFinderCreater

exec(MAIN_FILE_CONTENT)
'''
def _get_main_file_wrapper_content(**kwargs):
    for k, v in kwargs.items():
        kwargs[k] = base64.b64encode(bytes(v, 'utf-8')).decode('utf-8')
    return TEMPLATE_MAIN_FILE_WRAPPER.format(**kwargs)

def _get_current_timestamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y%m%d-%H%M%S-%f')
    return timestamp

VALID_PYTHON_VERSIONS = ['PY2', 'PY3']
def make_par(main_file,
             python_version='PY3',
             label=None,
             import_paths=['.'],
             fix_file_vars=True,
             keep_bazel_build_daemon=True):
    if main_file is None or not os.path.exists(main_file):
        raise Exception('{} does not exist!'.format(main_file))

    target_name, main_file_ext = os.path.splitext(os.path.basename(main_file))
    if main_file_ext.lower() != '.py':
        raise Exception('main_file ({}) is not a .py file!'.format(main_file))

    if python_version not in VALID_PYTHON_VERSIONS:
        raise Exception('Invalid python_version ({})! Must be one of {}.'.format(
                python_version, VALID_PYTHON_VERSIONS))

    # For safety and simplicity, use the current directory as top_dir.
    top_dir = '.'
    top_dir = os.path.abspath(top_dir)

    if os.path.commonprefix([os.path.join(top_dir, ''), os.path.abspath(main_file)]) != os.path.join(top_dir, ''):
        raise Exception('main_file ({}) is not in the current directory tree!'.format(main_file))

    main_file = os.path.relpath(os.path.abspath(main_file), top_dir)
    main_file_orig = main_file
    os.chdir(top_dir)

    if fix_file_vars:
        with open(main_file, 'r') as f:
            main_file_wrapper_content = _get_main_file_wrapper_content(
                    top_path=top_dir,
                    main_file=main_file,
                    main_file_content=f.read())
        main_file = '__par_main_wrapper__.py'
        with open(main_file, 'w') as f:
            f.write(main_file_wrapper_content)

    with open('WORKSPACE', 'w') as f:
        f.write(_get_workspace_file_content())
    with open('BUILD', 'w') as f:
        f.write(_get_build_file_content(
            target_name=target_name,
            python_version=python_version,
            main_file=main_file,
            import_paths=import_paths,
        ))

    p = subprocess.Popen(['bazel', 'clean'], stderr=subprocess.PIPE)
    _, stderr_data = p.communicate()
    if not isinstance(stderr_data, str):
        stderr_data = stderr_data.decode('utf-8')
    if p.returncode != 0:
        print(stderr_data, file=sys.stderr)
        raise Exception("'bazel clean' failed!")

    par_filename = '{}.par'.format(target_name)
    p = subprocess.Popen(['bazel', 'build', par_filename], stderr=subprocess.PIPE)
    _, stderr_data = p.communicate()
    if not isinstance(stderr_data, str):
        stderr_data = stderr_data.decode('utf-8')

    generated_par_file = None
    for l in stderr_data.splitlines():
        candidate = l.strip()
        if os.path.basename(candidate) == par_filename and os.path.exists(candidate):
            generated_par_file = candidate
            break
    if p.returncode != 0 or generated_par_file is None:
        print(stderr_data, file=sys.stderr)
        raise Exception('Par file not found!')

    out_filename = '{}.{}'.format(par_filename, _get_current_timestamp())
    if label is not None:
        out_filename += '.'
        out_filename += label

    out_file = os.path.join(os.path.split(main_file_orig)[0], out_filename)
    shutil.copy2(generated_par_file, out_file)

    if not keep_bazel_build_daemon:
        p = subprocess.Popen(['bazel', 'shutdown'])
        p.wait()

    return out_file


"""
This file is part of Sherpa Federated Learning Framework.

Sherpa Federated Learning Framework is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 3.

Sherpa Federated Learning Framework is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import print_function
from __future__ import unicode_literals

import re
import inspect
import os
import shutil
import six

try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib

from structure import PAGES


shfl_dir = pathlib.Path(__file__).resolve().parents[1]


def get_function_signature(function, method=True):
    wrapped = getattr(function, '_original_function', None)
    if wrapped is None:
        signature = inspect.getfullargspec(function)
    else:
        signature = inspect.getfullargspec(wrapped)
    defaults = signature.defaults
    if method:
        args = signature.args[1:]
    else:
        args = signature.args
    if defaults:
        kwargs = zip(args[-len(defaults):], defaults)
        args = args[:-len(defaults)]
    else:
        kwargs = []
    st = '%s.%s(' % (clean_module_name(function.__module__), function.__name__)

    for a in args:
        st += str(a) + ', '
    for a, v in kwargs:
        if isinstance(v, str):
            v = '\'' + v + '\''
        st += str(a) + '=' + str(v) + ', '
    if kwargs or args:
        signature = st[:-2] + ')'
    else:
        signature = st + ')'
    return post_process_signature(signature)


def get_class_signature(cls):
    try:
        class_signature = get_function_signature(cls.__init__)
        class_signature = class_signature.replace('__init__', cls.__name__)
    except (TypeError, AttributeError):
        # in case the class inherits from object and does not
        # define __init__
        class_signature = "{clean_module_name}.{cls_name}()".format(
            clean_module_name=cls.__module__,
            cls_name=cls.__name__
        )
    return post_process_signature(class_signature)


def post_process_signature(signature):

    return signature


def clean_module_name(name):

    return name


def class_to_source_link(cls):
    module_name = clean_module_name(cls.__module__)
    path = module_name.replace('.', '/')
    path += '.py'
    line = inspect.getsourcelines(cls)[-1]
    link = ('https://github.com/sherpaai/'
            'Sherpa.ai-Federated-Learning-Framework/blob/master/' + path + '#L' + str(line))

    return '<a href="' + link + '" target="_blank">[source]</a>'


def code_snippet(snippet):
    result = '```python\n'
    result += snippet.encode('unicode_escape').decode('utf8') + '\n'
    result += '```\n'
    return result


def count_leading_spaces(s):
    ws = re.search(r'\S', s)
    if ws:
        return ws.start()
    else:
        return 0


def process_list_block(docstring, starting_point, section_end,
                       leading_spaces, marker):
    ending_point = docstring.find('\n\n', starting_point)
    block = docstring[starting_point:
                      (ending_point - 1 if ending_point > -1
                       else section_end)]
    # Place marker for later reinjection.
    docstring_slice = docstring[
        starting_point:section_end].replace(block, marker)
    docstring = (docstring[:starting_point] +
                 docstring_slice +
                 docstring[section_end:])
    lines = block.split('\n')
    # Remove the computed number of leading white spaces from each line.
    lines = [re.sub('^' + ' ' * leading_spaces, '', line) for line in lines]
    # Usually lines have at least 4 additional leading spaces.
    # These have to be removed, but first the list roots have to be detected.
    top_level_regex = r'^    ([^\s\\\(]+):(.*)'
    top_level_replacement = r'- __\1__:\2'
    lines = [re.sub(top_level_regex, top_level_replacement, line)
             for line in lines]
    # All the other lines get simply the 4 leading space (if present) removed
    lines = [re.sub(r'^    ', '', line) for line in lines]
    # Fix text lines after lists
    indent = 0
    text_block = False
    for i in range(len(lines)):
        line = lines[i]
        spaces = re.search(r'\S', line)
        if spaces:
            # If it is a list element
            if line[spaces.start()] == '-':
                indent = spaces.start() + 1
                if text_block:
                    text_block = False
                    lines[i] = '\n' + line
            elif spaces.start() < indent:
                text_block = True
                indent = spaces.start()
                lines[i] = '\n' + line
        else:
            text_block = False
            indent = 0
    block = '\n'.join(lines)
    return docstring, block


def process_docstring(docstring):
    # First, extract code blocks and process them.
    code_blocks = []
    if '```' in docstring:
        tmp = docstring[:]
        while '```' in tmp:
            tmp = tmp[tmp.find('```'):]
            index = tmp[3:].find('```') + 6
            snippet = tmp[:index]
            # Place marker in docstring for later reinjection.
            docstring = docstring.replace(
                snippet, '$CODE_BLOCK_%d' % len(code_blocks))
            snippet_lines = snippet.split('\n')
            # Remove leading spaces.
            num_leading_spaces = snippet_lines[-1].find('`')
            snippet_lines = ([snippet_lines[0]] +
                             [line[num_leading_spaces:]
                             for line in snippet_lines[1:]])
            # Most code snippets have 3 or 4 more leading spaces
            # on inner lines, but not all. Remove them.
            inner_lines = snippet_lines[1:-1]
            leading_spaces = None
            for line in inner_lines:
                if not line or line[0] == '\n':
                    continue
                spaces = count_leading_spaces(line)
                if leading_spaces is None:
                    leading_spaces = spaces
                if spaces < leading_spaces:
                    leading_spaces = spaces
            if leading_spaces:
                snippet_lines = ([snippet_lines[0]] +
                                 [line[leading_spaces:]
                                  for line in snippet_lines[1:-1]] +
                                 [snippet_lines[-1]])
            snippet = '\n'.join(snippet_lines)
            code_blocks.append(snippet)
            tmp = tmp[index:]

    # Format docstring lists.
    section_regex = r'\n( +)# (.*)\n'
    section_idx = re.search(section_regex, docstring)
    shift = 0
    sections = {}
    while section_idx and section_idx.group(2):
        anchor = section_idx.group(2)
        leading_spaces = len(section_idx.group(1))
        shift += section_idx.end()
        next_section_idx = re.search(section_regex, docstring[shift:])
        if next_section_idx is None:
            section_end = -1
        else:
            section_end = shift + next_section_idx.start()
        marker = '$' + anchor.replace(' ', '_') + '$'
        docstring, content = process_list_block(docstring,
                                                shift,
                                                section_end,
                                                leading_spaces,
                                                marker)
        sections[marker] = content
        # `docstring` has changed, so we can't use `next_section_idx` anymore
        # we have to recompute it
        section_idx = re.search(section_regex, docstring[shift:])

    # Format docstring section titles.
    docstring = re.sub(r'\n(\s+)# (.*)\n',
                       r'\n\1__\2__\n\n',
                       docstring)

    # Strip all remaining leading spaces.
    lines = docstring.split('\n')
    docstring = '\n'.join([line.lstrip(' ') for line in lines])

    # Reinject list blocks.
    for marker, content in sections.items():
        docstring = docstring.replace(marker, content)

    # Reinject code blocks.
    for i, code_block in enumerate(code_blocks):
        docstring = docstring.replace(
            '$CODE_BLOCK_%d' % i, code_block)
    return docstring


def read_file(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def collect_class_methods(cls, methods):
    if isinstance(methods, (list, tuple)):
        return [getattr(cls, m) if isinstance(m, str) else m for m in methods]
    methods = []
    for _, method in inspect.getmembers(cls, predicate=inspect.isroutine):
        methods.append(method)
    return methods


def render_function(function, method=True):
    subblocks = []
    signature = get_function_signature(function, method=method)
    if method:
        signature = signature.replace(
            clean_module_name(function.__module__) + '.', '')
        subblocks.append('### ' + function.__name__ + '\n')
    else:
        subblocks.append('## ' + function.__name__ + '\n')
    subblocks.append(code_snippet(signature))
    docstring = function.__doc__
    if docstring:
        subblocks.append(process_docstring(docstring))
    return '\n\n'.join(subblocks)


def read_page_data(page_data, type):
    assert type in ['classes', 'functions', 'methods']
    data = page_data.get(type, [])
    for module in page_data.get('all_module_{}'.format(type), []):
        module_data = []
        for name in dir(module):
            module_member = getattr(module, name)
            if (inspect.isclass(module_member) and type == 'classes' or
               inspect.isfunction(module_member) and type == 'functions'):
                instance = module_member
                if module.__name__ in instance.__module__:
                    if instance not in module_data:
                        module_data.append(instance)
        module_data.sort(key=lambda x: id(x))
        data += module_data
    return data


def get_module_docstring(filepath):
    """Extract the module docstring.
    Also finds the line at which the docstring ends.
    """
    co = compile(open(filepath, encoding='utf-8').read(), filepath, 'exec')
    if co.co_consts and isinstance(co.co_consts[0], six.string_types):
        docstring = co.co_consts[0]
    else:
        print('Could not get the docstring from ' + filepath)
        docstring = ''
    return docstring, co.co_firstlineno


def copy_examples(examples_dir, destination_dir):
    """Copy the examples directory in the docs.
    Prettify files by extracting the docstrings written in Markdown.
    """
    pathlib.Path(destination_dir).mkdir(exist_ok=True)
    for file in os.listdir(examples_dir):
        if not file.endswith('.py'):
            continue
        module_path = os.path.join(examples_dir, file)
        docstring, starting_line = get_module_docstring(module_path)
        destination_file = os.path.join(destination_dir, file[:-2] + 'md')
        with open(destination_file, 'w+', encoding='utf-8') as f_out, \
                open(os.path.join(examples_dir, file),
                     'r+', encoding='utf-8') as f_in:

            f_out.write(docstring + '\n\n')

            # skip docstring
            for _ in range(starting_line):
                next(f_in)

            f_out.write('```python\n')
            # next line might be empty.
            line = next(f_in)
            if line != '\n':
                f_out.write(line)

            # copy the rest of the file.
            for line in f_in:
                f_out.write(line)
            f_out.write('```')


def generate(sources_dir):
    """Generates the markdown files for the docs.
    # Arguments
        sources_dir: Where to put the markdown files.
    """
    template_dir = os.path.join(str(shfl_dir), 'docs', 'templates')

    print('Cleaning up existing sources directory.')
    if os.path.exists(sources_dir):
        shutil.rmtree(sources_dir)

    print('Populating sources directory with templates.')
    shutil.copytree(template_dir, sources_dir)

    readme = read_file(os.path.join(str(shfl_dir), 'README.md'))
    index = read_file(os.path.join(template_dir, 'index.md'))
    index = index.replace('{{autogenerated}}', readme[readme.find('##'):])
    with open(os.path.join(sources_dir, 'index.md'), 'w', encoding='utf-8') as f:
        f.write(index)

    print('Generating docs for Sherpa.ai Federated Learning Framework')
    for page_data in PAGES:
        classes = read_page_data(page_data, 'classes')

        blocks = []
        class_and_methods = False
        for element in classes:
            if not isinstance(element, (list, tuple)):
                element = (element, [])
            cls = element[0]
            subblocks = []
            signature = get_class_signature(cls)
            subblocks.append('<span style="float:right;">' +
                             class_to_source_link(cls) + '</span>')
            if element[1] or class_and_methods:
                class_and_methods = True
                subblocks.append('## ' + cls.__name__ + ' class\n')
            else:
                subblocks.append('### ' + cls.__name__ + '\n')
            subblocks.append(code_snippet(signature))
            docstring = cls.__doc__
            if docstring:
                subblocks.append(process_docstring(docstring))
            methods = collect_class_methods(cls, element[1])
            if methods:
                subblocks.append('\n---')
                subblocks.append('## ' + cls.__name__ + ' methods\n')
                subblocks.append('\n---\n'.join(
                    [render_function(method, method=True)
                     for method in methods]))
            blocks.append('\n'.join(subblocks))

        methods = read_page_data(page_data, 'methods')

        for method in methods:
            blocks.append(render_function(method, method=True))

        functions = read_page_data(page_data, 'functions')

        for function in functions:
            blocks.append(render_function(function, method=False))

        if not blocks:
            raise RuntimeError('Found no content for page ' +
                               page_data['page'])

        mkdown = '\n----\n\n'.join(blocks)
        # Save module page.
        # Either insert content into existing page,
        # or create page otherwise.
        page_name = page_data['page']
        path = os.path.join(sources_dir, page_name)
        if os.path.exists(path):
            template = read_file(path)
            if '{{autogenerated}}' not in template:
                raise RuntimeError('Template found for ' + path +
                                   ' but missing {{autogenerated}}'
                                   ' tag.')
            mkdown = template.replace('{{autogenerated}}', mkdown)
            print('...inserting autogenerated content into template:', path)
        else:
            print('...creating new page with autogenerated content:', path)
        subdir = os.path.dirname(path)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(mkdown)

    shutil.copyfile(os.path.join(str(shfl_dir), 'README.md'),
                    os.path.join(str(sources_dir), 'index.md'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'install.md'),
                    os.path.join(str(sources_dir), 'install.md'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'CONTRIBUTING.md'),
                    os.path.join(str(sources_dir), 'CONTRIBUTING.md'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'docs/logo.png'),
                    os.path.join(str(sources_dir), 'logo.png'))

    if not os.path.exists(os.path.join(str(sources_dir), 'stylesheets')):
        os.makedirs(os.path.join(str(sources_dir), 'stylesheets'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'docs/sherpa.css'),
                    os.path.join(str(sources_dir), 'stylesheets/sherpa.css'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'docs/fonts.css'),
                    os.path.join(str(sources_dir), 'stylesheets/fonts.css'))

    if not os.path.exists(os.path.join(str(sources_dir), 'fonts')):
        os.makedirs(os.path.join(str(sources_dir), 'fonts'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'docs/fonts/Ubuntu-Bold.ttf'),
                    os.path.join(str(sources_dir), 'fonts/Ubuntu-Bold.ttf'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'docs/fonts/Ubuntu-BoldItalic.ttf'),
                    os.path.join(str(sources_dir), 'fonts/Ubuntu-BoldItalic.ttf'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'docs/fonts/Ubuntu-Italic.ttf'),
                    os.path.join(str(sources_dir), 'fonts/Ubuntu-Italic.ttf'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'docs/fonts/Ubuntu-Light.ttf'),
                    os.path.join(str(sources_dir), 'fonts/Ubuntu-Light.ttf'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'docs/fonts/Ubuntu-LightItalic.ttf'),
                    os.path.join(str(sources_dir), 'fonts/Ubuntu-LightItalic.ttf'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'docs/fonts/Ubuntu-Medium.ttf'),
                    os.path.join(str(sources_dir), 'fonts/Ubuntu-Medium.ttf'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'docs/fonts/Ubuntu-MediumItalic.ttf'),
                    os.path.join(str(sources_dir), 'fonts/Ubuntu-MediumItalic.ttf'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'docs/fonts/Ubuntu-Regular.ttf'),
                    os.path.join(str(sources_dir), 'fonts/Ubuntu-Regular.ttf'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'docs/fonts/UbuntuMono-Bold.ttf'),
                    os.path.join(str(sources_dir), 'fonts/UbuntuMono-Bold.ttf'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'docs/fonts/UbuntuMono-BoldItalic.ttf'),
                    os.path.join(str(sources_dir), 'fonts/UbuntuMono-BoldItalic.ttf'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'docs/fonts/UbuntuMono-Italic.ttf'),
                    os.path.join(str(sources_dir), 'fonts/UbuntuMono-Italic.ttf'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'docs/fonts/UbuntuMono-Regular.ttf'),
                    os.path.join(str(sources_dir), 'fonts/UbuntuMono-Regular.ttf'))

    if not os.path.exists(os.path.join(str(sources_dir), 'js')):
        os.makedirs(os.path.join(str(sources_dir), 'js'))
    shutil.copyfile(os.path.join(str(shfl_dir), 'docs/sherpa.js'),
                    os.path.join(str(sources_dir), 'js/sherpa.js'))

if __name__ == '__main__':
    generate(os.path.join(str(shfl_dir), 'docs', 'sources'))

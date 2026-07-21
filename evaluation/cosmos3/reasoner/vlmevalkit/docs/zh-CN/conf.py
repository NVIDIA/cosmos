# flake8: noqa
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import ast
import os
import subprocess
import sys

import pytorch_sphinx_theme
from sphinx.builders.html import StandaloneHTMLBuilder

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'VLMEvalKit'
copyright = '2023, VLMEvalKit'
author = 'VLMEvalKit Authors'

# The full version, including alpha/beta/rc tags
version_file = '../../vlmeval/__init__.py'


def get_version():
    with open(version_file, 'r') as f:
        file_content = f.read()
    # Parse the file content into an abstract syntax tree (AST)
    tree = ast.parse(file_content, filename=version_file)

    # Iterate through the body of the AST, looking for an assignment to __version__
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '__version__':
                    return node.value.s
    raise ValueError('__version__ not found')


release = get_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
    'notfound.extension',
    'sphinxcontrib.jquery',
    'sphinx_design',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

language = 'cn'

# The master toctree document.
root_doc = 'index'
html_context = {
    'github_version': 'latest',
}
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# yapf: disable
html_theme_options = {
    'menu': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/open-compass/VLMEvalKit'
        },
    ],
    # Specify the language of shared menu
    'menu_lang': 'cn',
    # Disable the default edit on GitHub
    'default_edit_on_github': False,
}
# yapf: enable

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'https://cdn.datatables.net/v/bs4/dt-1.12.1/datatables.min.css',
    'css/readthedocs.css'
]
html_js_files = [
    'https://cdn.datatables.net/v/bs4/dt-1.12.1/datatables.min.js',
    'js/custom.js'
]

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'vlmevalkitdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (root_doc, 'vlmevalkit.tex', 'VLMEvalKit Documentation', author,
     'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(root_doc, 'vlmevalkit', 'VLMEvalKit Documentation', [author],
              1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (root_doc, 'vlmevalkit', 'VLMEvalKit Documentation', author,
     'VLMEvalKit Authors', 'AGI evaluation toolbox and benchmark.',
     'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# set priority when building html
StandaloneHTMLBuilder.supported_image_types = [
    'image/svg+xml', 'image/gif', 'image/png', 'image/jpeg'
]

# -- Extension configuration -------------------------------------------------
# Ignore >>> when copying code
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

# Auto-generated header anchors
myst_heading_anchors = 3
# Enable "colon_fence" extension of myst.
myst_enable_extensions = ['colon_fence', 'dollarmath']

# Configuration for intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'mmengine': ('https://mmengine.readthedocs.io/en/latest/', None),
    'transformers':
    ('https://huggingface.co/docs/transformers/main/en/', None),
}
napoleon_custom_sections = [
    # Custom sections for data elements.
    ('Meta fields', 'params_style'),
    ('Data fields', 'params_style'),
]

# Disable docstring inheritance
autodoc_inherit_docstrings = False
# Mock some imports during generate API docs.
autodoc_mock_imports = ['rich', 'attr', 'einops']
# Disable displaying type annotations, these can be very verbose
autodoc_typehints = 'none'

# The not found page
notfound_template = '404.html'


def builder_inited_handler(app):
    subprocess.run(['./cp_origin_docs.sh'])


def setup(app):
    app.connect('builder-inited', builder_inited_handler)

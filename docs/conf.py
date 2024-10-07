# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Cellmap Segmentation Challenge"
copyright = "2024, Jeff Rhoades, Emma Avetissian, Davis Vann Bennett, Marwan Zouinkhi"
author = "Jeff Rhoades, Emma Avetissian, Davis Vann Bennett, Marwan Zouinkhi"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    # "sphinx.ext.linkcode",
    "sphinx.ext.coverage",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    # "special-members": "__init__",
    "special-members": "__len__,__getitem__",
    "undoc-members": True,
    # "exclude-members": "__weakref__",
}
autodoc_typehints = "both"
autodoc_class_signature = "mixed"
autoclass_content = "init"
autosectionlabel_maxdepth = 2
coverage_show_missing_items = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = "CellMap Segmentation Challenge"
html_logo = "https://raw.githubusercontent.com/janelia-cellmap/cellmap-data/main/docs/source/_static/CellMapLogo.png"
html_favicon = "https://raw.githubusercontent.com/janelia-cellmap/cellmap-data/main/docs/source/_static/favicon.ico"
html_theme_options = {
    "show_navbar_depth": 3,
    # "home_page_in_toc": True,
}

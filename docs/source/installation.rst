Installation
=====

This repo requires the creation of a virtual environment. A Python version below 3.12 is required.

.. _prepare_the_environment:
Prepare the venv
----------------

Create a virtual environment with Python module as shown here

.. code-block:: console

    $ python3 -m venv venv

Activate it, and install the required packages

.. code-block:: console

    $ source venv/bin/activate
    (venv) $ pip3 install -r requirements.txt

.. _install_pearlm:

Install PEARLM
------------

To use PEARLM framework, first install it using pip:

.. code-block:: console

    (venv) $ pip3 install .

.. _additional_data:

Additional data
---------------

PEARLM requires the datasets and the embeddings from **data.zip** and **embedding-weights.zip** stored
in this `drive repository <https://drive.google.com/drive/folders/1e0uFWb6iJ6MXHtslZsqV8qRYC0Pl_AR7?usp=sharing>`.

Download both files and extract them inside the PEARLM repository.


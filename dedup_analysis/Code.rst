The Match-Maker documentation Code
==================================

There are three files.

1. Deduplication_algo.py 
2. CoreBGgeneration.py
3. prepare_4_EM.py

Below we provide the code docs for all three in the respective order.

------------

The core file that connects them all (Deduplication_algo.py)
------------------------------------------------------------

This is the main file that links two database entries. Look through the main function to see that names of the supplier is enough for match making to begin. CSV-CSV comparison will be added soon. 

.. automodule:: Deduplication_algo
   :members:
   :noindex:

------------

The Probability generating module (CoreBGgeneration.py)
-------------------------------------------------------

This is the module used to generate probability distributions that is used to find a match.

.. automodule:: CoreBGgeneration
   :members:
   :noindex:

------------

The preparation module (prepare_4_EM.py)
----------------------------------------

Cleaning and parameterization of raw input is done here. 

.. automodule:: prepare_4_EM
   :members:
   :noindex:




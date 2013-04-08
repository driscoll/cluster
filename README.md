cluster.py
=======

Hierarchical clustering for small collections of strings

Strings are transformed into sets of k-shingles and 
clustered in non-Euclidean space according to Jaccard distances.

Kevin Driscoll, 2013, kedrisco@usc.edu

Example
------

The file *joc_titles_since_2003.txt* contains the titles of 633 articles published in the Journal of Communication since 2003. The following command will transform the titles into 2-shingles and iteratively cluster the sets of shingles until any cluster reaches a diameter of 0.9. 

        $ python cluster.py -d 0.9 -k 2 -o joc joc_titles_since_2003.txt

This will output two CSV files: one with clusters as rows and one with cluster members as rows.

Requirements
------

* >= Python 2.6
* NLTK, http://nltk.org
* Twitter_NLP, https://github.com/aritter/twitter_nlp/

TODO 
------

* Normalize distance measures for strings of very different lengths 
* Include alternative tokenizing approaches 
* Pass function into SuperCluster for custom stop conditions

References
------

Rajaraman, A. & Ullman, J. D. (2011). Mining of Massive Datasets. Cambridge University Press.



#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Non-Euclidean clustering for small collections of strings 

Kevin Driscoll, 2013, kedrisco@usc.edu

Reference: 
Rajaraman, A. & Ullman, J. D. (2011). Mining of Massive Datasets. Cambridge University Press.

TODO Next steps
* Normalize for strings of very different lengths (e.g., blog comments and tweets)
* Pass in function as stop condition 

"""

from contextlib import closing
import collections
import copy
import csv
import fileinput
import glob
import heapq
import itertools
import multiprocessing
import operator
import optparse
import re
import sys

# These modules are not in the Standard Library
import nltk # http://nltk.org/
import twokenize # https://github.com/aritter/twitter_nlp

# Used for rough performance measures
import time

# Detect Python version
VERSION = sys.version_info[:2]

# Setup multiprocessing logger for debugging
logger = multiprocessing.log_to_stderr()
logger.setLevel(multiprocessing.SUBWARNING)

# The diameter of a cluster is the maximum distance
# between any two elements in the cluster.
# For Jaccard distance, it will be a value between 0 and 1.
# Clustering stops when the most recent cluster
# meets or exceeds the value of MAX_DIAMETER.
MAX_DIAMETER = 0.9

# Number of tokens to combine when shingling a string
KSHINGLES = 2

# Project-specific stopwords
STOPWORDS = nltk.corpus.stopwords.words('english')
STOPWORDS.extend([
                    "'s",
                    '"',
                    "'",
                    ':',
                    '.',
                    ','
                ])

class ReversibleKeyDict(collections.MutableMapping):
    """Hash table that uses unordered tuples for keys
        ('a', 'b') and ('b', 'a') will return the same value
    """

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs)) # use the free update to set keys

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)
        
    def __keytransform__(self, key):
        return tuple(sorted(key))


def filterstopwords(tokens, stopwords):
    filtered_tokens = []
    for token in itertools.ifilterfalse(lambda x: x.lower() in stopwords, tokens):
        filtered_tokens.append(token.lower())
    return filtered_tokens

def consecutive(tokens, k):
    """Return overlapping subgroups of tokens of length k
        e.g., if k = 3, ['the', 'quick', 'brown', 'fox'] will yield
                ['the','quick','brown'], ['quick', 'brown', 'fox']
       http://docs.python.org/2/library/itertools.html#recipes
    """
    iterables = itertools.tee(tokens, k)
    for i in range(1, k):
        for count in range(0,i):
            iterables[i].next()
    return itertools.izip(*iterables)

def tokens_to_kshingles(tokens, k):
    """ Returns a set of strings composed of adjacent tokens of size k
        e.g., if k = 3, ['the', 'quick', 'brown', 'fox'] will yield
                ['thequickbrown'], ['quickbrownfox']
    """
    if len(tokens) >= k:
        return set([''.join(g) for g in consecutive(tokens, k)]) 
    else:
        return set([''.join(tokens)])

def kshinglize_custom(args):
    """ Tokenizes string s, removes stopwords, and returns a set of k-shingles
    """
    s, k, stopwords = args
    return kshinglize(s, k, stopwords)

def kshinglize(s, k=KSHINGLES, stopwords=STOPWORDS):
    """ Tokenizes string s, removes stopwords, and returns a set of k-shingles
    """
    s = s.strip().lower()
    tokens_raw = twokenize.tokenize(s)
    tokens = filterstopwords(tokens_raw, stopwords)
    return tokens_to_kshingles(tokens, k)
    
def _calculate_distance(((eid_a, shingles_a), (eid_b, shingles_b))):
    if (shingles_a and shingles_b):
        jd = nltk.jaccard_distance(shingles_a, shingles_b)
    else:
        # One of the elements has no shingles
        jd = 1.0
    return ((eid_a, eid_b), jd) 

def build_distance_table(kshingles):
    """ Create a hash table of the Jaccard distance
        between all elements in the dict kshingles.
        kshingles = { id : set(shingle1, shingle2, ...) }

    """
    distance = ReversibleKeyDict()
    with closing(multiprocessing.Pool()) as pool:
        for (eid_a, eid_b), jd in pool.imap_unordered(_calculate_distance, itertools.combinations(kshingles.iteritems(), 2), 250):
            distance[eid_a, eid_b] = jd
    pool.join()
    return distance

class Cluster:

    def __init__(self, members, clustroid=None, diameter=None, radius=None, **args):
        if type(members) == list:
            members = set(members)
        self.members = copy.deepcopy(members)
        self.clustroid = clustroid 
        self.diameter = diameter
        self.radius = radius
        
class SuperCluster:
    """
        id_shingles should be a dict of 
            { id : set(shingle, shingle, ...) }
        for tweets this might be 
            { tweet_id: set(shingle, shingle, ...) }

    """

    def __init__(self, id_shingles, max_diameter=MAX_DIAMETER):

        self.max_diameter = max_diameter

        # Compute distance between all pairs of elements 
        # 3000 elements took about ~20m on Atom N450, 2gig ram
        sys.stderr.write('Initializing distance table...') 
        start = time.time()
        self.distance_table = build_distance_table(id_shingles)
        sys.stderr.write(' {0}s\n'.format(str(time.time()-start)))

        # Organize distance table into a priority queue
        sys.stderr.write('Constructing priority queue...')
        start = time.time()
        self.distance_queue = []
        for (origin, destination), jd in self.distance_table.iteritems():
            heapq.heappush(self.distance_queue, [jd, origin, destination])
        sys.stderr.write(' {0}s\n'.format(str(time.time()-start)))

        # Initially, each element is alone in its own cluster.
        self.clusters = {}
        for e_id in sorted(id_shingles.keys()):
            self.clusters[e_id] = Cluster(
                                    set([e_id]),
                                    clustroid=e_id,
                                    diameter=0,
                                    radius=0
                                  )

        # Clusterize!
        sys.stderr.write('Clustering...\n')
        self.clusterize()
   
    def clusterize(self):
        # Merge clusters until we reach the maximum diameter
        last_diameter = self.merge_nearest_clusters()
        while (last_diameter < self.max_diameter) and (len(self.distance_queue) > 0):
            last_diameter = self.merge_nearest_clusters()
        sys.stderr.write('Merging unclustered...\n')
        self.merge_unclustered()

    def merge_nearest_clusters(self):
        """ Merge the next closest pair of clusters
            Returns the diameter of the newly merged cluster
            Returns None if there are no more cluster pairs
        """
        if not len(self.distance_queue):
            return None 

        # Find the pair of clusters with the shortest distance
        jd, origin, destination = heapq.heappop(self.distance_queue)

        # Remove these two clusters from the hash table
        C = self.clusters.pop(origin)
        D = self.clusters.pop(destination)

        # Merge and add newly merged cluster to the hash table
        cluster = self.merge_clusters(C, D)
        self.clusters[cluster.clustroid] = cluster

        # Remove all clusters from the distance queue that refer
        #   to any member of the new cluster 
        """
        # TODO this is concise but very slow
        # About 10s with 4000 clusters on a fast quad-core machine
        self.distance_queue[:] = itertools.ifilterfalse(
                                    lambda x: set(x[1:]) & set(cluster.members), 
                                    self.distance_queue
                                 )
        """
        # TODO this is simple and faster
        # But still ~4-5s with 4000 clusters on a fast quad-core machine
        dq = []
        for entry in self.distance_queue:
            jd, id_a, id_b = entry
            if not id_a in cluster.members:
                if not id_b in cluster.members:
                    dq.append(entry)
        self.distance_queue[:] = dq

        # Re-order the performance queue
        heapq.heapify(self.distance_queue)

        # Calculate distance from the clustroid to all other clusters
        # Add to the queue
        for e_id in self.clusters.keys():
            if not e_id == cluster.clustroid:
                jd = self.distance_table[cluster.clustroid, e_id]
                heapq.heappush(self.distance_queue, [jd, cluster.clustroid, e_id])

        # Return the diameter of the newly merged cluster
        return cluster.diameter

    def merge_clusters(self, *clusters_to_merge):
        if not clusters_to_merge:
            return None
        elif len(clusters_to_merge) == 1:
            cluster = Cluster(
                clusters_to_merge[0].members,
                clusters_to_merge[0].clustroid,
                clusters_to_merge[0].diameter,
                clusters_to_merge[0].radius
            )
        else:
            members = set()
            for cluster in clusters_to_merge:
                members = members | cluster.members
            clustroid = self.calculate_clustroid(members)
            diameter = self.calculate_radius(members)
            radius = self.calculate_radius(members, clustroid)
            cluster = Cluster(
                        members,
                        clustroid,
                        diameter,
                        radius
                      )
        return cluster

    def merge_unclustered(self):

        # Identify clusters with only one member
        unclustered_clusters = [] 
        for cluster in self.clusters.itervalues():
            if len(cluster.members) == 1:
                unclustered_clusters.append(cluster)

        # Remove all single member clusters from the hashtable
        for cluster in unclustered_clusters:
            del self.clusters[cluster.clustroid]

        # Count up the number of unclustered clusters
        count = len(unclustered_clusters)
        
        # Merge the unclustered clusters
        uncluster = self.merge_clusters(*unclustered_clusters)

        # Add to the hashtable
        if uncluster:
            self.clusters[uncluster.clustroid] = uncluster
    
        return count

    def calculate_clustroid(self, member_ids):
        """ The clustroid is the member with the shortest distance
        to all other members. We use the sum of squared distances.
        Returns a Tweet ID
        """
        if (type(member_ids) == set):
            member_ids = list(member_ids)
        distance_sums = []
        for i in range(len(member_ids)):
            origin = member_ids[i]
            dist_sum = 0
            for j in range(len(member_ids)):
                if not i == j:
                    destination = member_ids[j]
                    jd = self.distance_table[origin, destination]
                    dist_sum += jd * jd
            distance_sums.append((dist_sum, origin))
        distance_sums.sort(key=lambda x: x[0])
        return distance_sums[0][1] # New clustroid

    def calculate_radius(self, members, clustroid=None):
        """ The radius of a cluster is the maximum distance from
            any member and the clustroid
        """
        if (type(members) == set):
            members = list(members)
        if not clustroid:
            clustroid = self.calculate_clustroid(members)
        radius = 0
        for destination in members:
            if not (destination == clustroid):
                jd = self.distance_table[clustroid, destination]
                if jd > radius:
                    radius = jd
        return radius

    def calculate_diameter(self, members):
        """ The diameter of a cluster is the maximum distance
            between any two members.
        """
        if (type(members) == set):
            members = list(members)
        diameter = 0
        for i in range(len(members)):
            origin = members[i]
            for j in range(len(members)):
                if not i == j:
                    destination = members[j]
                    jd = self.distance_table[origin, destination]
                    if jd > diameter:
                        diameter = jd
        return diameter

    def iter_clusters_by_size(self, reverse=True):
        for cluster in sorted(self.clusters.itervalues(), key=lambda c: len(c.members), reverse=reverse):
            yield cluster


if __name__=="__main__":

    parser = optparse.OptionParser(usage="Usage: python %prog [options] INPUTFILE")
    parser.add_option('-k', '--kshingles', help='Number of tokens to combine when shingling [default: %default]', dest='kshingles', action='store', type='int')
    parser.add_option('-d', '--diameter', help='Maximum cluster diameter [maximum: 1, default: %default]', dest='diameter', action='store', type='float')
    parser.add_option("--stopwords", help='Project-specific stopwords (separated by commas)', type='str', nargs=1, dest="stopwords")
    parser.add_option('-o', '--outfile', help='Prefix for the output files', dest='outfile', action='store', type='str')
    (options, args) = parser.parse_args()

    if options.kshingles:
        if VERSION < (2, 7):
            sys.stderr.write('Warning: The k-shingles commandline option is only available on Python 2.7 or higher. Proceeding with default value: {0}\n'.format(KSHINGLES))
        else:
            sys.stderr.write('Option: k-shingles will be composed of {0} tokens.\n'.format(options.kshingles))
            KSHINGLES = options.kshingles
    if options.stopwords:
        if VERSION < (2, 7):
            sys.stderr.write('Warning: The stopwords commandline option is only available on Python 2.7 or higher.\n')
        else:
            sys.stderr.write('Option: Custom stopwords: {0}\n'.format(options.stopwords))
            STOPWORDS.extend(options.stopwords.split(','))
    if options.diameter:
        sys.stderr.write('Option: Maximum cluster diameter: {0}\n'.format(options.diameter))
        MAX_DIAMETER = options.diameter
    if options.outfile:
        sys.stderr.write('Option: Output filenames will begin with: {0}\n'.format(options.outfile))
        OUTFILE = options.outfile + '_'
    else:
        OUTFILE = ''

    sys.stderr.write('Reading strings into memory...\n')
    corpus = [] 
    for line in fileinput.input(args):
        corpus.append(line.strip())

    sys.stderr.write('Assembling shingles from the corpus...\n')
    with closing(multiprocessing.Pool()) as pool:
        if VERSION < (2, 7):
            # In <= Python 2.6, we can only send one arg to kshinglize
            shingles = dict(enumerate(pool.imap(kshinglize, corpus)))
        else:
            shingles = dict(enumerate(pool.imap(kshinglize_custom, [(s, KSHINGLES, STOPWORDS) for s in corpus])))
    pool.join()

    sys.stderr.write('Initializing supercluster...\n')
    supercluster = SuperCluster(shingles, MAX_DIAMETER)

    sys.stderr.write('Exporting clusters CSV...\n')
    fn = OUTFILE + "clusters.csv"
    f = open(fn, 'wb')
    csvw = csv.writer(f, dialect='excel')
    headings = [
        u'cluster_id',
        u'member_count',
        u'diameter',
        u'radius',
        u'clustroid_text'
    ]
    with open(fn, 'wb') as f:
        csvw = csv.writer(f, dialect='excel')
        csvw.writerow(headings)
        for clustroid, cluster in supercluster.clusters.iteritems():
            row = [
                clustroid,
                len(cluster.members),
                cluster.diameter,
                cluster.radius,
                corpus[clustroid]
            ]
            csvw.writerow(row)

    sys.stderr.write('Exporting cluster members CSV...\n')
    fn = OUTFILE + "cluster_members.csv"
    headings = [
        u'cluster',
        u'text'
    ]
    with open(fn, 'wb') as f:
        csvw = csv.writer(f, dialect='excel')
        csvw.writerow(headings)
        for clustroid, cluster in supercluster.clusters.iteritems():
            for member_id in cluster.members:
                line = corpus[member_id]
                if line:
                    row = [
                        clustroid,
                        line
                    ]
                    csvw.writerow(row)


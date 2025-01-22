import pandas
import numpy
from scipy import sparse
from scipy.spatial import distance


def edge_values(mat, nodes, **kwargs):
    vals = pandas.Series(mat.data)
    return vals[~numpy.isnan(vals)]

def count_connections(mat, nodes, **kwargs):
    ncon = mat.nnz
    npairs = len(nodes) * (len(nodes) - 1)
    return (ncon, npairs)

def num_connections(mat, nodes, **kwargs):
    return mat.nnz

def connection_probability(mat, nodes, **kwargs):
    con_pairs = count_connections(mat, nodes, **kwargs)
    if con_pairs[1] == 0:
        return numpy.nan
    return con_pairs[0] / con_pairs[1]

def connection_probability_within(mat, nodes, props, interval, **kwargs):
    assert numpy.all(numpy.array(interval) >= 0)
    D = distance.squareform(distance.pdist(nodes[props]))
    mat = numpy.array(mat.todense() > 0)
    mask = (D > interval[0]) & (D <= interval[1])
    if numpy.any(mask):
        p = mat[mask].mean()
    else:
        p = numpy.nan
    return p

def reciprocal_probability(mat, nodes, **kwargs):
    mat = ((mat.todense() > 0) & (mat.todense().transpose() > 0))
    return connection_probability(sparse.csc_matrix(mat), nodes, **kwargs)

def extra_and_missing_reciprocals(mat, nodes, con_prob, **kwargs):
    mat = mat > 0
    rec = sparse.csc_matrix(mat.todense() & (mat.todense().transpose()))
    expected = con_prob * mat.nnz
    measured = rec.nnz
    return measured - expected

def count_downwards_minus_upwards(mat, nodes, prop, **kwargs):
    try:
        delta = nodes[prop].values.reshape((-1, 1)) - nodes[prop].values.reshape((1, -1))
    except:
        print(nodes.columns)
    mat = mat.tocoo()
    delta_con = delta[mat.row, mat.col]
    return numpy.sum(numpy.sign(delta_con))

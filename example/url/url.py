from reb import P

scheme = P.n(P.nic(':/?#'), 1) + ':'

hier = P.n01('//' + P.n(P.nic('/?#'))) + P.n(P.nic('?#'))

query = P.n01('?' + P.n(P.nic('#')))

fragment = P.n01('#' + P.n(P.ANYCHAR))

url = P.tag(P.n01(scheme), tag='scheme') \
        + P.tag(hier, tag='hierachy') \
        + P.tag(query, tag='query') \
        + P.tag(fragment, tag='fragment') 

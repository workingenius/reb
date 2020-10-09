from reb import P

scheme = P.example(
    P.n(P.nic(':/?#'), 1) + ':',
        'http:',
        'https:'
)

hier = P.example(
    P.n01('//' + P.n(P.nic('/?#'))) + P.n(P.nic('?#')),
        '//google.com',
        'localhost',
        '127.0.0.1:8080',
)


a_query = P.tag(P.n(P.nic('#&')), tag='query')

query = P.example(
    P.n01('?' + P.n(a_query + '&') + a_query),
        '',
        '?a=1',
        '?a=1&b=2',
)

fragment = P.example(
    P.n01('#' + P.n(P.ANYCHAR)),
        '',
        '#head'
)

url = P.tag(P.n01(scheme), tag='scheme') \
        + P.tag(hier, tag='hierachy') \
        + query \
        + P.tag(fragment, tag='fragment') 

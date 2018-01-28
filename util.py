import getopt


def digest_args(args):
    identifier = ''
    size = 5
    rules = 0
    iterations = '1'
    n_random_mappings = '0'
    diffuse = 0
    pad = 0
    opts, args = getopt.getopt(args[1:],
                               's:r:I:R:h?',
                               ['diffuse=', 'pad=', 'id='])
    for o, a in opts:
        if o == '-s':
            size = int(a)
        elif o == '-r':
            rules = a
        elif o == '-I':
            iterations = a
        elif o == '-R':
            n_random_mappings = a
        elif o == '--diffuse':
            diffuse = int(a)
        elif o == '--pad':
            pad = int(a)
        elif o == '--id':
            identifier = a

    return identifier, size, rules, iterations, n_random_mappings, diffuse, pad

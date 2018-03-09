def update_link(a, b, c, rule):
    """Updates a Chainer link object according to a rule.

    This function update a link ``c`` with a rule
    ``c_param = rule(a_param, b_param)`` for all parameters and persistents
    in ``a``, ``b``, and ``c``. The actual objects that are passed to the
    ``rule`` are :class:`chainer.Parameter`.

    Args:
        a (:class:`chainer.Link`): Chainer Link that is decomposed and passed
            to ``rule`` as a first argument. Must have same link hierarchy as
            `b` and `c`.
        b (:class:`chainer.Link`): Chainer Link that is decomposed and passed
            to ``rule`` as a second argument. Must have same link hierarchy as
            `c` and `a`.
        c (:class:`chainer.Link`): Chainer Link that is updated by ``rule``,
            ``a``, and ``b``. Must have same link hierarchy as `a` and `b`.
        rull (callable): Callable object that takes two arguments. The actual
            arguments that are passed to this callable are
            :class:`chainer.Parameter` objects which represent one layer in a
            neural network.
    """
    update_parameters(a, b, c, rule)
    update_persistents(a, b, c, rule)


def update_parameters(a, b, c, rule):
    """Updates a Chainer link object according to a rule.

    This function update a link ``c`` with a rule
    ``c_param = rule(a_param, b_param)`` for all parameters in ``a``, `b`, and
    ``c``. The actual objects that are passed to the ``rule`` are
    :class:`chainer.Parameter`. This function does not update the persistents
    in the link ``c``.

    Args:
        a (:class:`chainer.Link`): Chainer Link that is decomposed and passed
            to ``rule`` as a first argument. Must have same link hierarchy as
            `b` and `c`.
        b (:class:`chainer.Link`): Chainer Link that is decomposed and passed
            to ``rule`` as a second argument. Must have same link hierarchy as
            `c` and `a`.
        c (:class:`chainer.Link`): Chainer Link that is updated by ``rule``,
            ``a``, and ``b``. Must have same link hierarchy as `a` and `b`.
        rull (callable): Callable object that takes two arguments. The actual
            arguments that are passed to this callable are
            :class:`chainer.Parameter` objects which represent one layer in a
            neural network.
    """
    for name, c_param in c.namedparams():
        a_param = getparam(a, name)
        b_param = getparam(b, name)
        c_param.copydata(rule(a_param, b_param))


def update_persistents(a, b, c, rule):
    """Updates a Chainer link object according to a rule.

    This function update a link ``c`` with a rule
    ``c_param = rule(a_param, b_param)`` for all persistents in ``a``, `b`, and
    ``c``. The actual objects that are passed to the ``rule`` are
    :class:`chainer.Parameter`. This function does not update the parameters in
    the link ``c``.

    Args:
        a (:class:`chainer.Link`): Chainer Link that is decomposed and passed
            to ``rule`` as a first argument. Must have same link hierarchy as
            `b` and `c`.
        b (:class:`chainer.Link`): Chainer Link that is decomposed and passed
            to ``rule`` as a second argument. Must have same link hierarchy as
            `c` and `a`.
        c (:class:`chainer.Link`): Chainer Link that is updated by ``rule``,
            ``a``, and ``b``. Must have same link hierarchy as `a` and `b`.
        rull (callable): Callable object that takes two arguments. The actual
            arguments that are passed to this callable are
            :class:`chainer.Parameter` objects which represent one layer in a
            neural network.
    """
    for name, c_link in c.namedlinks():
        a_link = getlink(a, name)
        b_link = getlink(b, name)
        for name in c_link.__dict__['_persistent']:
            a_param = a_link.__dict__[name]
            b_param = b_link.__dict__[name]
            c_link.__dict__[name] = rule(a_param, b_param)


def getparam(link, name, depth='-1'):
    """Get a parameter according to its name.
    
    This function returns a parameter which has a name ``name``. The name
    should be same format as names generated from
    :method:`chainer.Link.namedparams`.  

    Args:
        link (:class:`chainer.Link`): Searching source.
        name (str): Name of searching parameter.
        depth (int): Depth of searching link, if it is smaller than or equal to
            0, it is infinity.
    """
    depth = float('inf') if depth < 1 else depth
    for _name, _param in link.namedparams():
        if _name == name and _name.count('/') <= depth:
            return _param


def getpersistent(link, name):
    """Get a persistent according to its name.
    
    This function returns a persistent which has a name ``name``. Unlike
    ``getparam()``, this function will not recursively search children links.

    Args:
        link (:class:`chainer.Link`): Searching source.
        name (str): Name of searching persistent.
    """
    for _link in link.links():
        for _name in _link.__dict__['_persistent']:
            if _name == name:
                return _link.__dict__[_name]


def getlink(link, name):
    """Get a child link according to its name.
    
    This function returns a child link which has a name ``name``. The name
    should be same format as names generated 

    Args:
        link (:class:`chainer.Link`): Searching source.
        name (str): Name of searching persistent.
    """
    for _name, _link in link.namedlinks():
        if _name == name:
            return _link

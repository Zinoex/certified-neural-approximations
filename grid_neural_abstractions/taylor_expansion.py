from juliacall import Main as jl
jl.seval("using TaylorModels")

import torch

from translators import JuliaTranslator


def first_order_certified_taylor_expansion(dynamics, expansion_point, delta):
    """
    A 1st-order Taylor expansion including residual (certified) of a function around a point.

    This is computed via TaylorModels.jl (Julia called via juliacall). To hide this
    from the user, we use our translator interface.

    :param dynamics: An object representing the dynamics to be expanded.
    :param expansion_point: The point around which to expand the dynamics.
    :param delta: The (hyperrectangular) radius of the expansion.
    :return: (a = f(c), B = Df(c), R) where the Taylor expansion is `f(c) + (x - c) Df(c) \oplus R`.
    """
    translator = JuliaTranslator()

    order = 1

    low, high = expansion_point - delta, expansion_point + delta
    dom = jl.IntervalBox._jl_call_nogil(low.to(torch.float64).numpy(), high.to(torch.float64).numpy())

    input_dim = dynamics.input_dim
    x = jl.seval("(order, c, dom, input_dim) -> [TaylorModelN(i, order, IntervalBox(c), dom) for i in 1:input_dim]")._jl_call_nogil(
        order, expansion_point.to(torch.float64).numpy(), dom, input_dim
    )

    y = dynamics.compute_dynamics(x, translator)

    # constant term (select zeroth order, first and only coefficient)
    a = jl.broadcast._jl_call_nogil(jl.seval("yi -> yi[0][1]"), y)
    a_lower = jl.broadcast._jl_call_nogil(jl.inf, a).to_numpy()
    a_upper = jl.broadcast._jl_call_nogil(jl.sup, a).to_numpy()

    # linear term (select first order, all coefficients)
    b = jl.broadcast._jl_call_nogil(jl.seval("yi -> yi[1][:]"), y)
    b = jl.broadcast._jl_call_nogil(jl.transpose, b)
    b = jl.reduce._jl_call_nogil(jl.vcat, b)
    b_lower = jl.broadcast._jl_call_nogil(jl.inf, b).to_numpy()
    b_upper = jl.broadcast._jl_call_nogil(jl.sup, b).to_numpy()

    # remainder
    r = jl.broadcast._jl_call_nogil(jl.remainder, y)
    r_lower = jl.broadcast._jl_call_nogil(jl.inf, r).to_numpy()
    r_upper = jl.broadcast._jl_call_nogil(jl.sup, r).to_numpy()

    return (a_lower, b_lower, r_lower), (a_upper, b_upper, r_upper)


def prepare_taylor_expansion(input_dim):
    jl.seval("n -> set_variables(Float64, \"x\", order=1, numvars=n)")._jl_call_nogil(input_dim)

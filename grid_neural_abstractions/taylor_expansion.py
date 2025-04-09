from juliacall import Main as jl
jl.seval("using TaylorModels")

from translators import JuliaTranslator


def first_order_certified_taylor_expansion(dynamics, expansion_point, delta):
    """
    A 1st-order Taylor expansion including residual (certified) of a function around a point.

    This is computed via TaylorModels.jl (Julia called via juliacall). To hide this
    from the user, we use our translator interface.

    :param dynamics: An object representing the dynamics to be expanded.
    :param expansion_point: The point around which to expand the dynamics.
    :param delta: The (hyperrectangular) radius of the expansion.
    :return: (a = f(c) - c Df(c), B = Df(c), R) where the Taylor expansion is `f(c) + (x - c) Df(c) \oplus R`.
    """
    translator = JuliaTranslator()

    order = 1

    low, high = expansion_point - delta, expansion_point + delta
    dom = jl.IntervalBox(low, high)

    input_dim = dynamics.input_dim
    x = jl.seval(f"[TaylorModelN(i, {order}, interval({expansion_point}), {dom}) for i in 1:{input_dim}]")
    # x = jl.seval(f"{z} - {expansion_point}")

    y = dynamics.compute_dynamics(x, translator)


def prepare_taylor_expansion(dynamics):
    n = dynamics.input_dim
    jl.seval(f"set_variables(Float64, \"x\", order=1, numvars={n})")

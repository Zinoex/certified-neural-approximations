from .translators.julia_translator import JuliaTranslator
import numpy as np
import time

jl = None
last_gc_time = 0  # Track when garbage collection was last performed

def check_jl_initialized():
    global jl
    if jl is None:
        from juliacall import Main

        jl = Main
        jl.seval("using TaylorModels")

def run_julia_gc():
    """Run Julia's garbage collector if enough time has passed since last collection"""
    global last_gc_time
    current_time = time.time()
    collection_interval = 30  # seconds
    # Only run GC every n seconds
    if current_time - last_gc_time > collection_interval:
        jl.seval("GC.gc(true)")
        last_gc_time = current_time

def first_order_certified_taylor_expansion(dynamics, expansion_point, delta):
    """
    A 1st-order Taylor expansion including residual (certified) of a function around a point.

    This is computed via TaylorModels.jl (Julia called via juliacall). To hide this
    from the user, we use our translator interface.

    :param dynamics: An object representing the dynamics to be expanded.
    :param expansion_point: The point around which to expand the dynamics.
    :param delta: The (hyperrectangular) radius of the expansion.
    :return: (a = f(c), B = Df(c), R) where the Taylor expansion is `f(c) + (x - c) Df(c) ⊕ R`.
    """

    # Import inside the function to allow multiprocessing
    check_jl_initialized()

    translator = JuliaTranslator()

    order = 1

    if not isinstance(expansion_point, np.ndarray):
        import torch
        expansion_point = expansion_point.to(torch.float64).numpy()
    
    low, high = expansion_point - delta, expansion_point + delta
    dom = jl.IntervalBox(low, high)

    input_dim = dynamics.input_dim
    
    # Initialize variables with default values to ensure they're always defined
    a_lower = a_upper = b_lower = b_upper = r_lower = r_upper = None
    
    # Create Taylor models
    x = jl.seval("(order, c, dom, input_dim) -> [TaylorModelN(i, order, IntervalBox(c), dom) for i in 1:input_dim]")(
        order, expansion_point, dom, input_dim
    )

    y = dynamics.compute_dynamics(x, translator)

    # constant term (select zeroth order, first and only coefficient)
    a = jl.broadcast(jl.seval("yi -> yi[0][1]"), y)
    a_lower = jl.broadcast(jl.inf, a)
    a_upper = jl.broadcast(jl.sup, a)

    # linear term (select first order, all coefficients)
    b = jl.broadcast(jl.seval("yi -> yi[1][:]"), y)
    b = jl.broadcast(jl.transpose, b)
    b = jl.reduce(jl.vcat, b)
    b_lower = jl.broadcast(jl.inf, b)
    b_upper = jl.broadcast(jl.sup, b)

    # remainder
    r = jl.broadcast(jl.remainder, y)
    r_lower = jl.broadcast(jl.inf, r)
    r_upper = jl.broadcast(jl.sup, r)
    
    # Run GC after expensive computation
    run_julia_gc()

    if input_dim>1:
        a_lower = a_lower.to_numpy()
        a_upper = a_upper.to_numpy()
        b_lower = b_lower.to_numpy()
        b_upper = b_upper.to_numpy()
        r_lower = r_lower.to_numpy()
        r_upper = r_upper.to_numpy()
    else:
        a_lower = np.array([[a_lower]])
        a_upper = np.array([[a_upper]])
        b_lower = np.array([[b_lower]])
        b_upper = np.array([[b_upper]])
        r_lower = np.array([[r_lower]])
        r_upper = np.array([[r_upper]])

    return (a_lower, b_lower, r_lower), (a_upper, b_upper, r_upper)


def prepare_taylor_expansion(n):
    check_jl_initialized()
    jl.seval("n -> set_variables(Float64, \"x\", order=2, numvars=n)")(n)

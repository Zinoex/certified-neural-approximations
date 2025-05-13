from .translators.julia_translator import JuliaTranslator
import numpy as np
import torch
import time
from .translators.taylor_translator import TaylorTranslator, CertifiedFirstOrderTaylorExpansion

jl = None

def check_jl_initialized():
    global jl
    if jl is None:
        from juliacall import Main

        jl = Main
        jl.seval("using TaylorModels")

def certified_taylor_expansion(dynamics, expansion_point, delta):
    """
    1st-order Taylor expansion of a function around a point.
    Computed without Julia call, using TaylorTranslator.
    """
    translator = TaylorTranslator()

    if not isinstance(expansion_point, np.ndarray):
        if isinstance(expansion_point, torch.Tensor):
            expansion_point = expansion_point.to(torch.float64).numpy()
        else:
            expansion_point = np.array(expansion_point, dtype=np.float64)

    if not isinstance(delta, np.ndarray):
        delta = np.array(delta, dtype=np.float64)

    if expansion_point.ndim == 0:
        expansion_point = np.array([expansion_point])
    if delta.ndim == 0:
        delta = np.array([delta])
    
    input_dim = expansion_point.shape[0]

    domain_lower = expansion_point - delta
    domain_upper = expansion_point + delta

    # Initialize the input for the Taylor expansion
    # This represents x as a Taylor model: x = expansion_point + 1*(x - expansion_point) + 0
    x_taylor = translator.to_format(expansion_point, domain_lower, domain_upper)

    # Compute the dynamics with the Taylor model input
    y_taylor = dynamics.compute_dynamics(x_taylor, translator)

    # Extract results
    # y_taylor.linear_approximation = (Df(c), f(c))
    # y_taylor.remainder = (R_lower, R_upper)
    
    b_coeffs = y_taylor.linear_approximation[0] # Df(c)
    a_coeffs = y_taylor.linear_approximation[1] # f(c)
    r_bounds = y_taylor.remainder             # (R_lower, R_upper)

    a_lower = a_coeffs
    a_upper = a_coeffs
    b_lower = b_coeffs
    b_upper = b_coeffs
    r_lower = r_bounds[0]
    r_upper = r_bounds[1]
    
    if input_dim > 1:
        # Ensure correct shapes if not already
        a_lower = np.array(a_lower).reshape(-1, 1) if not isinstance(a_lower, np.ndarray) or a_lower.ndim == 1 else a_lower
        a_upper = np.array(a_upper).reshape(-1, 1) if not isinstance(a_upper, np.ndarray) or a_upper.ndim == 1 else a_upper
        b_lower = np.array(b_lower)
        r_lower = np.array(r_lower).reshape(-1, 1) if not isinstance(r_lower, np.ndarray) or r_lower.ndim == 1 else r_lower
        r_upper = np.array(r_upper).reshape(-1, 1) if not isinstance(r_upper, np.ndarray) or r_upper.ndim == 1 else r_upper
    else:
        b_lower = np.array(b_lower).reshape(y_taylor.linear_approximation[0].shape[0], input_dim) # Ensure b is [output_dim, input_dim]
        b_upper = np.array(b_upper).reshape(y_taylor.linear_approximation[0].shape[0], input_dim) # Ensure b is [output_dim, input_dim]
        r_lower = np.array([[r_lower.item()]])
        r_upper = np.array([[r_upper.item()]])

    lower = (a_lower, b_lower, r_lower)
    upper = (a_upper, b_upper, r_upper)


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

    # return certified_taylor_expansion_julia(dynamics, expansion_point, delta)

    # Import inside the function to allow multiprocessing
    check_jl_initialized()

    translator = JuliaTranslator()

    order = 1

    # expansion_point = expansion_point.to(torch.float64).numpy()
    
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
    # run_julia_gc()

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

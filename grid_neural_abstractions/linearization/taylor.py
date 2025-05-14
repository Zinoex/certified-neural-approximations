from grid_neural_abstractions.certification_results import AugmentedSample
from grid_neural_abstractions.translators import JuliaTranslator
import numpy as np
from juliacall import Main as jl
jl.seval("using TaylorModels")


def prepare_taylor_expansion(n):
    jl.seval("n -> set_variables(Float64, \"x\", order=2, numvars=n)")(n)


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

    assert isinstance(expansion_point, np.ndarray), "Expansion point must be a numpy array"
    assert isinstance(delta, np.ndarray), "Delta must be a numpy array"

    translator = JuliaTranslator()

    order = 1

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

    if input_dim > 1:
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


class TaylorLinearization:
    def __init__(self, dynamics):
        self.dynamics = dynamics
        prepare_taylor_expansion(self.dynamics.input_dim)

    def linearize(self, samples):
        """
        Linearizes a batch of samples using Taylor expansion.
        """

        # Placeholder for the actual linearization logic
        # This should return a list of linearized samples
        return [self.linearize_sample(sample) for sample in samples]

    def linearize_sample(self, sample):
        taylor_pol_lower, taylor_pol_upper = \
            first_order_certified_taylor_expansion(
                self.dynamics, sample.center, sample.radius
            )

        # Unpack the Taylor expansion components
        # taylor_pol_lower <-- (f(c), Df(c), R_min)
        # taylor_pol_upper <-- (f(c), Df(c), R_max)
        f_c_lower = taylor_pol_lower[0]  # f(c) - function value at center
        f_c_upper = taylor_pol_upper[0]  # f(c) - function value at center
        df_c_lower = taylor_pol_lower[1]  # Df(c) - gradient at center
        df_c_upper = taylor_pol_upper[1]  # Df(c) - gradient at center
        r_lower = taylor_pol_lower[2]  # Minimum remainder term
        r_upper = taylor_pol_upper[2]  # Maximum remainder term

        A_upper = df_c_upper[sample.output_dim]
        b_upper = -np.dot(df_c_upper[sample.output_dim], sample.center) + f_c_upper[sample.output_dim] + r_upper[sample.output_dim]

        A_lower = df_c_lower[sample.output_dim]
        b_lower = -np.dot(df_c_lower[sample.output_dim], sample.center) + f_c_lower[sample.output_dim] + r_lower[sample.output_dim]

        max_gap = r_upper[sample.output_dim] - r_lower[sample.output_dim]

        return AugmentedSample.from_certification_region(
            sample,
            ((A_lower, b_lower), (A_upper, b_upper), max_gap)
        )

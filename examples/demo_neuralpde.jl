using PaddleChainRules.Paddle: paddle, PaddleModuleWrapper, PaddleFCNet

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, DiffEqFlux
using Quadrature, Cubature, Optimisers
import ModelingToolkit: Interval, infimum, supremum

@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# 2D PDE
eq  = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)

# Boundary conditions
bcs = [u(0,y) ~ 0.0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
       u(x,0) ~ 0.0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]
# Space and time domains
domains = [x ∈ Interval(0.0,1.0),
           y ∈ Interval(0.0,1.0)]

# Discretization
dx = 0.1
# number of dimensions
dim = 2 

# pde system
@named pde_system = PDESystem(eq,bcs,domains,[x,y],[u(x, y)])

# callback
cb = function (p,l)
    println("Current loss is: $l")
    return false
end

# paddle
# error when setting float32
paddle.set_default_dtype("float64")
paddleNN = paddle.nn.Sequential(
        paddle.nn.Linear(2, 16),
        paddle.nn.Sigmoid(),
        paddle.nn.Linear(16, 16),
        paddle.nn.Sigmoid(),
        paddle.nn.Linear(16, 1)
    )

paddlewrap = PaddleModuleWrapper(paddleNN)

# or use a constructor for full connected network
paddlewrap = PaddleFCNet(2, 1, 3, 16;dtype="float64", activation="sigmoid")

initθ, _ = Optimisers.destructure(paddlewrap)
discretization = PhysicsInformedNN(paddlewrap, StochasticTraining(100;bcs_points = 40), init_params =initθ)
## or other training strategy
# discretization = PhysicsInformedNN(paddlewrap, GridTraining(dx), init_params =initθ)
## QuadratureTraining() works very slow here. 
# discretization = PhysicsInformedNN(paddlewrap, QuadratureTraining(), init_params =initθ)
prob = discretize(pde_system,discretization)

# solve system
res = GalacticOptim.solve(prob, DiffEqFlux.ADAM(0.1); cb = cb, maxiters=4000)
prob = remake(prob, u0=res.minimizer)
res = GalacticOptim.solve(prob, DiffEqFlux.ADAM(0.01); cb = cb, maxiters=2000)
phi = discretization.phi


# error
xs,ys = [infimum(d.domain):dx/10:supremum(d.domain) for d in domains]
analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)

u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = abs.(u_predict .- u_real)
maximum(diff_u)
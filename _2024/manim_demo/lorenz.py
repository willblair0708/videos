from manim import *
from scipy.integrate import odeint
from scipy.integrate import solve_ivp


def lorenz_system(t, state, sigma=10, rho=28, beta=8 / 3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def ode_solution_points(function, state0, time, dt=0.01):
    solution = solve_ivp(
        function,
        t_span=(0, time),
        y0=state0,
        t_eval=np.arange(0, time, dt)
    )
    return solution.y.T


class LorenzAttractor(ThreeDScene):
    def construct(self):
        # Set up axes
        axes = ThreeDAxes(
            x_range=(-50, 50, 5),
            y_range=(-50, 50, 5),
            z_range=(-0, 50, 5),
            x_length=16,
            y_length=16,
            z_length=8,
        )
        axes.set_width(config.frame_width)
        axes.center()

        # Set up camera
        self.set_camera_orientation(phi=43*DEGREES, theta=76*DEGREES)
        self.begin_ambient_camera_rotation(rate=0.3)
        self.add(axes)

        # Add the equations
        equations = MathTex(
            R"""
            \begin{aligned}
            \frac{\mathrm{d} x}{\mathrm{~d} t} & =\sigma(y-x) \\
            \frac{\mathrm{d} y}{\mathrm{~d} t} & =x(\rho-z)-y \\
            \frac{\mathrm{d} z}{\mathrm{~d} t} & =x y-\beta z
            \end{aligned}
            """,
            font_size=30
        )
        equations.set_color_by_tex_to_color_map({
            "x": RED,
            "y": GREEN,
            "z": BLUE,
        })
        equations.to_corner(UL)
        self.add_fixed_in_frame_mobjects(equations)
        self.play(Write(equations))

        # Compute a set of solutions
        epsilon = 1e-5
        evolution_time = 30
        n_points = 10
        states = [
            [10, 10, 10 + n * epsilon]
            for n in range(n_points)
        ]
        colors = color_gradient([BLUE_E, BLUE_A], len(states))

        curves = VGroup()
        for state, color in zip(states, colors):
            points = ode_solution_points(lorenz_system, state, evolution_time)
            curve = VMobject()
            curve.set_points_smoothly([
                axes.c2p(*point) for point in points
            ])
            curve.set_stroke(color, 1, opacity=0.25)
            curves.add(curve)

        curves.set_stroke(width=2, opacity=1)

        # Display dots moving along those trajectories
        dots = VGroup(*[Dot(color=color, radius=0.1).set_stroke(BLACK, 1) for color in colors])

        def update_dots(dots):
            for dot, curve in zip(dots, curves):
                dot.move_to(curve.get_end())
            return dots

        dots.add_updater(update_dots)
        self.add(dots)

        # Animate the curves
        curves.set_opacity(0)
        self.play(
            *[Create(curve, rate_func=linear) for curve in curves],
            run_time=evolution_time,
        )
        self.wait()


    pass

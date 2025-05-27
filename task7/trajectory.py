import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import lines
from copy import deepcopy
import math

class LohrentzSystem:
    def __init__(self, prandtl, rayleigh) -> None:
        self.prandtl = prandtl
        self.rayleigh = rayleigh

    def right_side(self, variables: np.ndarray) -> np.ndarray:
        return np.array([self.prandtl * (variables[1] - variables[0]),
                           self.rayleigh * variables[0] - variables[1] - variables[0] * variables[2],
                           -variables[2] + variables[0] * variables[1]])
    
    def jacobian(self, variables: np.ndarray) -> np.ndarray:
        return np.array([[-self.prandtl, self.prandtl, 0.0],
                         [self.rayleigh - variables[2], -1.0, -variables[0]],
                         [variables[1], variables[0], -1.0]])

def step(dt: float,
         system: LohrentzSystem,
         initial: np.ndarray,
         newton_precision: float =  1.0e-4) -> np.ndarray | None:
    half_dt = dt / 2.0
    const_part = initial + system.right_side(initial) * half_dt
    variables = deepcopy(initial)

    # newton iterations
    for _ in range(100):
        newton_function = const_part - variables + system.right_side(variables) * half_dt
        newton_inv_deriv = -linalg.inv(np.identity(3) + system.jacobian(variables) * half_dt)
        variables = variables - newton_inv_deriv.dot(newton_function)

        if linalg.norm(newton_function) < newton_precision:
            return variables

    return None


if __name__=='__main__':
    TIME_STEP = 0.01
    TIME_MAX = 1.0e2
    TIME_STEPS = int(TIME_MAX / TIME_STEP)
    LOHRENZ_SYS = LohrentzSystem(prandtl=20.0, rayleigh=28.0)

    history = np.empty((TIME_STEPS + 1, 3))
    history[0] = np.array([0.0, 0.5, 1.05])
    history_velocity = np.empty((TIME_STEPS + 1, 3))
    history_velocity[0] = LOHRENZ_SYS.right_side(history[0])
    for i in range(TIME_STEPS):
        history[i + 1] = step(TIME_STEP, LOHRENZ_SYS, history[i])
        history_velocity[i + 1] = LOHRENZ_SYS.right_side(history[i + 1])
        if history[i + 1] is None:
            break;

    print(len(history), "points")

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(*history.T, lw = 0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    def add_phase_trajectory_plot(coordinate_id: int):
        first_attractor_coordinate = [
            math.sqrt(LOHRENZ_SYS.rayleigh - 1),
            math.sqrt(LOHRENZ_SYS.rayleigh - 1),
            LOHRENZ_SYS.rayleigh - 1,
        ]
        second_attractor_coordinate = [
            -math.sqrt(LOHRENZ_SYS.rayleigh - 1),
            -math.sqrt(LOHRENZ_SYS.rayleigh - 1),
            LOHRENZ_SYS.rayleigh - 1,
        ]

        ax1 = plt.figure().add_subplot()
        coordinates = [coord[coordinate_id] for coord in history]
        velocities = [vel[coordinate_id] for vel in history_velocity]
        ax1.plot(coordinates, velocities)

        ymin = min(velocities)
        ymax = max(velocities)

        attrctr_1 = first_attractor_coordinate[coordinate_id]
        attrctr_2 = second_attractor_coordinate[coordinate_id]
        stationary_1 = lines.Line2D([attrctr_1, attrctr_1], [ymin, ymax], color = 'red', linestyle='-', label = 'first attractor')
        stationary_2 = lines.Line2D([attrctr_2, attrctr_2], [ymin, ymax], color = 'orange', label = r'second attractor')

        ax1.add_line(stationary_1)
        ax1.add_line(stationary_2)

        coordinate_names = ['X','Y','Z']
        name = coordinate_names[coordinate_id]
        ax1.set_xlabel(rf'${name}$')
        velocity_names = [r'$\dot{X}$',r'$\dot{Y}$',r'$\dot{Z}$']
        ax1.set_ylabel(velocity_names[coordinate_id])
        ax1.legend(handles = [stationary_1, stationary_2])

    add_phase_trajectory_plot(0)
    add_phase_trajectory_plot(1)
    add_phase_trajectory_plot(2)
    plt.show()

    # def lorenz(xyz, *, s=10, r=28, b=2.667):
    #     """
    #     Parameters
    #         ----------
    #     xyz : array-like, shape (3,)
    #        Point of interest in three-dimensional space.
    #     s, r, b : float
    #        Parameters defining the Lorenz attractor.
    #
    #     Returns
    #     -------
    #     xyz_dot : array, shape (3,)
    #        Values of the Lorenz attractor's partial derivatives at *xyz*.
    #     """
    #     x, y, z = xyz
    #     x_dot = s*(y - x)
    #     y_dot = r*x - y - x*z
    #     z_dot = x*y - b*z
    #     return np.array([x_dot, y_dot, z_dot])
    #
    #
    # dt = 0.01
    # num_steps = 10000
    #
    # xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
    # xyzs[0] = (0., 1., 1.05)  # Set initial values
    # # Step through "time", calculating the partial derivatives at the current point
    # # and using them to estimate the next point
    # for i in range(num_steps):
    #     xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt
    #
    # # Plot
    # ax = plt.figure().add_subplot(projection='3d')
    #
    # ax.plot(*xyzs.T, lw=0.5)
    # ax.set_xlabel("X Axis")
    # ax.set_ylabel("Y Axis")
    # ax.set_zlabel("Z Axis")
    # ax.set_title("Lorenz Attractor")
    #
    # plt.show()

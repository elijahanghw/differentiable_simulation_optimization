import jax
import jax.numpy as jnp

BODY_MASS = 0.5
BODY_INERTIA = 0.01
MOTOR_MASS = 0.2
ARM_DENSITY = 0.5 # kg/m
THRUST_COEFFICIENT = 15

def morphology(l: float):

    m = BODY_MASS + 2*MOTOR_MASS + 2*l*ARM_DENSITY
    J =  BODY_INERTIA + 2*MOTOR_MASS * l**2 + 2*(ARM_DENSITY*l**3)/3

    Bf = jnp.array([[0, 0],
                    [-THRUST_COEFFICIENT, -THRUST_COEFFICIENT]])
    
    Bm = jnp.array([[-THRUST_COEFFICIENT*l, THRUST_COEFFICIENT*l]])

    return Bf, Bm, m, J
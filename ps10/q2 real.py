#Hi, please using mouse magnify the simulator screen to see better.Thank you
from pylab import *

h = 2e-18*10
hbar = 1.0546e-36
L = 1e-8
M = 9.109e-31
N = 1000 # Grid slices

a = L/N

def complex_arg(trans):	
    def f(y):
        return trans(real(y)) + 1j*trans(imag(y))

    return f
	
@complex_arg
def dst(y):
    """
    Perform dst transform for real argument
    """
    N = len(y)
    y2 = empty(2*N,float)
    y2[0] = y2[N] = 0.0
    y2[1:N] = y[1:]
    y2[:N:-1] = -y[1:]
    a = -imag(rfft(y2))[:N]
    a[0] = 0.0

    return a

#1D inverse DST Type-I

@complex_arg
def idst(a):
    N = len(a)
    c = empty(N+1,complex)
    c[0] = c[N] = 0.0
    c[1:N] = -1j*a[1:]
    y = irfft(c)[:N]
    y[0] = 0.0

    return y

ksi = zeros(N+1,complex)

def ksi0(x):
    x0 = L/2
    sigma = 1e-10
    k = 5e10
    return exp(-(x-x0)**2/2/sigma**2)*exp(1j*k*x)

x = linspace(0,L,N+1)
ksi[:] = ksi0(x)
ksi[[0,N]]=0

b0 = dst(ksi)

t = 1e-18
b_ = b0*exp(1j*pi**2*hbar*arange(1,N+2)**2/2/M/L**2*t)

ksi_ = idst(b_)
plot(x - L/2, real(ksi_))
grid(True, which='both', linestyle='--', linewidth=0.5)
xlabel("Position (X) [m]")
ylabel("Amplitude (Real Part) [arbitrary units]")
title("Wave Function Plot")
axhline(0, color='black', linewidth=0.8)
axvline(0, color='black', linewidth=0.8, linestyle='--')
ticks = linspace(-L/2, L/2, 5)
xticks(ticks, [f"{tick:.1e}" for tick in ticks])
yticks(linspace(-1, 1, 5), [f"{tick:.1f}" for tick in linspace(-1, 1, 5)])
savefig("q2p0")
show()

from vpython import canvas, curve, rate, vector, color, label

# Create a canvas with a white background
scene = canvas(background=color.white)

# Create the curve object with initial points and set the color to blue
ksi_c = curve(color=color.blue)

# Add coordinate lines
x_axis = curve(color=color.black)
y_axis = curve(color=color.black)

# Draw x-axis
x_axis.append(vector(-L/2, 0, 0))
x_axis.append(vector(L/2, 0, 0))

# Draw y-axis at x = 0 where the wave function reflects
y_axis.append(vector(0, -1e-8, 0))
y_axis.append(vector(0, 1e-8, 0))

# Add labels for axes
label(pos=vector(L/2 + 1e-9, -2e-9, 0), text="Position (X) [m]", color=color.black, box=False)
label(pos=vector(-L/2 + 1e-9, 1e-8, 0), text="Amplitude (Real Part)", color=color.black, box=False, align="center")

# i Add numeric scales to the axes
for i in range(-2, 3):
    x_label_pos = vector(i * L / 4, -1e-9, 0)
    label(pos=x_label_pos, text=f"{i * L / 4:.1e}", color=color.black, box=False)

for i in range(-1, 2):
    y_label_pos = vector(0, i * 5e-9, 0)
    label(pos=y_label_pos, text=f"{i * 0.5:.1f}", color=color.black, box=False)

# now this is to Initialize the curve with zero positions
for xi in x - L/2:
    ksi_c.append(vector(xi, 0, 0))

t = 0
while True:
    rate(30)  # Controls the animation speed
    b_ = b0 * exp(1j * pi**2 * hbar * arange(1, N + 2)**2 / (2 * M * L**2) * t)
    ksi_ = idst(b_)

    # Update the curve points dynamically with real part only
    for i in range(len(x)):
        ksi_c.modify(i, vector(x[i] - L/2, real(ksi_[i]) * 1e-9, 0))
    
    t += h * 5

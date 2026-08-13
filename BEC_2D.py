# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 10:46:31 2025

@author: jrodr
"""
# -*- coding: utf-8 -*-
"""
2D Gross-Pitaevskii Equation Solver

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============================================================================
# 1. Grid and Time Setup
# =============================================================================
n = 256            # Grid resolution
x_max = 5.0
y_max = 5.0
t_max = 3.0       # Total simulation time 
dt = 0.005        # Time step
steps_per_frame = 10 # Physics steps per visual frame

# Calculate exactly how many frames are needed to reach t_max
total_frames = int(t_max / (dt * steps_per_frame))

dx = (2 * x_max) / n
dy = (2 * y_max) / n

# 1D Spatial coordinates (endpoint=False prevents period overlap in FFT)
x = np.linspace(-x_max, x_max, n, endpoint=False)
y = np.linspace(-y_max, y_max, n, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# Momentum (k) Space Setup (Native ordering, no fftshift needed)
kx = np.fft.fftfreq(n, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(n, d=dy) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')

# =============================================================================
# 2. Physics Parameters & Operators
# =============================================================================
a = -10.0     # Nonlinear interaction strength (g)
w = 5.0     # Harmonic trapping frequency (omega)

# Kinetic Energy Operator in Momentum Space
T_k = (KX**2 + KY**2) / 2.0
op_K = np.exp(-1j * T_k * dt)  # Full-step momentum evolution operator

# Static Trapping Potential
V_trap = 0.5 * (w**2) * (X**2 + Y**2)

# Initial Wavefunction: Gaussian profile
psi = np.exp(-(X**2 + Y**2)) * (1 + 0j)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx * dy) # Normalize probability to 1

# =============================================================================
# 3. Animation Setup
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 6))
fig.canvas.manager.set_window_title('Split-Step GPE Solver')

# 'animated=True' has been REMOVED so the image actually renders
img = ax.imshow(
    np.abs(psi.T)**2, 
    cmap='magma', 
    extent=[-x_max, x_max, -y_max, y_max], 
    origin='lower', 
    vmax=np.max(np.abs(psi)**2) * 1.5, # Fixed scale prevents flickering
    vmin=0
)

ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(img, ax=ax, label=r'$|\psi|^2$')

# Bulletproof time text: Placed inside the axes with a dark background box
time_text = ax.text(
    0.03, 0.96, "BEC Density | t = 0.0000", 
    transform=ax.transAxes, 
    color='white', 
    fontsize=11, 
    fontweight='bold', 
    ha='left', 
    va='top',
    bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3')
)

# =============================================================================
# 4. Time Evolution Loop (Split-Step)
# =============================================================================
def update(frame):
    global psi
    
    for _ in range(steps_per_frame):
        # --- STEP A: Half-step in Real Space ---
        V_total = V_trap + a * np.abs(psi)**2
        psi *= np.exp(-1j * V_total * (dt / 2.0))
        
        # --- STEP B: Full-step in Momentum Space ---
        psi_k = np.fft.fft2(psi)
        psi_k *= op_K
        psi = np.fft.ifft2(psi_k)
        
        # --- STEP C: Half-step in Real Space ---
        V_total = V_trap + a * np.abs(psi)**2
        psi *= np.exp(-1j * V_total * (dt / 2.0))

    # Update visual arrays. (frame + 1) ensures the text matches the step taken.
    current_t = (frame + 1) * steps_per_frame * dt
    img.set_array(np.abs(psi.T)**2)
    time_text.set_text(f"BEC Density | t = {current_t:.4f}")
    
    return img, time_text

# Run the animation. repeat=False stops it when it hits t_max!
ani = animation.FuncAnimation(
    fig, update, frames=total_frames, interval=30, blit=False, repeat=False
)

plt.show()
    
    
    

    

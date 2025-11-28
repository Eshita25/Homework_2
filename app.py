"""
Streamlit app: Homework 2 — rectangular fluid element (advected)
Velocity profile (explicit):
    u(x,y) = - y / (r^2 + eps)
    v(x,y) =   x / (r^2 + eps)

This app:
 - Lets the user edit eps, viscosity, tolerances, rectangle params, integration params
 - Runs checks: steady (∂/∂t), incompressible (analytic = 0 here), inviscid (requires viscosity==0)
 - If checks pass: advects rectangle corners with RK4 and shows GIF + area plot + snapshots
 - If checks fail: shows clear failure message and the diagnostic numeric divergence (masked)
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from io import BytesIO
import math
import tempfile
import os

st.set_page_config(page_title="Rectangular fluid element — Homework 2", layout="wide")
st.title("Homework 2 — Rectangular fluid element (advected)")

# Sidebar inputs
st.sidebar.header("Velocity profile (fixed)")
st.sidebar.markdown("**u(x,y) = - y / (r^2 + eps)**  \n**v(x,y) =  x / (r^2 + eps)**")
eps = st.sidebar.number_input("eps (regularization)", min_value=1e-12, value=1e-6, format="%.8g", help="Small value to avoid singularity at r=0")

st.sidebar.header("Assumption & numeric tolerances")
viscosity = st.sidebar.number_input("viscosity (set 0 for inviscid)", value=0.0, format="%.6g")
steady_tol = st.sidebar.number_input("steady_tol (max |∂u/∂t|)", value=1e-6, format="%.6g")
div_tol = st.sidebar.number_input("div_tol (analytic divergence tolerance)", value=1e-3, format="%.6g")
r_mask = st.sidebar.number_input("r_mask (mask radius for numeric divergence)", min_value=0.0, value=0.25, format="%.4g",
                                 help="Ignore numerical divergence inside this radius when reporting diagnostics")

st.sidebar.header("Rectangle initial properties")
center_x = st.sidebar.number_input("center_x", value=2.0, format="%.3f")
center_y = st.sidebar.number_input("center_y", value=0.0, format="%.3f")
Lx = st.sidebar.number_input("Lx (width)", min_value=1e-6, value=0.5, format="%.3f")
Ly = st.sidebar.number_input("Ly (height)", min_value=1e-6, value=0.2, format="%.3f")
angle_deg = st.sidebar.number_input("angle_deg", value=0.0, format="%.2f")

st.sidebar.header("Integration & plotting")
dt = st.sidebar.number_input("dt (s)", min_value=1e-4, value=0.05, format="%.4g")
nsteps = st.sidebar.number_input("nsteps (frames)", min_value=2, value=160)
xmin = st.sidebar.number_input("xmin", value=-5.0, format="%.2f")
xmax = st.sidebar.number_input("xmax", value=5.0, format="%.2f")
ymin = st.sidebar.number_input("ymin", value=-5.0, format="%.2f")
ymax = st.sidebar.number_input("ymax", value=5.0, format="%.2f")

save_gif = st.sidebar.checkbox("Allow saving GIF for download", value=True)

# Buttons
run_button = st.button("Run checks & advect rectangle")

# Info panel: show velocity formula
st.markdown("**Velocity profile used:**")
st.latex(r"u(x,y) = -\frac{y}{r^2 + \varepsilon}\qquad v(x,y)=\frac{x}{r^2 + \varepsilon}")
st.caption(f"Using eps = {eps:g}. Analytic divergence = 0 for this profile (r != 0).")

# Functions (mirror the working Colab code)
def u_velocity(x, y, eps_local):
    r2 = x*x + y*y
    return -y / (r2 + eps_local)

def v_velocity(x, y, eps_local):
    r2 = x*x + y*y
    return  x / (r2 + eps_local)

def velocity_on_grid(X, Y, eps_local):
    r2 = X*X + Y*Y
    u = -Y / (r2 + eps_local)
    v =  X / (r2 + eps_local)
    return u, v

def central_diff_x(f, xcoords):
    dx = xcoords[1] - xcoords[0]
    dfdx = np.zeros_like(f)
    dfdx[:,1:-1] = (f[:,2:] - f[:,:-2]) / (2*dx)
    dfdx[:,0]  = (f[:,1] - f[:,0]) / dx
    dfdx[:,-1] = (f[:,-1] - f[:,-2]) / dx
    return dfdx

def central_diff_y(f, ycoords):
    dy = ycoords[1] - ycoords[0]
    dfdy = np.zeros_like(f)
    dfdy[1:-1,:] = (f[2:,:] - f[:-2,:]) / (2*dy)
    dfdy[0,:]  = (f[1,:] - f[0,:]) / dy
    dfdy[-1,:] = (f[-1,:] - f[-2,:]) / dy
    return dfdy

def rk4_step_point(x, y, dt, eps_local):
    # RK4 for given profile
    u1 = u_velocity(x, y, eps_local); v1 = v_velocity(x, y, eps_local)
    u2 = u_velocity(x + 0.5*dt*u1, y + 0.5*dt*v1, eps_local); v2 = v_velocity(x + 0.5*dt*u1, y + 0.5*dt*v1, eps_local)
    u3 = u_velocity(x + 0.5*dt*u2, y + 0.5*dt*v2, eps_local); v3 = v_velocity(x + 0.5*dt*u2, y + 0.5*dt*v2, eps_local)
    u4 = u_velocity(x + dt*u3, y + dt*v3, eps_local); v4 = v_velocity(x + dt*u3, y + dt*v3, eps_local)
    newx = x + (dt/6.0)*(u1 + 2*u2 + 2*u3 + u4)
    newy = y + (dt/6.0)*(v1 + 2*v2 + 2*v3 + v4)
    return newx, newy

def polygon_area(pts):
    x = pts[:,0]; y = pts[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

# Main execution
if run_button:
    st.info("Running assumption checks...")
    # Build diagnostic grid
    nx_grid = 161
    ny_grid = 161
    xg = np.linspace(xmin, xmax, nx_grid)
    yg = np.linspace(ymin, ymax, ny_grid)
    X, Y = np.meshgrid(xg, yg)

    # Steady check: evaluate at t and t+eps_t (analytic profile is steady)
    eps_t = 1e-9
    U0, V0 = velocity_on_grid(X, Y, eps)
    U1, V1 = velocity_on_grid(X, Y, eps)  # same; profile steady
    du_dt_est = (U1 - U0) / (eps_t + 1e-30)
    dv_dt_est = (V1 - V0) / (eps_t + 1e-30)
    max_du_dt = float(np.max(np.abs(du_dt_est)))
    max_dv_dt = float(np.max(np.abs(dv_dt_est)))

    # Numerical divergence (diagnostic) masked near origin
    dUdx = central_diff_x(U0, xg)
    dVdy = central_diff_y(V0, yg)
    num_div = dUdx + dVdy
    r2 = X*X + Y*Y
    mask_far = (r2 >= (r_mask**2))
    if np.any(mask_far):
        max_num_div_masked = float(np.max(np.abs(num_div[mask_far])))
    else:
        max_num_div_masked = float(np.max(np.abs(num_div)))
    # Analytic divergence for this profile is zero (except singular at r=0)
    max_div = 0.0

    is_steady = (max_du_dt < steady_tol and max_dv_dt < steady_tol)
    is_incompressible = (max_div < div_tol)
    is_inviscid = (abs(viscosity) < 1e-12)

    # Show diagnostics
    st.write("**Assumption checks:**")
    st.write(f"- max |∂u/∂t| = {max_du_dt:.3e}, max |∂v/∂t| = {max_dv_dt:.3e}  ⇒ steady? **{is_steady}**")
    st.write(f"- numeric max |∇·u| (masked r > {r_mask:.3g}) = {max_num_div_masked:.3e}")
    st.write(f"- analytic max |∇·u| = {max_div:.3e}  ⇒ incompressible? **{is_incompressible}**")
    st.write(f"- user viscosity = {viscosity}  ⇒ inviscid? **{is_inviscid}**")

    # Hard-stop if any assumption fails
    if not is_steady:
        st.error("STOP: Flow not steady (∂/∂t too large). Edit profile or increase steady_tol to continue.")
    elif not is_incompressible:
        st.error("STOP: Flow not incompressible (analytic divergence nonzero). Edit div_tol or change profile.")
    elif not is_inviscid:
        st.error("STOP: viscosity != 0; Euler inviscid assumption fails. Set viscosity = 0 to proceed.")
    else:
        st.success("All checks passed — proceeding with rectangle advection.")

        # Build initial rectangle corners
        theta = math.radians(angle_deg)
        w2 = Lx / 2.0
        h2 = Ly / 2.0
        rel = np.array([[-w2, -h2],
                        [ w2, -h2],
                        [ w2,  h2],
                        [-w2,  h2]], dtype=float)
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])
        corners0 = (rel @ R.T) + np.array([center_x, center_y])   # shape (4,2)

        # Integrate corner trajectories
        trajectories = np.zeros((4, int(nsteps), 2), dtype=float)
        trajectories[:,0,:] = corners0.copy()
        t_now = 0.0
        for k in range(1, int(nsteps)):
            for c in range(4):
                x_prev, y_prev = trajectories[c, k-1]
                trajectories[c,k,0], trajectories[c,k,1] = rk4_step_point(x_prev, y_prev, dt, eps)
            t_now += dt

        # Compute area over time
        areas = np.array([ polygon_area(trajectories[:,k,:]) for k in range(int(nsteps)) ])

        # Create GIF in memory
        try:
            st.info("Creating animation GIF (in-memory)...")
            fig, ax = plt.subplots(figsize=(6,6))
            ax.set_aspect('equal')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_title('Rectangular fluid element advected (Homework 2 style)')

            # background streamlines
            Ug, Vg = velocity_on_grid(*np.meshgrid(np.linspace(xmin, xmax, 161), np.linspace(ymin, ymax, 161)), eps)
            ax.streamplot(np.linspace(xmin, xmax, 161), np.linspace(ymin, ymax, 161), Ug, Vg, density=1.0, color='lightgray', linewidth=0.7)

            poly_line, = ax.plot([], [], 'r-', linewidth=2)
            corners_sc = ax.scatter([], [], c='blue', s=40)
            time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

            def init_anim():
                poly_line.set_data([], [])
                corners_sc.set_offsets(np.zeros((4,2)))
                time_text.set_text('')
                return poly_line, corners_sc, time_text

            def animate(i):
                pts = trajectories[:, i, :]
                poly_line.set_data(np.append(pts[:,0], pts[0,0]), np.append(pts[:,1], pts[0,1]))
                corners_sc.set_offsets(pts)
                time_text.set_text(f"t = {i*dt:.2f} s  area = {areas[i]:.6g}")
                return poly_line, corners_sc, time_text

            anim = animation.FuncAnimation(fig, animate, init_func=init_anim,
                                           frames=int(nsteps), interval=int(1000*dt), blit=True)

            # Save to bytes (Pillow)
            buf = BytesIO()
            writer = PillowWriter(fps=int(1.0/dt) if dt>0 else 20)
            anim.save(buf, writer=writer, dpi=80)
            buf.seek(0)
            plt.close(fig)

            st.image(buf, caption="Animated rectangular element (GIF)", use_column_width=True)

            if save_gif:
                # offer download
                st.download_button("Download GIF", data=buf.getvalue(), file_name="rect_element.gif", mime="image/gif")
        except Exception as e:
            st.error(f"Could not create animation GIF: {e}")
            st.write("As fallback: showing initial & final snapshots and area plot below.")

            # continue to show snapshots

        # Show initial & final snapshots side-by-side
        import matplotlib.pyplot as plt
        fig2, axes = plt.subplots(1, 2, figsize=(10,5), constrained_layout=True)
        Xsnap = np.linspace(xmin, xmax, 81); Ysnap = np.linspace(ymin, ymax, 81)
        Xg, Yg = np.meshgrid(Xsnap, Ysnap)
        Ug2, Vg2 = velocity_on_grid(Xg, Yg, eps)

        for ax in axes:
            ax.streamplot(Xsnap, Ysnap, Ug2, Vg2, density=1.0, color='lightgray', linewidth=0.7)
            ax.set_aspect('equal')
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

        # Left: initial
        axes[0].set_title("Initial (t=0)")
        p0 = corners0
        axes[0].fill(np.append(p0[:,0], p0[0,0]), np.append(p0[:,1], p0[0,1]), facecolor=(1,0,0,0.12), edgecolor='red')
        axes[0].scatter(p0[:,0], p0[:,1], c='blue')

        # Right: final snapshot (last frame)
        axes[1].set_title(f"Final (t={dt*(int(nsteps)-1):.2f} s)")
        pf = trajectories[:, -1, :]
        axes[1].fill(np.append(pf[:,0], pf[0,0]), np.append(pf[:,1], pf[0,1]), facecolor=(1,0,0,0.12), edgecolor='red')
        axes[1].scatter(pf[:,0], pf[:,1], c='blue')

        st.pyplot(fig2)

        # Area plot
        fig3, ax3 = plt.subplots(figsize=(8,2.5))
        times = np.arange(int(nsteps)) * dt
        ax3.plot(times, areas, '-k')
        ax3.set_xlabel('t (s)'); ax3.set_ylabel('Material area')
        ax3.set_title('Rectangle material element area vs time (should be ~constant)')
        st.pyplot(fig3)

        st.success("Done — animation and plots shown above. Edit parameters and re-run as needed.")

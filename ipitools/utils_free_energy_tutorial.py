import numpy as np
import matplotlib.pyplot as plt
from ipitools.io_ipi import read_ipi_output
from pathlib import Path
from ipi.utils.units import Constants, unit_to_user, unit_to_internal
from ipi.pes.morse import MorseHarmonic_driver
from ipi.utils.io import read_file
plt.style.use(Path(__file__).parent/'jlab.mplstyle')


def plot_pes_morse(D, a, z0, k, save_path=None):
    """
    Plot a two-dimensional Morse+harmonic potential energy surface and a 1D cut.

    Parameters
    ----------
    D : float
        Depth of the Morse well [Hartree].
    a : float
        Range parameter of the Morse potential [a0^-1].
    z0 : float
        Equilibrium position of the Morse potential [a0].
    k : float
        Force constant of the harmonic potential in x [Hartree/a0^2].
    show : bool
        If True, display the figure using plt.show().
    save_path : str or Path, optional
        If provided, save the figure to this path.

    Notes
    -----
    - The left panel shows contours of U(x, z) = Morse(z) + 0.5*k*(x - x0)^2.
    - The right panel shows the 1D Morse potential U(z) along z.
    """
    # Grid for 2D PES
    x = np.linspace(-2.0, 2.0, 300)
    z = np.linspace(0.0, 6.0, 300)
    X, Z = np.meshgrid(x, z)
    PES = MorseHarmonic_driver(D, a, z0, k)
    q = np.stack([X, np.zeros_like(X), Z], axis=-1)
    V_morse = PES.potential(q)
    # 1D cut along z
    z_cut = z
    q = np.stack(2*[np.zeros_like(z_cut)]+[z_cut], axis=-1)
    V_z = PES.potential(q)

    # Plot
    fig, (ax2D, ax1D) = plt.subplots(1, 2, figsize=(9, 4))

    # Left: 2D PES
    im = ax2D.contour(X, Z, V_morse, levels=np.linspace(0, 0.014, 15), cmap='viridis')
    ax2D.set_title("Model PES ($x$--$z$ plane)")
    ax2D.set_xlabel("$x~[a_0]$")
    ax2D.set_ylabel("$z~[a_0]$")
    ax2D.set_aspect('equal')
    fig.colorbar(im, ax=ax2D, label=r"$U~[\mathrm{Ha}]$")

    # Right: 1D Morse(z)
    ax1D.plot(z_cut, V_z, color='tab:red', lw=2)
    ax1D.axvline(PES.z0, ls='--', color='gray')
    ax1D.set_title("1D Morse Potential (along $z$)")
    ax1D.set_xlabel("$z~[a_0]$")
    ax1D.set_ylabel(r"$U(z)~[\mathrm{Ha}]$")
    ax1D.grid(True)
    ax1D.set_ylim(-0.001, 0.01)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)


def plot_geometry_opt(filepath, save_path=None):
    """
    Plot potential energy and force modulus vs. optimization step.

    Parameters
    ----------
    filepath : str or Path
        Path to the i-PI output file containing 'potential' and 'forcemod' data.
    save_path : str or Path, optional
        If provided, the figure will be saved to this path.
    """
    try:
        data = read_ipi_output(filepath)
    except FileNotFoundError:
        print(f"Could not locate the output file {filepath}, falling back to pre-computed results")
        backup = Path(__file__).parents[1] / '02-free-energy' / 'results' / 'geometry_optimization' / 'simulation.out'
        data = read_ipi_output(backup)

    U = data['potential']
    F = data['forcemod']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)

    ax1.plot(U, marker='o', lw=1)
    ax1.set_yscale('log')
    ax1.set_xlabel("Optimization step")
    ax1.set_ylabel("Potential Energy [Ha]")

    ax2.plot(F, marker='s', lw=1, color='tab:red')
    ax2.set_yscale('log')
    ax2.set_xlabel("Optimization step")
    ax2.set_ylabel("Force Modulus [Ha/Bohr]")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)


def compare_pes_1d_z(hess, q0, V0,
                     D=0.2, a=0.75, k=1.0,
                     span_z=(-2.0, 3.0), npts=300,
                     save_path=None):
    """
    Compare Morse+harmonic vs. 3D harmonic potential along the z direction only.

    Parameters
    ----------
    hess : ndarray of shape (3, 3)
        Hessian matrix in atomic units.
    q0 : ndarray of shape (3,)
        Reference coordinates [a0].
    V0 : float
        Potential energy at q0 [Hartree].
    D : float, optional
        Depth of the Morse well [eV] (default 0.2 eV).
    a : float, optional
        Range parameter of the Morse potential [AA^-1] (default 3.042).
    k : float, optional
        Harmonic spring constant [eV/AA^2] (default 17.5).
    span_z : tuple of two floats, optional
        (zmin, zmax) range for z-axis, in atomic units, relative to q0[2] (default (-2.0, 3.0)).
    npts : int, optional
        Number of points along z (default 300).
    save_path : str or Path, optional
        If provided, save the figure to this path.

    Notes
    -----
    - The reference potential is U_ref(x0,y0,z) = V0 + 0.5 * (dq)^T hess (dq).
    - The new potential is U_new(x0,y0,z) = Morse(z) + 0.5 * k_xy * 0,
      where k_xy = 0.5*(hess[0,0] + hess[1,1]) and x=x0, y=y0.
    - Both curves and their difference are plotted in two stacked subplots.
    """

    # anharmonic potential
    z0 = q0[-1]
    PES = MorseHarmonic_driver(D, a, unit_to_user("length", "angstrom", z0), k)
    # harmonic approximation
    def pes_h(x, y, z):
        dq = np.array([x, y, z]) - q0
        return V0 + 0.5 * np.linalg.multi_dot((dq, hess, dq))

    # build grid
    x0, y0, z0 = q0
    z_vals = np.linspace(z0 + span_z[0],
                         z0 + span_z[1], npts)
    q = np.stack([x0*np.ones_like(z_vals), y0*np.ones_like(z_vals), z_vals], axis=-1)
    
    # pointwise evaluation along z
    V_h_z = np.array([pes_h(x0, y0, z) for z in z_vals])
    V_anh_z = PES.potential(q)

    # plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(z_vals, V_h_z,    label='Harmonic',    lw=2)
    ax.plot(z_vals, V_anh_z,    label='Morse', lw=2, ls='--')
    ax.set_ylabel(f'$U(z)$ [Ha]')
    ax.set_xlabel(f'$z$ [a$_0$]')
    ax.legend()
    ax.set_title(r'$U(\mathbf{q})$ at $\mathbf{q} = (x_0, \, y_0, \, z)$')    
    ax.set_ylim((0,0.005))

    if save_path:
        fig.savefig(save_path, dpi=300)


def plot_boltzmann_distribution_morse_vs_harmonic(
        T_low, T_high, span_z=None, save_path=None,
        D=0.2, a=3.042, z0=1.111, k=17.5):
    """
    Plot Boltzmann-distributed histograms for Morse vs. harmonic potentials at two temperatures.
    """

    MorsePES = MorseHarmonic_driver(D, a, z0, k)
    k_r = 2 * MorsePES.De * MorsePES.a**2
    z0 = MorsePES.z0

    def V_harm(z):
        dz = z - z0
        return 0.5 * k_r * dz**2

    V_morse = lambda z: MorsePES.potential(np.stack(2*[np.zeros_like(z)] + [z], axis=-1))

    if span_z is None:
        span_z = (-0.6, 0.8)
    zmin, zmax = z0 + np.asarray(span_z)
    z = np.linspace(zmin, zmax, 1000)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=False, sharex=True)
    axt = [ax.twinx() for ax in axs]
    from scipy.integrate import quad
    for T, ax, twin in zip([T_low, T_high], axs, axt):
        beta = 1 / (Constants.kb * unit_to_internal("temperature", "kelvin", T))
        ax.set_title(f"T = {T:.0f} K")
        for PES, Plabel, Vlabel, Vstyle in zip(
            [V_morse, V_harm],
            ["Morse", "Harmonic"],
            [r"$U_{\mathrm{Morse}}$", r"$U_{\mathrm{HO}}$"],
            ['k-', 'r--']
        ):
            distribution = lambda z: np.exp(-beta * PES(z))
            partf = quad(distribution, zmin - 0.5, zmax + 1.0)[0]
            ax.fill_between(z, distribution(z) / partf, label=Plabel, alpha=0.5)
            twin.plot(z, PES(z), Vstyle, lw=2, label=Vlabel)
            ax.set_xlabel("$z~[a_0]$")
            ax.set_ylabel("Probability Density")
            twin.set_ylabel("Potential [Ha]")
            ax.set_xlim(zmin, zmax)
            twin.set_ylim(-0.0001, 0.0051)

    axs[0].legend(loc="upper left")
    axt[0].legend(loc="upper right")

    fig.suptitle("Boltzmann Sampling: Morse vs. Harmonic Approximation")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)


def TI_integrand(dir_list, base_path, filename, skip=50, save_path=None):
    """
    Plot the thermodynamic integration (TI) integrand with error bars and return raw arrays.

    Parameters
    ----------
    dir_list : list of str
        List of lambda subdirectory names (e.g. ['0.0', '0.2', ..., '1.0']).
    base_path : str or Path
        Base directory containing the lambda subfolders.
    filename : str
        Name of the file (e.g. 'simulation.pots') inside each lambda folder.
    skip : int, optional
        Number of initial frames to discart (thermalization)
    save_path : str or Path, optional
        If provided, save the figure to this path.

    Returns
    -------
    du_list : ndarray of shape (N,)
        Mean values of U1 - U0 (difference in potential components) at each lambda, in Hartree units.
    duerr_list : ndarray of shape (N,)
        Standard error of the mean for the difference at each lambda, in Hartree units.
    """
    du_list = []
    duerr_list = []
    l_list = [float(l) for l in dir_list]

    for x in dir_list:
        fullpath = Path(base_path).joinpath(x, filename)
        data = read_ipi_output(fullpath)
        du = data['pot_component_raw(1)'] - data['pot_component_raw(0)']
        du = du[skip:]
        du_list.append(du.mean())
        duerr_list.append(du.std(ddof=1) / (np.sqrt(len(du))))

    du_list = np.asarray(du_list)
    duerr_list = np.asarray(duerr_list)

    du_kj = unit_to_user("energy", "j/mol", du_list/1000)
    err_kj = unit_to_user("energy", "j/mol", duerr_list/1000)

    # Plot
    plt.figure(figsize=(5, 3))
    plt.plot(l_list, du_kj, color='blue', label='TI integrand')
    plt.fill_between(l_list,
                     (du_kj - err_kj),
                     (du_kj + err_kj),
                     color='blue', alpha=0.2)
    plt.title("Thermodynamic Integration")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\langle U_{\mathrm{a}} - U_{\mathrm{h}} \rangle_{\lambda}$ [kJ/mol]")
    plt.grid(True)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    return du_list, duerr_list


def print_energy(energy, error=None):
    """Pretty print energy in internal i-PI units (Ha), converting to
    mHa, meV and kJ/mol.

    """

    Ha_to_meV = unit_to_user("energy", "electronvolt", 1000.0)
    Ha_to_kJmol = unit_to_user("energy", "j/mol", 1.0)/1000

    if error is None:
        print(f"   {energy*1000:.3f} [mHa]")
        print(f"   {energy*Ha_to_meV:.2f} [meV]")
        print(f"   {energy*Ha_to_kJmol:.3f} [kJ/mol]\n")
    else:
        print(f"   {energy*1000:.3f} ± {error*1000:.3f} [mHa]")
        print(f"   {energy*Ha_to_meV:.2f} ± {error*Ha_to_meV:.2f} [meV]")
        print(f"   {energy*Ha_to_kJmol:.3f} ± {error*Ha_to_kJmol:.3f} [kJ/mol]\n")



def mass_TI_integrand(path,
                      temperature,
                      dir_list=['0.2','0.4','0.6','0.8','1.0'],
                      skip=50,
                      save_path=None):
    """
    Compute the mass thermodynamic integration (Mass-TI) integrand, drop the first
    half of each kinetic energy window, plot results, and return raw arrays.

    Parameters
    ----------
    tutorialpath : str or Path
        Path to the 'mass_ti' directory containing subfolders for each g value.
    temperature : float
        Temperature in Kelvin.
    dir_list : list of str, optional
        List of g values (as strings) for subdirectories (default ['0.2','0.4',...,'1.0']).
    skip : int, optional
        Number of initial frames to skip (thermalisation)
    save_path : str or Path, optional
        If provided, save the figure to this path.

    Returns
    -------
    du_list : ndarray of shape (N,)
        Integrand values ( (mean kinetic - analytic classical kinetic)/g ) in Hartree units.
    stderr_list : ndarray of shape (N,)
        Corresponding standard error values in Hartree units.
    g_list : ndarray of shape (N,)
        Array of float g values used.
    """
    # Conversion factors
    beta = 1/(Constants.kb * unit_to_internal("temperature", "kelvin", temperature))

    # Prepare arrays
    g_list      = np.array([float(d) for d in dir_list])
    du_list     = []  # (mean kinetic )/g, in Hartree
    stderr_list = []  # standard error of integrand, in Hartree

    # Analytic classical kinetic energy at temperature T [Ha] (1D)
    K_cl_exact = 1/(2*beta)
    du_list = [0.0]         # g = 0, classical limit
    stderr_list = [0.0]     # first point known exactly

    def read_Kcv_z(filename):
        file_handle = open(filename, "r")
        Kcv_zz = []
        while True:
            try:
                ret = read_file("xyz", file_handle)
                Kcv_zz.append(ret["atoms"].q[-1]) 
            except EOFError:
                break
            except:
                raise
        return np.asarray(Kcv_zz)


    for dir_str, g in zip(dir_list, g_list):
        K_q = read_Kcv_z(f"{path}/{dir_str}/simulation.cv.xyz")[skip:]
        mean_q = K_q.mean()
        du_list.append((mean_q - K_cl_exact) / g)
        stderr_q = K_q.std(ddof=1) / np.sqrt(len(K_q))
        stderr_list.append(stderr_q / g)

    g_list = np.concatenate(([0.0], g_list))
    du_list = np.array(du_list)
    stderr_list = np.array(stderr_list)

    # Plot the integrand in kJ/mol
    plt.figure(figsize=(8, 5))
    plt.errorbar(g_list,
                 unit_to_user("energy", "j/mol", 1.0e-3) * du_list,
                 yerr = unit_to_user("energy", "j/mol", 1.0e-3) * stderr_list,
                 fmt='o-',
                 capsize=3)
    plt.xlabel("g")
    plt.ylabel(r"$(\langle K(g) \rangle - \langle K \rangle^{\mathrm{c}})/g $ [kJ/mol]")
    plt.title("Mass-TI Integrand")
    plt.grid(alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    return du_list, stderr_list, g_list


def plot_free_energy_decomposition(
        F_ch, 
        dF_ch2ca, dF_ch2ca_err, 
        dF_ca2qa, dF_ca2qa_err,
        F_qh,
        dF_qh2qa, dF_qh2qa_err, 
        F_ca_exact, F_qa_exact):
    
    fig, (ax0, ax1) = plt.subplots(
        ncols=2, sharey=True, gridspec_kw={'width_ratios': [5, 3]})
    fig.set_size_inches((8, 5))
    
    Ha_to_kJmol = unit_to_user("energy", "j/mol", 1.0e-3)
    F_ch = F_ch * Ha_to_kJmol
    dF_ch2ca = dF_ch2ca * Ha_to_kJmol
    dF_ca2qa = dF_ca2qa * Ha_to_kJmol
    dF_ch2ca_err = dF_ch2ca_err * Ha_to_kJmol
    dF_ca2qa_err = dF_ca2qa_err * Ha_to_kJmol
    F_qh = F_qh * Ha_to_kJmol
    dF_qh2qa = dF_qh2qa * Ha_to_kJmol
    dF_qh2qa_err = dF_qh2qa_err * Ha_to_kJmol
    F_ca_exact = F_ca_exact * Ha_to_kJmol
    F_qa_exact = F_qa_exact * Ha_to_kJmol

    labels = [r'$F_{\mathrm{ch}}$', r'$\Delta F_{\mathrm{ch} \to \mathrm{ca}} $', 
              r'$F_{\mathrm{ca}}$', r'$\Delta F_{\mathrm{ca} \to \mathrm{qa}} $',
              r'$F_{\mathrm{qa}}$']
    colors = ["tab:green", "tab:pink", "tab:red", "tab:cyan", "tab:blue"] 
    yerr = [np.nan, dF_ch2ca_err, dF_ch2ca_err, dF_ca2qa_err, 
            np.sqrt(dF_ch2ca_err**2 + dF_ca2qa_err**2)]
    F_ca = F_ch + dF_ch2ca
    yvals = [F_ch, dF_ch2ca, F_ca, dF_ca2qa, F_ca + dF_ca2qa]
    
    ax0.bar(labels, yvals, yerr=yerr, capsize=5, color=colors)
    ax0: plt.Axes
    ax0.axhline(F_ca_exact, c="tab:red", ls="--")
    ax0.axhline(F_qa_exact, c="tab:blue", ls="--")
    ax0.set_title("Mass-TI route")
    ax0.set_ylabel('Free energy [kJ/mol]')

    labels = [r'$F_{\mathrm{qh}}$', r'$\Delta F_{\mathrm{qh} \to \mathrm{qa}} $', 
              r'$F_{\mathrm{qa}}$']
    colors = ["tab:orange", "tab:cyan", "tab:blue"] 
    yerr = [np.nan, dF_qh2qa_err, dF_qh2qa_err]
    yvals = [F_qh, dF_qh2qa, F_qh + dF_qh2qa]
    ax1.bar(labels, yvals, yerr=yerr, capsize=5, color=colors)
    ax1: plt.Axes
    ax1.axhline(F_qa_exact, c="tab:blue", ls="--")
    ax1.set_title("PIMD-TI route")


def plot_MassTI_free_energy_decomposition(
        F_ch, 
        dF_ch2ca, dF_ch2ca_err, 
        dF_ca2qa, dF_ca2qa_err,
        F_ca_exact, F_qa_exact):

    fig, ax = plt.subplots()
    fig.set_size_inches((8, 5))

    Ha_to_kJmol = unit_to_user("energy", "j/mol", 1.0e-3)
    F_ch = F_ch * Ha_to_kJmol
    dF_ch2ca = dF_ch2ca * Ha_to_kJmol
    dF_ca2qa = dF_ca2qa * Ha_to_kJmol
    dF_ch2ca_err = dF_ch2ca_err * Ha_to_kJmol
    dF_ca2qa_err = dF_ca2qa_err * Ha_to_kJmol
    F_ca_exact = F_ca_exact * Ha_to_kJmol
    F_qa_exact = F_qa_exact * Ha_to_kJmol

    F_ca = F_ch + dF_ch2ca
    F_qa = F_ca + dF_ca2qa

    yvals = [F_ch, dF_ch2ca, F_ca, dF_ca2qa, F_qa]
    yerr = [np.nan, dF_ch2ca_err, dF_ch2ca_err, dF_ca2qa_err, 
            np.sqrt(dF_ch2ca_err**2 + dF_ca2qa_err**2)]

    labels = [r'$F_{\mathrm{ch}}$', 
              r'$\Delta F_{\mathrm{ch} \to \mathrm{ca}}$', 
              r'$F_{\mathrm{ca}}$', 
              r'$\Delta F_{\mathrm{ca} \to \mathrm{qa}}$', 
              r'$F_{\mathrm{qa}}$']

    colors = ["tab:green", "tab:pink", "tab:red", "tab:cyan", "tab:blue"]

    ax.bar(labels, yvals, yerr=yerr, capsize=5, color=colors)

    ax.axhline(F_ca_exact, c="tab:red", ls="--", label=r'$F_{\mathrm{ca}}^{\mathrm{exact}}$')
    ax.axhline(F_qa_exact, c="tab:blue", ls="--", label=r'$F_{\mathrm{qa}}^{\mathrm{exact}}$')

    ax.set_ylabel('Free energy [kJ/mol]')
    ax.set_title("Mass-TI Free Energy Decomposition")
    ax.legend()
    fig.tight_layout()
    plt.show()



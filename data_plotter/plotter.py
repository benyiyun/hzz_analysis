import os
import time
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
from matplotlib.ticker import AutoMinorLocator

# Shared directory for communication
SHARED_DIR = "/app/shared/"
INPUT_FILE = os.path.join(SHARED_DIR, "processed_data.npy")
OUTPUT_PLOT = os.path.join(SHARED_DIR, "output_plot.png")

# Wait for the processed data file
while not os.path.exists(INPUT_FILE):
    print("Waiting for processed data...")
    time.sleep(5)

# Load processed data from .npy file
all_data = np.load(INPUT_FILE, allow_pickle=True).item()

print("Loaded processed data. Generating plot...")

# Define constants
lumi = 10
GeV = 1.0
fraction = 1.0

# X-axis range of the plot
xmin, xmax = 80 * GeV, 250 * GeV
step_size = 5 * GeV
bin_edges = np.arange(start=xmin, stop=xmax + step_size, step=step_size)
bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

# For identification and naming
samples = {

    'data': {
        'list' : ['data_A','data_B','data_C','data_D'], # data is from 2016, first four periods of data taking (ABCD)
    },

    r'Background $Z,t\bar{t}$' : { # Z + ttbar
        'list' : ['Zee','Zmumu','ttbar_lep'],
        'color' : "#6b59d3" # purple
    },

    r'Background $ZZ^*$' : { # ZZ
        'list' : ['llll'],
        'color' : "#ff0000" # red
    },

    r'Signal ($m_H$ = 125 GeV)' : { # H -> ZZ -> llll
        'list' : ['ggH125_ZZ4lep','VBFH125_ZZ4lep','WH125_ZZ4lep','ZH125_ZZ4lep'],
        'color' : "#00cdff" # light blue
    },

}

# Plot data
data_x, _ = np.histogram(all_data['data']['mass'], bins=bin_edges)
data_x_errors = np.sqrt(data_x)

# Plot signal
signal_x = np.array(all_data['Signal ($m_H$ = 125 GeV)']['mass'])
signal_weights = np.ones_like(signal_x)
signal_color = samples['Signal ($m_H$ = 125 GeV)']['color']

# Prepare MC background samples
mc_x = []
mc_weights = []
mc_colors = []
mc_labels = []

for sample_name, sample_data in all_data.items():
    if sample_name not in ['data', 'Signal ($m_H$ = 125 GeV)']:
        mc_x.append(np.array(sample_data['mass']))
        mc_weights.append(np.ones_like(mc_x[-1]))  # Use ones if weights are missing
        mc_colors.append(samples[sample_name]['color'])
        mc_labels.append(sample_name)

# *************
# Main plot
# *************
plt.figure(figsize=(10, 7))
main_axes = plt.gca()

# Plot data points
main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors, fmt='ko', label='Data')

# Plot MC backgrounds
mc_heights = main_axes.hist(mc_x, bins=bin_edges, weights=mc_weights, stacked=True, color=mc_colors, label=mc_labels)
mc_x_tot = mc_heights[0][-1]

# Compute MC statistical uncertainty
mc_x_err = np.sqrt(np.histogram(np.hstack(mc_x), bins=bin_edges, weights=np.hstack(mc_weights) ** 2)[0])

# Plot the signal
main_axes.hist(signal_x, bins=bin_edges, bottom=mc_x_tot, weights=signal_weights, color=signal_color, label=r'Signal ($m_H$ = 125 GeV)')

# Plot statistical uncertainty
main_axes.bar(bin_centres, 2 * mc_x_err, alpha=0.5, bottom=mc_x_tot - mc_x_err, color='none', hatch="////", width=step_size, label='Stat. Unc.')

# Set limits and labels
main_axes.set_xlim(left=xmin, right=xmax)
main_axes.xaxis.set_minor_locator(AutoMinorLocator())
main_axes.tick_params(which='both', direction='in', top=True, right=True)
main_axes.set_xlabel(r'4-lepton invariant mass $\mathrm{m_{4l}}$ [GeV]', fontsize=13, x=1, horizontalalignment='right')
main_axes.set_ylabel(f'Events / {step_size} GeV', y=1, horizontalalignment='right')
main_axes.set_ylim(bottom=0, top=np.amax(data_x) * 1.6)
main_axes.yaxis.set_minor_locator(AutoMinorLocator())

# Annotations
plt.text(0.05, 0.93, 'ATLAS Open Data', transform=main_axes.transAxes, fontsize=13)
plt.text(0.05, 0.88, 'for education', transform=main_axes.transAxes, style='italic', fontsize=8)
plt.text(0.05, 0.82, f'$\sqrt{{s}}$=13 TeV, $\int$L dt = {lumi * fraction} fb$^{{-1}}$', transform=main_axes.transAxes)
plt.text(0.05, 0.76, r'$H \rightarrow ZZ^* \rightarrow 4\ell$', transform=main_axes.transAxes)

# Draw legend
main_axes.legend(frameon=False)

# Save the plot
plt.savefig(OUTPUT_PLOT)
print(f"Plot saved to {OUTPUT_PLOT}")
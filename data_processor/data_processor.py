import uproot
import awkward as ak
import vector
import os
import time
import infofile
import numpy as np
import concurrent.futures  # Import concurrent.futures

MeV = 0.001  # Convert energy to GeV
lumi = 10  # Set luminosity for MC data
fraction = 1.0

# Define shared directory for file-based communication
SHARED_DIR = "/app/shared/"
OUTPUT_FILE = os.path.join(SHARED_DIR, "processed_data.npy")

# Ensure the shared directory exists
os.makedirs(SHARED_DIR, exist_ok=True)

path = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/"

samples = {
    'data': {'list': ['data_A', 'data_B', 'data_C', 'data_D']},
    r'Background $Z,t\bar{t}$': {'list': ['Zee', 'Zmumu', 'ttbar_lep'], 'color': "#6b59d3"},
    r'Background $ZZ^*$': {'list': ['llll'], 'color': "#ff0000"},
    r'Signal ($m_H$ = 125 GeV)': {'list': ['ggH125_ZZ4lep', 'VBFH125_ZZ4lep', 'WH125_ZZ4lep', 'ZH125_ZZ4lep'], 'color': "#00cdff"},
}

# Cut functions
def cut_lep_type(lep_type):
    sum_lep_type = lep_type[:, 0] + lep_type[:, 1] + lep_type[:, 2] + lep_type[:, 3]
    return (sum_lep_type != 44) & (sum_lep_type != 48) & (sum_lep_type != 52)

def cut_lep_charge(lep_charge):
    return lep_charge[:, 0] + lep_charge[:, 1] + lep_charge[:, 2] + lep_charge[:, 3] != 0

def calc_mass(lep_pt, lep_eta, lep_phi, lep_E):
    p4 = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "E": lep_E})
    return (p4[:, 0] + p4[:, 1] + p4[:, 2] + p4[:, 3]).M * MeV

def calc_weight(weight_variables, sample, events):
    info = infofile.infos[sample]
    xsec_weight = (lumi * 1000 * info["xsec"]) / (info["sumw"] * info["red_eff"])  # *1000 to go from fb-1 to pb-1
    total_weight = xsec_weight 
    for variable in weight_variables:
        total_weight = total_weight * events[variable]
    return total_weight

# Function to process a single sample
def process_sample(sample_name, val, path, fraction, weight_variables):
    if sample_name == 'data':
        prefix = "Data/"
    else:
        prefix = "MC/mc_" + str(infofile.infos[val]["DSID"]) + "."
    fileString = path + prefix + val + ".4lep.root"  # file name to open
    print(f"\tProcessing: {fileString}")

    start = time.time() 
    print(f"\t{val}:")

    with uproot.open(fileString + ":mini") as t:
        tree = t

        sample_data = []
        for data in tree.iterate(variables + weight_variables, 
                    library="ak", 
                    entry_stop=tree.num_entries * fraction,  # process up to numevents*fraction
                    step_size=1000000):  # Number of events in this batch
            data['leading_lep_pt'] = data['lep_pt'][:, 0]
            data['sub_leading_lep_pt'] = data['lep_pt'][:, 1]
            data['third_leading_lep_pt'] = data['lep_pt'][:, 2]
            data['last_lep_pt'] = data['lep_pt'][:, 3]

            lep_type = data['lep_type']
            data = data[~cut_lep_type(lep_type)]
            lep_charge = data['lep_charge']
            data = data[~cut_lep_charge(lep_charge)]
            
            data['mass'] = calc_mass(data['lep_pt'], data['lep_eta'], data['lep_phi'], data['lep_E'])

            if 'data' not in val:  # Only calculates weights if the data is MC
                data['totalWeight'] = calc_weight(weight_variables, val, data)
                nOut = sum(data['totalWeight'])  # sum of weights passing cuts in this batch 
            else:
                nOut = len(data)
            elapsed = time.time() - start  # time taken to process
            print(f"\t\t nIn: {len(data)},\t nOut: \t{nOut}\t in {round(elapsed, 1)}s")  # events before and after

            sample_data.append(data)

    return ak.concatenate(sample_data)

# Read and process data using concurrent.futures
all_data = {}
variables = ["lep_type", "lep_charge", "lep_pt", "lep_eta", "lep_phi", "lep_E"]
weight_variables = ["mcWeight", "scaleFactor_PILEUP", "scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_LepTRIGGER"]

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for s in samples:
        print(f'Processing '+s+' samples')
        for val in samples[s]['list']:
            futures.append(executor.submit(process_sample, s, val, path, fraction, weight_variables))

    # Wait for all threads to complete and collect results
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if s not in all_data:
            all_data[s] = result
        else:
            all_data[s] = ak.concatenate([all_data[s], result])

# Save processed data to a .npy file in the shared volume
np.save(OUTPUT_FILE, all_data)

print(f"Processed data saved to {OUTPUT_FILE}")

    



import numpy as np
import pandas as pd
import torch
import os
from readdataset import ALDDataset_N

def create_dataset_csv(Nlist, output_file="thickness_profiles_N.csv", include_train=True, include_test=True, directory="./dataset", sample_fraction=1.0):
    """
    Creates a CSV file containing profile (N), tdose, and tsat values from dataset files.
    
    Args:
        Nlist: List of thickness values (N) to include
        output_file: Name of the output CSV file
        include_train: Whether to include training data
        include_test: Whether to include testing data
        directory: Directory containing the dataset files
        sample_fraction: Fraction of data to include (1.0 = all data, 0.5 = half of the data)
    """
    # Create a list to store all data
    all_data = []
    
    for N in Nlist:
        output_file_N = output_file #.replace("N",f"{N}")
        print(f"Processing thickness profile N={N}...")
        
        # Process training data if requested
        if include_train:
            try:
                # Load the dataset
                ald_train = ALDDataset_N(N, train=True, directory=directory)
                
                # Get all data at once
                profile, tdose, tsat = ald_train[:]

                
                # Convert from PyTorch tensors to numpy arrays if needed
                if isinstance(profile, torch.Tensor):
                    profile = profile.numpy()
                if isinstance(tdose, torch.Tensor):
                    tdose = tdose.numpy()
                if isinstance(tsat, torch.Tensor):
                    tsat = tsat.numpy()

                # Convert from log scale if the dataset uses it
                tdose_values = np.exp(tdose)
                tsat_values = np.exp(tsat)

                #print(profile.shape, profile)
                #print(tdose.shape, tdose_values.shape, tdose)
                #print(tsat.shape, tsat_values.shape, tsat)
                
                # Add data to the list (with sampling)
                num_samples = int(len(tdose_values) * sample_fraction)
                indices = np.random.choice(len(tdose_values), num_samples, replace=False) if sample_fraction < 1.0 else range(len(tdose_values))
                
                for i in indices:
                    all_data.append({
                        'mean_thickness': np.mean(profile[i]),
                        'std_thickness': np.std(profile[i]),
                        'tdose': tdose_values[i][0],
                        'tsat': tsat_values[i][0]
                    })
                
                print(f"  Added {len(tdose_values)} training samples for N={N}")
            except Exception as e:
                print(f"Error processing training data for N={N}: {e}")
        
        # Process testing data if requested
        if include_test:
            try:
                # Load the dataset
                ald_test = ALDDataset_N(N, train=False, directory=directory)
                
                # Get all data at once
                profile, tdose, tsat = ald_test[:]
                
                # Convert from PyTorch tensors to numpy arrays if needed
                if isinstance(profile, torch.Tensor):
                    profile = profile.numpy()
                if isinstance(tdose, torch.Tensor):
                    tdose = tdose.numpy()
                if isinstance(tsat, torch.Tensor):
                    tsat = tsat.numpy()
                
                # Convert from log scale if the dataset uses it
                tdose_values = np.exp(tdose)
                tsat_values = np.exp(tsat)
                
                # Add data to the list (with sampling)
                num_samples = int(len(tdose_values) * sample_fraction)
                indices = np.random.choice(len(tdose_values), num_samples, replace=False) if sample_fraction < 1.0 else range(len(tdose_values))
                
                for i in indices:
                    all_data.append({
                        'mean_thickness': np.mean(profile[i]),
                        'std_thickness': np.std(profile[i]),
                        'tdose': tdose_values[i][0],
                        'tsat': tsat_values[i][0]
                    })
                
                print(f"  Added {len(tdose_values)} testing samples for N={N}")
            except Exception as e:
                print(f"Error processing testing data for N={N}: {e}")
    
            # Create a DataFrame and save to CSV
            if all_data:
                df = pd.DataFrame(all_data)
                df.to_csv(output_file_N, index=False)
                print(f"Successfully created CSV file: {output_file_N}")
                print(f"Total samples: {len(all_data)}")
            else:
                print("No data was collected. Please check your dataset files.")
        
        
if __name__ == "__main__":
    # List of thickness values to include
    Nlist = [4]
    
    # Set random seed for reproducible sampling
    np.random.seed(42)
    
    # Create the CSV file (with all data by default)
    create_dataset_csv(Nlist, output_file=f"thickness_profiles_{Nlist[0]}.csv", sample_fraction=0.005)
    
    # Optional: Create a smaller dataset with only a fraction of the data
    # Uncomment the line below to create a dataset with only 20% of the data
    # create_dataset_csv(Nlist, output_file="thickness_profiles_sampled.csv", sample_fraction=0.2)
    
    # Print a sample of the data
    try:
        df = pd.read_csv(f"thickness_profiles_{Nlist[0]}.csv")
        print("\nSample of the data:")
        print(df.head(10))
        print("\nSummary statistics:")
        print(df.describe())
    except Exception as e:
        print(f"Error loading the created CSV file: {e}")

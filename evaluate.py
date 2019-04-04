import os
import argparse
import json

import numpy as np
import torch.utils.data as Data

from metrics import *
from ped_dataset import *


class RunConfig:
    sequence_length = 20
    min_sequence_length = 10
    observed_history = 8

    sample = False
    num_samples = 20
    sample_angle_std = 25

    dataset_paths = [
                     "./data/eth_univ",
                     "./data/eth_hotel",
                     "./data/ucy_zara01",
                     "./data/ucy_zara02",
                     "./data/ucy_univ"
                    ]

def rel_to_abs(rel_traj, start_pos):
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)

def constant_velocity_model(observed, sample=False):
    """
    CVM can be run with or without sampling. A call to this function always
    generates one sample if sample option is true.
    """
    obs_rel = observed[1:] - observed[:-1]
    deltas = obs_rel[-1].unsqueeze(0)
    if sample:
            sampled_angle = np.random.normal(0, RunConfig.sample_angle_std, 1)[0]
            theta = (sampled_angle * np.pi)/ 180.
            c, s = np.cos(theta), np.sin(theta)
            rotation_mat = torch.tensor([[c, s],[-s, c]])
            deltas = torch.t(rotation_mat.matmul(torch.t(deltas.squeeze(dim=0)))).unsqueeze(0)
    y_pred_rel = deltas.repeat(12, 1, 1)
    return y_pred_rel

def evaluate_testset(testset):
    testset_loader = Data.DataLoader(dataset=testset, batch_size=1, shuffle=True)

    with torch.no_grad():

        avg_displacements =  []
        final_displacements = []
        for seq_id, (batch_x, batch_y) in enumerate(testset_loader):

            observed = batch_x.permute(1, 0, 2)
            y_true_rel, masks = batch_y
            y_true_rel = y_true_rel.permute(1, 0, 2)

            sample_avg_disp = []
            sample_final_disp = []
            samples_to_draw = RunConfig.num_samples if RunConfig.sample else 1
            for i in range(samples_to_draw):

                # predict and convert to absolute
                y_pred_rel = constant_velocity_model(observed, sample=RunConfig.sample)
                y_pred_abs = rel_to_abs(y_pred_rel, observed[-1])
                predicted_positions = y_pred_abs.permute(1, 0, 2)

                # convert true label to absolute
                y_true_abs = rel_to_abs(y_true_rel, observed[-1])
                true_positions = y_true_abs.permute(1, 0, 2)

                # compute errors
                avg_displacement = avg_disp(predicted_positions, [true_positions, masks])
                final_displacement = final_disp(predicted_positions, [true_positions, masks])
                sample_avg_disp.append(avg_displacement)
                sample_final_disp.append(final_displacement)

            avg_displacement = min(sample_avg_disp)
            final_displacement = min(sample_final_disp)
            avg_displacements.append(avg_displacement)
            final_displacements.append(final_displacement)

        avg_displacements = np.mean(avg_displacements)
        final_displacements = np.mean(final_displacements)
        return avg_displacements, final_displacements

def load_datasets():
    datasets = []
    datasets_size = 0
    for dataset_path in RunConfig.dataset_paths:
        dataset_path = dataset_path.replace('~', os.environ['HOME'])
        print("Loading dataset {}".format(dataset_path))
        dataset = PedDataset(dataset_path=dataset_path, sequence_length=RunConfig.sequence_length, observed_history=RunConfig.observed_history, \
                              min_sequence_length=RunConfig.min_sequence_length)
        datasets.append(dataset)
        datasets_size += len(dataset)
    print("Size of all datasets: {}".format(datasets_size))
    return datasets

def parse_commandline():
    parser = argparse.ArgumentParser(description='Runs an evaluation of the Constant Velocity Model.')
    parser.add_argument('--sample', default=RunConfig.sample, action='store_true', help='Turns on the sampling for the CVM (OUR-S).')
    args = parser.parse_args()
    return args

def main():
    args = parse_commandline()
    RunConfig.sample = args.sample
    if RunConfig.sample:
        print("Sampling activated.")

    datasets = load_datasets()
    testset_results = []
    for i, testset in enumerate(datasets):
        print("Evaluating testset {}".format(testset.name))
        avg_displacements, final_displacements = evaluate_testset(testset)
        testset_results.append([testset.name, avg_displacements, final_displacements])
    
    print("\n== Results for testset evaluations ==")
    total_avg_disp, total_final_disp = 0, 0
    for name, avg_displacements, final_displacements in testset_results:
        print("- Testset: {}".format(name))
        print("ADE: {}".format(avg_displacements))
        print("FDE: {}".format(final_displacements))
        total_avg_disp += avg_displacements
        total_final_disp += final_displacements
    print("- Average")
    print("*ADE: {}".format(total_avg_disp/len(testset_results)))
    print("*FDE: {}".format(total_final_disp/len(testset_results)))

if __name__ == "__main__":
    main()

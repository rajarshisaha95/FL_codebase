"""
This script contains the evaluation scripts for obtaining different plots for federated learning simulations
with intermittent connectivity between clients
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot_fully_connected_sortpart3():

    comm_rounds = 1000
    arr_max_idx = int(comm_rounds / 5)

    # FedAvg with perfect connectivity
    FedAvg_NoDropout_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                                 'resnet20/fed_avg-sort_part_cls_3/Result_fed_avg-0.npy')
    FedAvg_NoDropout_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                                 'resnet20/fed_avg-sort_part_cls_3/Result_fed_avg-1.npy')
    FedAvg_NoDropout_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                                 'resnet20/fed_avg-sort_part_cls_3/Result_fed_avg-2.npy')
    FedAvg_NoDropout_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                                 'resnet20/fed_avg-sort_part_cls_3/Result_fed_avg-3.npy')
    FedAvg_NoDropout_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                                 'resnet20/fed_avg-sort_part_cls_3/Result_fed_avg-4.npy')
    FedAvg_NoDropout = 1 / 5 * (FedAvg_NoDropout_0[:arr_max_idx] + FedAvg_NoDropout_1[:arr_max_idx] +
                                FedAvg_NoDropout_2[:arr_max_idx] + FedAvg_NoDropout_3[:arr_max_idx] +
                                FedAvg_NoDropout_4[:arr_max_idx])

    # Blind FedAvg with intermittent connectivity
    FedAvg_Blind_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                             'resnet20/fed_avg_blind-sort_part_cls_3/Result_fed_avg_blind-0.npy')
    FedAvg_Blind_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                             'resnet20/fed_avg_blind-sort_part_cls_3/Result_fed_avg_blind-1.npy')
    FedAvg_Blind_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                             'resnet20/fed_avg_blind-sort_part_cls_3/Result_fed_avg_blind-2.npy')
    FedAvg_Blind_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                             'resnet20/fed_avg_blind-sort_part_cls_3/Result_fed_avg_blind-3.npy')
    FedAvg_Blind_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                             'resnet20/fed_avg_blind-sort_part_cls_3/Result_fed_avg_blind-4.npy')
    FedAvg_Blind = 1 / 5 * (FedAvg_Blind_0[:arr_max_idx] + FedAvg_Blind_1[:arr_max_idx] +
                            FedAvg_Blind_2[:arr_max_idx] + FedAvg_Blind_3[:arr_max_idx] +
                            FedAvg_Blind_4[:arr_max_idx])

    # Non-Blind FedAvg with intermittent connectivity
    FedAvg_NonBlind_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                                'resnet20/fed_avg_nonblind-sort_part_cls_3/Result_fed_avg_nonblind-0.npy')
    FedAvg_NonBlind_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                                'resnet20/fed_avg_nonblind-sort_part_cls_3/Result_fed_avg_nonblind-1.npy')
    FedAvg_NonBlind_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                                'resnet20/fed_avg_nonblind-sort_part_cls_3/Result_fed_avg_nonblind-2.npy')
    FedAvg_NonBlind_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                                'resnet20/fed_avg_nonblind-sort_part_cls_3/Result_fed_avg_nonblind-3.npy')
    FedAvg_NonBlind_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                                'resnet20/fed_avg_nonblind-sort_part_cls_3/Result_fed_avg_nonblind-4.npy')
    FedAvg_NonBlind = 1 / 5 * (FedAvg_NonBlind_0[:arr_max_idx] + FedAvg_NonBlind_1[:arr_max_idx]
                               + FedAvg_NonBlind_2[:arr_max_idx] + FedAvg_NonBlind_3[:arr_max_idx]
                               + FedAvg_NonBlind_4[:arr_max_idx])

    # ColRel with intermittent connectivity client-PS and client-client connections (p_{ij} = 0.5)
    ColRel_pt5_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                           'resnet20/colrel_int-sort_part_cls_3_FC_pt5/Result_colrel_int-0.npy')
    ColRel_pt5_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                           'resnet20/colrel_int-sort_part_cls_3_FC_pt5/Result_colrel_int-1.npy')
    ColRel_pt5_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                           'resnet20/colrel_int-sort_part_cls_3_FC_pt5/Result_colrel_int-2.npy')
    ColRel_pt5_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                           'resnet20/colrel_int-sort_part_cls_3_FC_pt5/Result_colrel_int-3.npy')
    ColRel_pt5_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                           'resnet20/colrel_int-sort_part_cls_3_FC_pt5/Result_colrel_int-4.npy')
    ColRel_pt5 = 1 / 5 * (ColRel_pt5_0[:arr_max_idx] + ColRel_pt5_1[:arr_max_idx] + ColRel_pt5_2[:arr_max_idx]
                          + ColRel_pt5_3[:arr_max_idx] + ColRel_pt5_4[:arr_max_idx])

    # ColRel with intermittent connectivity client-PS and client-client connections (p_{ij} = 0.9)
    ColRel_pt9_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-sort_part_cls_3_FC_pt9/Result_colrel_int-0.npy')
    ColRel_pt9_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-sort_part_cls_3_FC_pt9/Result_colrel_int-1.npy')
    ColRel_pt9_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-sort_part_cls_3_FC_pt9/Result_colrel_int-2.npy')
    ColRel_pt9_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-sort_part_cls_3_FC_pt9/Result_colrel_int-3.npy')
    ColRel_pt9_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-sort_part_cls_3_FC_pt9/Result_colrel_int-4.npy')
    ColRel_pt9 = 1 / 5 * (ColRel_pt9_0[:arr_max_idx] + ColRel_pt9_1[:arr_max_idx] + ColRel_pt9_2[:arr_max_idx]
                          + ColRel_pt9_3[:arr_max_idx] + ColRel_pt9_4[:arr_max_idx])



    # Plot results
    comm_round_array = 5 * np.arange(1, comm_rounds / 5 + 1)
    fig, ax = plt.subplots()
    ax.plot(comm_round_array, FedAvg_NoDropout, label="FedAvg - Perfect Connectivity", color="k", marker='o',
            markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.plot(comm_round_array, FedAvg_Blind, label="FedAvg - Blind", color="r", marker='P',
            markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.plot(comm_round_array, FedAvg_NonBlind, label="FedAvg - Non-Blind", color="b", marker='^',
            markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.plot(comm_round_array, ColRel_pt5, label="ColRel - Erdos-Renyi " + "$\mathbf{(p_c = 0.5)}$", color="m",
            marker='s', markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.plot(comm_round_array, ColRel_pt9, label="ColRel - Erdos-Renyi " + "$\mathbf{(p_c = 0.9)}$", color="g",
            marker='v', markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.set_xlabel("Communication rounds", weight="bold", size=20)
    plt.xticks(weight='bold', fontsize="20")
    ax.set_ylabel("Test accuracy (%)", weight="bold", size=20)
    plt.yticks(weight='bold', fontsize="20")
    ax.set_title("Non-IID local data distribution", weight="bold", size=20)
    plt.grid(which="both", axis="both")
    legend_properties = {'weight': 'bold', 'size': 18}
    plt.legend(loc="lower right", prop=legend_properties)
    plt.show()


def plot_mmWaveCluster_sortpart3_Perf_Int_compare_topology2():

    comm_rounds = 1000
    arr_max_idx = int(comm_rounds / 5)

    # FedAvg with perfect connectivity
    FedAvg_NoDropout_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg-sort_part_cls_3/Result_fed_avg-0.npy')
    FedAvg_NoDropout_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg-sort_part_cls_3/Result_fed_avg-1.npy')
    FedAvg_NoDropout_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg-sort_part_cls_3/Result_fed_avg-2.npy')
    FedAvg_NoDropout_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg-sort_part_cls_3/Result_fed_avg-3.npy')
    FedAvg_NoDropout_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg-sort_part_cls_3/Result_fed_avg-4.npy')
    FedAvg_NoDropout = 1 / 5 * (FedAvg_NoDropout_0[:arr_max_idx] + FedAvg_NoDropout_1[:arr_max_idx] +
                                FedAvg_NoDropout_2[:arr_max_idx] + FedAvg_NoDropout_3[:arr_max_idx] +
                                FedAvg_NoDropout_4[:arr_max_idx])

    # Blind FedAvg with intermittent connectivity
    FedAvg_Blind_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                             'fed_avg_blind-sort_part3_mmWave_cluster/Result_fed_tmp-0.npy')
    FedAvg_Blind_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                             'fed_avg_blind-sort_part3_mmWave_cluster/Result_fed_tmp-1.npy')
    FedAvg_Blind_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                             'fed_avg_blind-sort_part3_mmWave_cluster/Result_fed_tmp-2.npy')
    FedAvg_Blind_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                             'fed_avg_blind-sort_part3_mmWave_cluster/Result_fed_tmp-3.npy')
    FedAvg_Blind_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                             'fed_avg_blind-sort_part3_mmWave_cluster/Result_fed_tmp-4.npy')
    FedAvg_Blind = 1 / 5 * (FedAvg_Blind_0[:arr_max_idx] + FedAvg_Blind_1[:arr_max_idx] +
                            FedAvg_Blind_2[:arr_max_idx] + FedAvg_Blind_3[:arr_max_idx] +
                            FedAvg_Blind_4[:arr_max_idx])

    # Non-Blind FedAvg with intermittent connectivity
    FedAvg_NonBlind_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                'fed_avg_nonblind-sort_part3_mmWave_cluster/Result_fed_tmp-0.npy')
    FedAvg_NonBlind_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                'fed_avg_nonblind-sort_part3_mmWave_cluster/Result_fed_tmp-1.npy')
    FedAvg_NonBlind_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                'fed_avg_nonblind-sort_part3_mmWave_cluster/Result_fed_tmp-2.npy')
    FedAvg_NonBlind_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                'fed_avg_nonblind-sort_part3_mmWave_cluster/Result_fed_tmp-3.npy')
    FedAvg_NonBlind_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                'fed_avg_nonblind-sort_part3_mmWave_cluster/Result_fed_tmp-4.npy')
    FedAvg_NonBlind = 1 / 5 * (FedAvg_NonBlind_0[:arr_max_idx] + FedAvg_NonBlind_1[:arr_max_idx]
                               + FedAvg_NonBlind_2[:arr_max_idx] + FedAvg_NonBlind_3[:arr_max_idx]
                               + FedAvg_NonBlind_4[:arr_max_idx])

    # ColRel with intermittent connectivity client-PS and perfect client-client connections
    ColRel_perf_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                            'colrel-sort_part_cls_3_topology2/Result_colrel-0.npy')
    ColRel_perf_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                            'colrel-sort_part_cls_3_topology2/Result_colrel-1.npy')
    ColRel_perf_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                            'colrel-sort_part_cls_3_topology2/Result_colrel-2.npy')
    ColRel_perf_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                            'colrel-sort_part_cls_3_topology2/Result_colrel-3.npy')
    ColRel_perf_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                            'colrel-sort_part_cls_3_topology2/Result_colrel-4.npy')
    ColRel_perf = 1 / 5 * (ColRel_perf_0[:arr_max_idx] + ColRel_perf_1[:arr_max_idx] + ColRel_perf_2[:arr_max_idx]
                          + ColRel_perf_3[:arr_max_idx] + ColRel_perf_4[:arr_max_idx])

    # ColRel with intermittent connectivity client-PS and client-client connections
    ColRel_int_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-sort_part_cls_3_topology2/Result_colrel_int-0.npy')
    ColRel_int_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-sort_part_cls_3_topology2/Result_colrel_int-1.npy')
    ColRel_int_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-sort_part_cls_3_topology2/Result_colrel_int-2.npy')
    ColRel_int_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-sort_part_cls_3_topology2/Result_colrel_int-3.npy')
    ColRel_int_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-sort_part_cls_3_topology2/Result_colrel_int-4.npy')
    ColRel_int = 1 / 5 * (ColRel_int_0[:arr_max_idx] + ColRel_int_1[:arr_max_idx] + ColRel_int_2[:arr_max_idx]
                          + ColRel_int_3[:arr_max_idx] + ColRel_int_4[:arr_max_idx])

    # Plot results
    comm_round_array = 5 * np.arange(1, comm_rounds / 5 + 1)
    fig, ax = plt.subplots()
    ax.plot(comm_round_array, FedAvg_NoDropout, label="FedAvg - Perfect client-PS", color="k", marker='o',
            markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.plot(comm_round_array, FedAvg_Blind, label="FedAvg - Blind", color="r", marker='P',
            markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.plot(comm_round_array, ColRel_perf, label="ColRel - Perfect client-client", color="m", marker='s',
            markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.plot(comm_round_array, ColRel_int, label="ColRel - Intermittent client-client", color="g", marker='v',
            markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.plot(comm_round_array, FedAvg_NonBlind, label="FedAvg - Non-Blind", color="b", marker='^',
            markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.set_xlabel("Communication rounds", weight="bold", size=20)
    plt.xticks(weight='bold', fontsize="20")
    ax.set_ylabel("Test accuracy (%)", weight="bold", size=20)
    plt.yticks(weight='bold', fontsize="20")
    ax.set_title("Non-IID local data distribution", weight="bold", size=20)
    plt.grid(which="both", axis="both")
    legend_properties = {'weight': 'bold', 'size': 18}
    plt.legend(loc="lower right", prop=legend_properties)
    plt.show()


def plot_mmWaveCluster_sortpart3_Perf_Int_compare_topology1():

    comm_rounds = 1000
    arr_max_idx = int(comm_rounds / 5)

    # FedAvg with perfect connectivity
    FedAvg_NoDropout_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg-sort_part_cls_3/Result_fed_avg-0.npy')
    FedAvg_NoDropout_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg-sort_part_cls_3/Result_fed_avg-1.npy')
    FedAvg_NoDropout_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg-sort_part_cls_3/Result_fed_avg-2.npy')
    FedAvg_NoDropout_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg-sort_part_cls_3/Result_fed_avg-3.npy')
    FedAvg_NoDropout_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg-sort_part_cls_3/Result_fed_avg-4.npy')
    FedAvg_NoDropout = 1 / 5 * (FedAvg_NoDropout_0[:arr_max_idx] + FedAvg_NoDropout_1[:arr_max_idx] +
                                FedAvg_NoDropout_2[:arr_max_idx] + FedAvg_NoDropout_3[:arr_max_idx] +
                                FedAvg_NoDropout_4[:arr_max_idx])

    # Blind FedAvg with intermittent connectivity
    FedAvg_Blind_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                             'fed_avg_blind-sort_part3_mmWave_cluster/Result_fed_tmp-0.npy')
    FedAvg_Blind_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                             'fed_avg_blind-sort_part3_mmWave_cluster/Result_fed_tmp-1.npy')
    FedAvg_Blind_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                             'fed_avg_blind-sort_part3_mmWave_cluster/Result_fed_tmp-2.npy')
    FedAvg_Blind_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                             'fed_avg_blind-sort_part3_mmWave_cluster/Result_fed_tmp-3.npy')
    FedAvg_Blind_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                             'fed_avg_blind-sort_part3_mmWave_cluster/Result_fed_tmp-4.npy')
    FedAvg_Blind = 1 / 5 * (FedAvg_Blind_0[:arr_max_idx] + FedAvg_Blind_1[:arr_max_idx] +
                            FedAvg_Blind_2[:arr_max_idx] + FedAvg_Blind_3[:arr_max_idx] +
                            FedAvg_Blind_4[:arr_max_idx])

    # Non-Blind FedAvg with intermittent connectivity
    FedAvg_NonBlind_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                'fed_avg_nonblind-sort_part3_mmWave_cluster/Result_fed_tmp-0.npy')
    FedAvg_NonBlind_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                'fed_avg_nonblind-sort_part3_mmWave_cluster/Result_fed_tmp-1.npy')
    FedAvg_NonBlind_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                'fed_avg_nonblind-sort_part3_mmWave_cluster/Result_fed_tmp-2.npy')
    FedAvg_NonBlind_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                'fed_avg_nonblind-sort_part3_mmWave_cluster/Result_fed_tmp-3.npy')
    FedAvg_NonBlind_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                'fed_avg_nonblind-sort_part3_mmWave_cluster/Result_fed_tmp-4.npy')
    FedAvg_NonBlind = 1 / 5 * (FedAvg_NonBlind_0[:arr_max_idx] + FedAvg_NonBlind_1[:arr_max_idx]
                               + FedAvg_NonBlind_2[:arr_max_idx] + FedAvg_NonBlind_3[:arr_max_idx]
                               + FedAvg_NonBlind_4[:arr_max_idx])

    # ColRel with intermittent connectivity client-PS and perfect client-client connections
    ColRel_perf_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                            'colrel-sort_part_cls_3_topology1/Result_colrel-0.npy')
    ColRel_perf_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                            'colrel-sort_part_cls_3_topology1/Result_colrel-1.npy')
    ColRel_perf_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                            'colrel-sort_part_cls_3_topology1/Result_colrel-2.npy')
    ColRel_perf_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                            'colrel-sort_part_cls_3_topology1/Result_colrel-3.npy')
    ColRel_perf_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                            'colrel-sort_part_cls_3_topology1/Result_colrel-4.npy')
    ColRel_perf = 1 / 5 * (ColRel_perf_0[:arr_max_idx] + ColRel_perf_1[:arr_max_idx] + ColRel_perf_2[:arr_max_idx]
                           + ColRel_perf_3[:arr_max_idx] + ColRel_perf_4[:arr_max_idx])

    # ColRel with intermittent connectivity client-PS and client-client connections
    ColRel_int_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-sort_part_cls_3_topology1/Result_colrel_int-0.npy')
    ColRel_int_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-sort_part_cls_3_topology1/Result_colrel_int-1.npy')
    ColRel_int_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-sort_part_cls_3_topology1/Result_colrel_int-2.npy')
    ColRel_int_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-sort_part_cls_3_topology1/Result_colrel_int-3.npy')
    ColRel_int_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-sort_part_cls_3_topology1/Result_colrel_int-4.npy')
    ColRel_int = 1 / 5 * (ColRel_int_0[:arr_max_idx] + ColRel_int_1[:arr_max_idx] + ColRel_int_2[:arr_max_idx]
                          + ColRel_int_3[:arr_max_idx] + ColRel_int_4[:arr_max_idx])

    # ColRel with intermittent connectivity client-PS and client-client connections (p_{ij} = 0.5)
    ColRel_pt5_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                           'resnet20/colrel_int-sort_part_cls_3_FC_pt5/Result_colrel_int-0.npy')
    ColRel_pt5_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                           'resnet20/colrel_int-sort_part_cls_3_FC_pt5/Result_colrel_int-1.npy')
    ColRel_pt5_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                           'resnet20/colrel_int-sort_part_cls_3_FC_pt5/Result_colrel_int-2.npy')
    ColRel_pt5_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                           'resnet20/colrel_int-sort_part_cls_3_FC_pt5/Result_colrel_int-3.npy')
    ColRel_pt5_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/'
                           'resnet20/colrel_int-sort_part_cls_3_FC_pt5/Result_colrel_int-4.npy')
    ColRel_pt5 = 1 / 5 * (ColRel_pt5_0[:arr_max_idx] + ColRel_pt5_1[:arr_max_idx] + ColRel_pt5_2[:arr_max_idx]
                          + ColRel_pt5_3[:arr_max_idx] + ColRel_pt5_4[:arr_max_idx])

    # Plot results
    comm_round_array = 5 * np.arange(1, comm_rounds / 5 + 1)
    fig, ax = plt.subplots()
    ax.plot(comm_round_array, FedAvg_NoDropout, label="FedAvg - Perfect client-PS", color="k", marker='o',
            markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.plot(comm_round_array, FedAvg_Blind, label="FedAvg - Blind", color="r", marker='P',
            markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.plot(comm_round_array, ColRel_perf, label="ColRel - Perfect client-client", color="m", marker='s',
            markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.plot(comm_round_array, ColRel_int, label="ColRel - Intermittent client-client", color="g", marker='v',
            markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.plot(comm_round_array, FedAvg_NonBlind, label="FedAvg - Non-Blind", color="b", marker='^',
            markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.set_xlabel("Communication rounds", weight="bold", size=20)
    plt.xticks(weight='bold', fontsize="20")
    ax.set_ylabel("Test accuracy (%)", weight="bold", size=20)
    plt.yticks(weight='bold', fontsize="20")
    ax.set_title("Non-IID local data distribution", weight="bold", size=20)
    plt.grid(which="both", axis="both")
    legend_properties = {'weight': 'bold', 'size': 18}
    plt.legend(loc="lower right", prop=legend_properties)
    plt.show()


def plot_fully_connected_iid_one_good_client():
    """
    Simulates the setting when all but one (good) clients have a connection probability of 0.1 to the PS
    and the gc has a connectivity of either 0.9 or 0.5
    """

    comm_rounds = 1000
    arr_max_idx = int(comm_rounds / 5)

    # FedAvg with perfect connectivity
    FedAvg_NoDropout_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg-iid/Result_fed_avg-0.npy')
    FedAvg_NoDropout_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg-iid/Result_fed_avg-1.npy')
    FedAvg_NoDropout_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg-iid/Result_fed_avg-2.npy')
    FedAvg_NoDropout_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg-iid/Result_fed_avg-3.npy')
    FedAvg_NoDropout_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg-iid/Result_fed_avg-4.npy')
    FedAvg_NoDropout = 1 / 5 * (FedAvg_NoDropout_0[:arr_max_idx] + FedAvg_NoDropout_1[:arr_max_idx] +
                                FedAvg_NoDropout_2[:arr_max_idx] + FedAvg_NoDropout_3[:arr_max_idx] +
                                FedAvg_NoDropout_4[:arr_max_idx])

    # Blind FedAvg with intermittent client-PS connectivity
    FedAvg_Blind_pt9_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg_blind-iid_gc_pt9/Result_fed_avg_blind-0.npy')
    FedAvg_Blind_pt9_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg_blind-iid_gc_pt9/Result_fed_avg_blind-1.npy')
    FedAvg_Blind_pt9_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg_blind-iid_gc_pt9/Result_fed_avg_blind-2.npy')
    FedAvg_Blind_pt9_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg_blind-iid_gc_pt9/Result_fed_avg_blind-3.npy')
    FedAvg_Blind_pt9_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                 'fed_avg_blind-iid_gc_pt9/Result_fed_avg_blind-4.npy')
    FedAvg_Blind_pt9 = 1 / 5 * (FedAvg_Blind_pt9_0[:arr_max_idx] + FedAvg_Blind_pt9_1[:arr_max_idx] +
                                FedAvg_Blind_pt9_2[:arr_max_idx] + FedAvg_Blind_pt9_3[:arr_max_idx] +
                                FedAvg_Blind_pt9_4[:arr_max_idx])

    # Non-Blind FedAvg with intermittent connectivity
    FedAvg_NonBlind_pt9_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                    'fed_avg_nonblind-iid_gc_pt9/Result_fed_avg_nonblind-0.npy')
    FedAvg_NonBlind_pt9_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                    'fed_avg_nonblind-iid_gc_pt9/Result_fed_avg_nonblind-1.npy')
    FedAvg_NonBlind_pt9_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                    'fed_avg_nonblind-iid_gc_pt9/Result_fed_avg_nonblind-2.npy')
    FedAvg_NonBlind_pt9_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                    'fed_avg_nonblind-iid_gc_pt9/Result_fed_avg_nonblind-3.npy')
    FedAvg_NonBlind_pt9_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                                    'fed_avg_nonblind-iid_gc_pt9/Result_fed_avg_nonblind-4.npy')
    FedAvg_NonBlind_pt9 = 1 / 5 * (FedAvg_NonBlind_pt9_0[:arr_max_idx] + FedAvg_NonBlind_pt9_1[:arr_max_idx]
                                   + FedAvg_NonBlind_pt9_2[:arr_max_idx] + FedAvg_NonBlind_pt9_3[:arr_max_idx]
                                   + FedAvg_NonBlind_pt9_4[:arr_max_idx])

    # ColRel with intermittent connectivity client-PS and client-client connections (p_{ij} = 0.9)
    ColRel_pt9_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-iid_gc_pt9_FC_pt9/Result_colrel_int-0.npy')
    ColRel_pt9_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-iid_gc_pt9_FC_pt9/Result_colrel_int-1.npy')
    ColRel_pt9_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-iid_gc_pt9_FC_pt9/Result_colrel_int-2.npy')
    ColRel_pt9_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-iid_gc_pt9_FC_pt9/Result_colrel_int-3.npy')
    ColRel_pt9_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-iid_gc_pt9_FC_pt9/Result_colrel_int-4.npy')
    ColRel_pt9 = 1 / 5 * (ColRel_pt9_0[:arr_max_idx] + ColRel_pt9_1[:arr_max_idx] + ColRel_pt9_2[:arr_max_idx]
                          + ColRel_pt9_3[:arr_max_idx] + ColRel_pt9_4[:arr_max_idx])

    # ColRel with intermittent connectivity client-PS and client-client connections (p_{ij} = 0.5)
    ColRel_pt5_0 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-iid_gc_pt9_FC_pt5/Result_colrel_int-0.npy')
    ColRel_pt5_1 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-iid_gc_pt9_FC_pt5/Result_colrel_int-1.npy')
    ColRel_pt5_2 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-iid_gc_pt9_FC_pt5/Result_colrel_int-2.npy')
    ColRel_pt5_3 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-iid_gc_pt9_FC_pt5/Result_colrel_int-3.npy')
    ColRel_pt5_4 = np.load('/Users/rajarshi/Documents/CollaborativeRelaying/results_intermittent/resnet20/'
                           'colrel_int-iid_gc_pt9_FC_pt5/Result_colrel_int-4.npy')
    ColRel_pt5 = 1 / 5 * (ColRel_pt5_0[:arr_max_idx] + ColRel_pt5_1[:arr_max_idx] + ColRel_pt5_2[:arr_max_idx]
                          + ColRel_pt5_3[:arr_max_idx] + ColRel_pt5_4[:arr_max_idx])

    # Plot results
    comm_round_array = 5 * np.arange(1, comm_rounds / 5 + 1)
    fig, ax = plt.subplots()
    ax.plot(comm_round_array, FedAvg_NoDropout, label="FedAvg - Perfect Connectivity", color="k", marker='o',
            markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.plot(comm_round_array, FedAvg_Blind_pt9, label="FedAvg - Blind",
            color="r", marker='P', markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.plot(comm_round_array, FedAvg_NonBlind_pt9, label="FedAvg - Non-Blind",
            color="b", marker='^', markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.plot(comm_round_array, ColRel_pt9, label="ColRel - Erdos-Renyi " + "$\mathbf{(p_c = 0.9)}$",
            color="g", marker='v', markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.plot(comm_round_array, ColRel_pt5, label="ColRel - Erdos-Renyi " + "$\mathbf{(p_c = 0.5)}$",
            color="m", marker='s', markevery=12, markersize=8, linestyle="dashed", linewidth=3.0)
    ax.set_xlabel("Communication rounds", weight="bold", size=20)
    plt.xticks(weight='bold', fontsize="20")
    ax.set_ylabel("Test accuracy (%)", weight="bold", size=20)
    plt.yticks(weight='bold', fontsize="20")
    ax.set_title("IID local data distribution " + "$\mathbf{(p_{gc} = 0.9)}$", weight="bold", size=20)
    plt.grid(which="both", axis="both")
    legend_properties = {'weight': 'bold', 'size': 18}
    plt.legend(loc="lower right", prop=legend_properties)
    plt.show()


if __name__ == "__main__":

    plot_fully_connected_sortpart3()
    # plot_mmWaveCluster_sortpart3_Perf_Int_compare_topology2()
    # plot_mmWaveCluster_sortpart3_Perf_Int_compare_topology1()
    # plot_fully_connected_iid_one_good_client()

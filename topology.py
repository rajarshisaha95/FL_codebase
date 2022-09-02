import numpy as np
np.set_printoptions(precision=2)


def fully_connected_topology(num_clients: int = 10):
    return np.ones((num_clients, num_clients))


def client_location_mmWave_random_circle_ring(num_clients: int = 10):
    """
    :param num_clients: Number of federated edge learning clients
    :return: probability of successful transmission to PS, and inter-client connectivity matrix
    """

    PS_loc = np.array([0, 0])

    circle_max_rad = 205                                        # Maximum distance of a client from PS (in meters)
    circle_min_rad = 105                                        # Minimum distance of a client from PS (in meters)
    circle_diff_rad = circle_max_rad - circle_min_rad

    # Distances of clients uniformly far from PS
    rad_client_uniform = circle_min_rad + (circle_diff_rad * np.sqrt(np.random.rand(num_clients)))

    # Direction of clients uniformly distributed around PS
    client_vec_deg = 2 * np.pi * np.random.rand(num_clients)

    # Determine the locations of the clients
    loc_clients = []
    for idx in range(num_clients):
        x = rad_client_uniform[idx] * np.cos(client_vec_deg[idx])
        y = rad_client_uniform[idx] * np.sin(client_vec_deg[idx])
        loc_clients.append([x, y])
    loc_clients = np.array(loc_clients)

    # Compute distances of clients from PS
    sub_clients_PS = loc_clients - PS_loc
    dist_clients_PS2 = np.zeros(len(sub_clients_PS))
    for idx in range(len(sub_clients_PS)):
        dist_clients_PS2[idx] = sub_clients_PS[idx][0] ** 2 + sub_clients_PS[idx][1] ** 2
    dist_clients_PS = np.sqrt(dist_clients_PS2)

    # Compute pairwise distances between clients
    dist_clients_clients = np.zeros([num_clients, num_clients])
    for i in range(len(loc_clients)):
        for j in range(len(loc_clients)):
            dist_vector = loc_clients[i] - loc_clients[j]
            dist_clients_clients[i][j] = np.linalg.norm(dist_vector, 2)

    # Compute probability of successful transmission to PS
    prob_success_PS = np.zeros(num_clients)
    for i in range(len(dist_clients_PS)):
        p = min(1, np.exp(-dist_clients_PS[i]/30 + 5.2))
        prob_success_PS[i] = np.round(p * 100) / 100

    # Determine connectivity amongst clients
    connectivity_mat_clients = np.zeros([num_clients, num_clients])
    for i in range(num_clients):
        for j in range(num_clients):
            p = min(1, np.exp(-dist_clients_clients[i][j]/30 + 5.2))
            if p > 0.95:
                connectivity_mat_clients[i][j] = 1

    with open('prob_success_PS.npy', 'wb') as f:
        np.save(f, prob_success_PS)
    with open('connectivity_mat_clients.npy', 'wb') as f:
        np.save(f, connectivity_mat_clients)

    return prob_success_PS, connectivity_mat_clients


def client_locations_mmWave_perfect_circle_ring(num_clients: int = 10):
    """
    :param num_clients: Number of federated edge learning clients
    :return: probability of successful transmission to PS, and inter-client connectivity matrix
    """

    PS_loc = np.array([0, 0])

    # Distances of clients from PS is the same for all clients
    circle_equiprob_rad = 205       # meters. For prob success ~ 0.23

    # Clients are arranged in a perfect ring
    client_vec_deg = np.array([2 * np.pi/num_clients * i for i in range(num_clients)])

    # Determine the locations of the clients
    loc_clients = []
    for idx in range(num_clients):
        x = circle_equiprob_rad * np.cos(client_vec_deg[idx])
        y = circle_equiprob_rad * np.sin(client_vec_deg[idx])
        loc_clients.append([x, y])
    loc_clients = np.array(loc_clients)

    # Compute distances of clients from PS
    sub_clients_PS = loc_clients - PS_loc
    dist_clients_PS2 = np.zeros(len(sub_clients_PS))
    for idx in range(len(sub_clients_PS)):
        dist_clients_PS2[idx] = sub_clients_PS[idx][0] ** 2 + sub_clients_PS[idx][1] ** 2
    dist_clients_PS = np.sqrt(dist_clients_PS2)

    # Compute pairwise distances between clients
    dist_clients_clients = np.zeros([num_clients, num_clients])
    for i in range(len(loc_clients)):
        for j in range(len(loc_clients)):
            dist_vector = loc_clients[i] - loc_clients[j]
            dist_clients_clients[i][j] = np.linalg.norm(dist_vector, 2)

    # Compute probability of successful transmission to PS
    prob_success_PS = np.zeros(num_clients)
    for i in range(len(dist_clients_PS)):
        p = min(1, np.exp(-dist_clients_PS[i] / 30 + 5.2))
        prob_success_PS[i] = np.round(p * 100) / 100

    # Determine connectivity amongst clients
    connectivity_mat_clients = np.zeros([num_clients, num_clients])
    for i in range(num_clients):
        for j in range(num_clients):
            p = min(1, np.exp(-dist_clients_clients[i][j] / 30 + 5.2))
            if p > 0.95:
                connectivity_mat_clients[i][j] = 1

    return prob_success_PS, connectivity_mat_clients


def client_locations_mmWave_clusters(num_clients: int = 10):
    """
    :param num_clients: Number of federated edge learning clients
    :return: probability of successful transmission to PS, and inter-client connectivity matrix
    """

    # PS is placed at origin
    PS_loc = np.array([0, 0])

    # Distance of good clients from PS
    circle_good_rad = 159                       # meters. For prob success ~ 0.9

    # Clients angles
    client_vec_deg = np.zeros(num_clients)
    client_vec_deg[3] = 2 * np.pi / 3
    client_vec_deg[6] = 4 * np.pi / 3

    x = np.zeros(num_clients)
    y = np.zeros(num_clients)

    # Determining the Cartesian coordinates of the clients with good connectivity
    x[0] = circle_good_rad * np.cos(client_vec_deg[0])
    y[0] = circle_good_rad * np.sin(client_vec_deg[0])
    x[3] = circle_good_rad * np.cos(client_vec_deg[3])
    y[3] = circle_good_rad * np.sin(client_vec_deg[3])
    x[6] = circle_good_rad * np.cos(client_vec_deg[6])
    y[6] = circle_good_rad * np.sin(client_vec_deg[6])

    d = 70                       # Distance of bad clients in each cluster to the good client of the cluster

    # Cluster 1
    ang1 = np.pi / 6
    x[1] = x[0] + d * np.cos(ang1)
    y[1] = y[0] + d * np.sin(ang1)

    ang2 = -np.pi / 6
    x[2] = x[0] + d * np.cos(ang2)
    y[2] = y[0] + d * np.sin(ang2)

    # Cluster 2
    ang4 = 2 * np.pi / 3 + np.pi / 6
    x[4] = x[3] + d * np.cos(ang4)
    y[4] = y[3] + d * np.sin(ang4)

    ang5 = 2 * np.pi / 3 - np.pi / 6
    x[5] = x[3] + d * np.cos(ang5)
    y[5] = y[3] + d * np.sin(ang5)

    # Cluster 3
    ang7 = 4 * np.pi / 3 - np.pi / 6
    x[7] = x[6] + d * np.cos(ang7)
    y[7] = y[6] + d * np.sin(ang7)

    ang8 = 4 * np.pi / 3
    x[8] = x[6] + d * np.cos(ang8)
    y[8] = y[6] + d * np.sin(ang8)

    ang9 = 4 * np.pi / 3 + np.pi / 6
    x[9] = x[6] + d * np.cos(ang9)
    y[9] = y[6] + d * np.sin(ang9)

    # Client locations
    loc_clients = []
    for idx in range(num_clients):
        loc_clients.append([x[idx], y[idx]])
    loc_clients = np.array(loc_clients)

    # Compute distances of clients from PS
    sub_clients_PS = loc_clients - PS_loc
    dist_clients_PS2 = np.zeros(len(sub_clients_PS))
    for idx in range(len(sub_clients_PS)):
        dist_clients_PS2[idx] = sub_clients_PS[idx][0] ** 2 + sub_clients_PS[idx][1] ** 2
    dist_clients_PS = np.sqrt(dist_clients_PS2)

    # Compute pairwise distances between clients
    dist_clients_clients = np.zeros([num_clients, num_clients])
    for i in range(len(loc_clients)):
        for j in range(len(loc_clients)):
            dist_vector = loc_clients[i] - loc_clients[j]
            dist_clients_clients[i][j] = np.linalg.norm(dist_vector, 2)

    # Compute probability of successful transmission to PS
    prob_success_PS = np.zeros(num_clients)
    for i in range(len(dist_clients_PS)):
        p = min(1, np.exp(-dist_clients_PS[i] / 30 + 5.2))
        prob_success_PS[i] = np.round(p * 100) / 100

    # Determine connectivity amongst clients
    connectivity_mat_clients = np.zeros([num_clients, num_clients])
    for i in range(num_clients):
        for j in range(num_clients):
            p = min(1, np.exp(-dist_clients_clients[i][j] / 30 + 5.2))
            if p > 0.99:
                connectivity_mat_clients[i][j] = 1

    with open('prob_success_PS.npy', 'wb') as f:
        np.save(f, prob_success_PS)
    with open('connectivity_mat_clients.npy', 'wb') as f:
        np.save(f, connectivity_mat_clients)

    return prob_success_PS, connectivity_mat_clients


def mmWave_success_prob(dist: float = 1):
    """
    Returns the success probability of transmission between two clients
    :param dist: Distance between transmitter and receiver
    """
    return min(1, np.exp(-dist / 30 + 5.2))


def dist_from_origin(x: float = 1, y: float = 1):
    """
    Returns distance from origin
    :param x -- X-coordinate
    :param y -- Y coordinate
    """

    return np.sqrt(x ** 2 + y ** 2)


def clustered_topology_intermittent_vs_perfect_compare(num_clients: int = 10, cluster1_ang: float = np.pi / 4,
                                                       cluster2_ang: float = np.pi / 2):
    """
    :param num_clients: Number of federated edge learning clients
    :param cluster1_ang: Angular orientation of first cluster
    :param cluster2_ang: Angular orientation of second cluster
    :return: probability of successful transmission to PS, and inter-client connectivity matrix
    """

    # PS is placed at origin
    PS_loc = np.array([0, 0])

    # Distance of good clients from PS
    circle_good_rad = 159                       # meters. For prob success ~ 0.9

    # Clients angles
    cluster1_angle = 0
    cluster2_angle = cluster1_ang
    cluster3_angle = cluster2_ang

    x = np.zeros(num_clients)
    y = np.zeros(num_clients)

    # Determining the Cartesian coordinates of the clients with good connectivity
    x[0] = circle_good_rad * np.cos(cluster1_angle)
    y[0] = circle_good_rad * np.sin(cluster1_angle)
    x[3] = circle_good_rad * np.cos(cluster2_angle)
    y[3] = circle_good_rad * np.sin(cluster2_angle)
    x[6] = circle_good_rad * np.cos(cluster3_angle)
    y[6] = circle_good_rad * np.sin(cluster3_angle)

    d = 70                       # Distance of bad clients in each cluster to the good client of the cluster

    # Cluster 1
    ang1 = np.pi / 6
    x[1] = x[0] + d * np.cos(ang1)
    y[1] = y[0] + d * np.sin(ang1)

    ang2 = -np.pi / 6
    x[2] = x[0] + d * np.cos(ang2)
    y[2] = y[0] + d * np.sin(ang2)

    # Cluster 2
    ang4 = 2 * np.pi / 3 + np.pi / 6
    x[4] = x[3] + d * np.cos(ang4)
    y[4] = y[3] + d * np.sin(ang4)

    ang5 = 2 * np.pi / 3 - np.pi / 6
    x[5] = x[3] + d * np.cos(ang5)
    y[5] = y[3] + d * np.sin(ang5)

    # Cluster 3
    ang7 = 4 * np.pi / 3 - np.pi / 6
    x[7] = x[6] + d * np.cos(ang7)
    y[7] = y[6] + d * np.sin(ang7)

    ang8 = 4 * np.pi / 3
    x[8] = x[6] + d * np.cos(ang8)
    y[8] = y[6] + d * np.sin(ang8)

    ang9 = 4 * np.pi / 3 + np.pi / 6
    x[9] = x[6] + d * np.cos(ang9)
    y[9] = y[6] + d * np.sin(ang9)

    # Client locations
    loc_clients = []
    for idx in range(num_clients):
        loc_clients.append([x[idx], y[idx]])
    loc_clients = np.array(loc_clients)

    # Compute distances of clients from PS
    sub_clients_PS = loc_clients - PS_loc
    dist_clients_PS2 = np.zeros(len(sub_clients_PS))
    for idx in range(len(sub_clients_PS)):
        dist_clients_PS2[idx] = sub_clients_PS[idx][0] ** 2 + sub_clients_PS[idx][1] ** 2
    dist_clients_PS = np.sqrt(dist_clients_PS2)

    # Compute pairwise distances between clients
    dist_clients_clients = np.zeros([num_clients, num_clients])
    for i in range(len(loc_clients)):
        for j in range(len(loc_clients)):
            dist_vector = loc_clients[i] - loc_clients[j]
            dist_clients_clients[i][j] = np.linalg.norm(dist_vector, 2)

    # Compute probability of successful transmission to PS
    prob_success_PS = np.zeros(num_clients)
    for i in range(len(dist_clients_PS)):
        p = min(1, np.exp(-dist_clients_PS[i] / 30 + 5.2))
        prob_success_PS[i] = np.round(p * 100) / 100

    # Determine connectivity amongst clients
    connectivity_mat_clients = np.zeros([num_clients, num_clients])
    P = np.zeros([num_clients, num_clients])                            # Connection probabilities to neighbors
    for i in range(num_clients):
        for j in range(num_clients):
            P[i][j] = min(1, np.exp(-dist_clients_clients[i][j] / 30 + 5.2))
            if P[i][j] > 0.99:
                connectivity_mat_clients[i][j] = 1

    with open('prob_success_PS.npy', 'wb') as f:
        np.save(f, prob_success_PS)
    with open('connectivity_mat_clients.npy', 'wb') as f:
        np.save(f, connectivity_mat_clients)
    with open('prob_ngbrs.npy', 'wb') as f:
        np.save(f, P)

    return prob_success_PS, P, connectivity_mat_clients


def client_locations_mmWave_clusters_intermittent(num_clients: int = 10):
    """
    :param num_clients: Number of federated edge learning clients
    :return: probability of successful transmission to PS, and inter-client connectivity matrix
    """

    # PS is placed at origin
    PS_loc = np.array([0, 0])

    # Distance of good clients from PS
    circle_good_rad = 159                       # meters. For prob success ~ 0.9

    # Clients angles
    client_vec_deg = np.zeros(num_clients)
    client_vec_deg[3] = 2 * np.pi / 3
    client_vec_deg[6] = 4 * np.pi / 3

    x = np.zeros(num_clients)
    y = np.zeros(num_clients)

    # Determining the Cartesian coordinates of the clients with good connectivity
    x[0] = circle_good_rad * np.cos(client_vec_deg[0])
    y[0] = circle_good_rad * np.sin(client_vec_deg[0])
    x[3] = circle_good_rad * np.cos(client_vec_deg[3])
    y[3] = circle_good_rad * np.sin(client_vec_deg[3])
    x[6] = circle_good_rad * np.cos(client_vec_deg[6])
    y[6] = circle_good_rad * np.sin(client_vec_deg[6])

    d = 159                       # Distance of bad clients in each cluster to the good client of the cluster

    # Cluster 1
    ang1 = 1.582
    x[1] = x[0] + d * np.cos(ang1)
    y[1] = y[0] + d * np.sin(ang1)
    print("Distance = {}".format(np.sqrt(x[1] ** 2 + y[1] ** 2)))

    ang2 = - 1.582
    x[2] = x[0] + d * np.cos(ang2)
    y[2] = y[0] + d * np.sin(ang2)

    # Cluster 2
    ang4 = 2 * np.pi / 3 + 1.582
    x[4] = x[3] + d * np.cos(ang4)
    y[4] = y[3] + d * np.sin(ang4)

    ang5 = 2 * np.pi / 3 - 1.582
    x[5] = x[3] + d * np.cos(ang5)
    y[5] = y[3] + d * np.sin(ang5)

    # Cluster 3
    ang7 = 4 * np.pi / 3 - 1.582
    x[7] = x[6] + d * np.cos(ang7)
    y[7] = y[6] + d * np.sin(ang7)

    ang8 = 4 * np.pi / 3 - 1.54
    x[8] = x[6] + d * np.cos(ang8)
    y[8] = y[6] + d * np.sin(ang8)

    ang9 = 4 * np.pi / 3 + 1.582
    x[9] = x[6] + d * np.cos(ang9)
    y[9] = y[6] + d * np.sin(ang9)

    # Client locations
    loc_clients = []
    for idx in range(num_clients):
        loc_clients.append([x[idx], y[idx]])
    loc_clients = np.array(loc_clients)

    # Compute distances of clients from PS
    sub_clients_PS = loc_clients - PS_loc
    dist_clients_PS2 = np.zeros(len(sub_clients_PS))
    for idx in range(len(sub_clients_PS)):
        dist_clients_PS2[idx] = sub_clients_PS[idx][0] ** 2 + sub_clients_PS[idx][1] ** 2
    dist_clients_PS = np.sqrt(dist_clients_PS2)

    # Compute pairwise distances between clients
    dist_clients_clients = np.zeros([num_clients, num_clients])
    for i in range(len(loc_clients)):
        for j in range(len(loc_clients)):
            dist_vector = loc_clients[i] - loc_clients[j]
            dist_clients_clients[i][j] = np.linalg.norm(dist_vector, 2)

    # Compute probability of successful transmission to PS
    prob_success_PS = np.zeros(num_clients)
    for i in range(len(dist_clients_PS)):
        p = min(1, np.exp(-dist_clients_PS[i] / 30 + 5.2))
        prob_success_PS[i] = np.round(p * 100) / 100

    # Determine connectivity amongst clients
    P = np.zeros([num_clients, num_clients])
    connectivity_mat_clients = np.zeros([num_clients, num_clients])
    for i in range(num_clients):
        for j in range(num_clients):
            p = min(1, np.exp(-dist_clients_clients[i][j] / 30 + 5.2))
            if p > 0.5:
                P[i][j] = p
                connectivity_mat_clients[i][j] = 1

    with open('prob_success_PS.npy', 'wb') as f:
        np.save(f, prob_success_PS)
    with open('connectivity_mat_clients.npy', 'wb') as f:
        np.save(f, connectivity_mat_clients)

    return prob_success_PS, P, connectivity_mat_clients


def client_locations_mmWave_clusters_perfect_conn(num_clients: int = 10):
    """
    :param num_clients: Number of federated edge learning clients
    :return: probability of successful transmission to PS, and inter-client connectivity matrix
    """

    # PS is placed at origin
    PS_loc = np.array([0, 0])

    # Distance of good clients from PS
    circle_good_rad = 159                       # meters. For prob success ~ 0.9

    # Clients angles
    client_vec_deg = np.zeros(num_clients)
    client_vec_deg[3] = 2 * np.pi / 3
    client_vec_deg[6] = 4 * np.pi / 3

    x = np.zeros(num_clients)
    y = np.zeros(num_clients)

    # Determining the Cartesian coordinates of the clients with good connectivity
    x[0] = circle_good_rad * np.cos(client_vec_deg[0])
    y[0] = circle_good_rad * np.sin(client_vec_deg[0])
    x[3] = circle_good_rad * np.cos(client_vec_deg[3])
    y[3] = circle_good_rad * np.sin(client_vec_deg[3])
    x[6] = circle_good_rad * np.cos(client_vec_deg[6])
    y[6] = circle_good_rad * np.sin(client_vec_deg[6])

    d = 159                       # Distance of bad clients in each cluster to the good client of the cluster

    # Cluster 1
    ang1 = 1.582
    x[1] = x[0] + d * np.cos(ang1)
    y[1] = y[0] + d * np.sin(ang1)
    print("Distance = {}".format(np.sqrt(x[1] ** 2 + y[1] ** 2)))

    ang2 = - 1.582
    x[2] = x[0] + d * np.cos(ang2)
    y[2] = y[0] + d * np.sin(ang2)

    # Cluster 2
    ang4 = 2 * np.pi / 3 + 1.582
    x[4] = x[3] + d * np.cos(ang4)
    y[4] = y[3] + d * np.sin(ang4)

    ang5 = 2 * np.pi / 3 - 1.582
    x[5] = x[3] + d * np.cos(ang5)
    y[5] = y[3] + d * np.sin(ang5)

    # Cluster 3
    ang7 = 4 * np.pi / 3 - 1.582
    x[7] = x[6] + d * np.cos(ang7)
    y[7] = y[6] + d * np.sin(ang7)

    ang8 = 4 * np.pi / 3 - 1.54
    x[8] = x[6] + d * np.cos(ang8)
    y[8] = y[6] + d * np.sin(ang8)

    ang9 = 4 * np.pi / 3 + 1.582
    x[9] = x[6] + d * np.cos(ang9)
    y[9] = y[6] + d * np.sin(ang9)

    # Client locations
    loc_clients = []
    for idx in range(num_clients):
        loc_clients.append([x[idx], y[idx]])
    loc_clients = np.array(loc_clients)

    # Compute distances of clients from PS
    sub_clients_PS = loc_clients - PS_loc
    dist_clients_PS2 = np.zeros(len(sub_clients_PS))
    for idx in range(len(sub_clients_PS)):
        dist_clients_PS2[idx] = sub_clients_PS[idx][0] ** 2 + sub_clients_PS[idx][1] ** 2
    dist_clients_PS = np.sqrt(dist_clients_PS2)

    # Compute pairwise distances between clients
    dist_clients_clients = np.zeros([num_clients, num_clients])
    for i in range(len(loc_clients)):
        for j in range(len(loc_clients)):
            dist_vector = loc_clients[i] - loc_clients[j]
            dist_clients_clients[i][j] = np.linalg.norm(dist_vector, 2)

    # Compute probability of successful transmission to PS
    prob_success_PS = np.zeros(num_clients)
    for i in range(len(dist_clients_PS)):
        p = min(1, np.exp(-dist_clients_PS[i] / 30 + 5.2))
        prob_success_PS[i] = np.round(p * 100) / 100

    # Determine connectivity amongst clients
    connectivity_mat_clients = np.zeros([num_clients, num_clients])
    for i in range(num_clients):
        for j in range(num_clients):
            p = min(1, np.exp(-dist_clients_clients[i][j] / 30 + 5.2))
            if p > 0.99:
                connectivity_mat_clients[i][j] = 1

    with open('prob_success_PS.npy', 'wb') as f:
        np.save(f, prob_success_PS)
    with open('connectivity_mat_clients.npy', 'wb') as f:
        np.save(f, connectivity_mat_clients)

    return prob_success_PS, connectivity_mat_clients


def client_locations_mmWave_clusters_perfect_conn2(num_clients: int = 10):
    """
    :param num_clients: Number of federated edge learning clients
    :return: probability of successful transmission to PS, and inter-client connectivity matrix
    """

    # PS is placed at origin
    PS_loc = np.array([0, 0])

    # Distance of good clients from PS
    circle_good_rad = 159                       # meters. For prob success ~ 0.9

    # Clients angles
    client_vec_deg = np.zeros(num_clients)
    client_vec_deg[3] = 2 * np.pi / 3
    client_vec_deg[6] = 4 * np.pi / 3

    x = np.zeros(num_clients)
    y = np.zeros(num_clients)

    # Determining the Cartesian coordinates of the clients with good connectivity
    x[0] = circle_good_rad * np.cos(client_vec_deg[0])
    y[0] = circle_good_rad * np.sin(client_vec_deg[0])
    x[3] = circle_good_rad * np.cos(client_vec_deg[3])
    y[3] = circle_good_rad * np.sin(client_vec_deg[3])
    x[6] = circle_good_rad * np.cos(client_vec_deg[6])
    y[6] = circle_good_rad * np.sin(client_vec_deg[6])

    d = 159                       # Distance of bad clients in each cluster to the good client of the cluster
    d1 = 156

    # Cluster 1
    ang1 = 1.582
    x[1] = x[0] + d1 * np.cos(ang1)
    y[1] = y[0] + d1 * np.sin(ang1)
    print("Distance = {}".format(np.sqrt(x[1] ** 2 + y[1] ** 2)))

    ang2 = - 1.582
    x[2] = x[0] + d1 * np.cos(ang2)
    y[2] = y[0] + d1 * np.sin(ang2)

    # Cluster 2
    ang4 = 2 * np.pi / 3 + 1.582
    x[4] = x[3] + d1 * np.cos(ang4)
    y[4] = y[3] + d1 * np.sin(ang4)

    ang5 = 2 * np.pi / 3 - 1.582
    x[5] = x[3] + d1 * np.cos(ang5)
    y[5] = y[3] + d1 * np.sin(ang5)

    # Cluster 3
    ang7 = 4 * np.pi / 3 - 1.582
    x[7] = x[6] + d1 * np.cos(ang7)
    y[7] = y[6] + d1 * np.sin(ang7)

    ang8 = 4 * np.pi / 3 - 1.54
    x[8] = x[6] + d * np.cos(ang8)
    y[8] = y[6] + d * np.sin(ang8)

    ang9 = 4 * np.pi / 3 + 1.582
    x[9] = x[6] + d1 * np.cos(ang9)
    y[9] = y[6] + d1 * np.sin(ang9)

    # Client locations
    loc_clients = []
    for idx in range(num_clients):
        loc_clients.append([x[idx], y[idx]])
    loc_clients = np.array(loc_clients)

    # Compute distances of clients from PS
    sub_clients_PS = loc_clients - PS_loc
    dist_clients_PS2 = np.zeros(len(sub_clients_PS))
    for idx in range(len(sub_clients_PS)):
        dist_clients_PS2[idx] = sub_clients_PS[idx][0] ** 2 + sub_clients_PS[idx][1] ** 2
    dist_clients_PS = np.sqrt(dist_clients_PS2)

    # Compute pairwise distances between clients
    dist_clients_clients = np.zeros([num_clients, num_clients])
    for i in range(len(loc_clients)):
        for j in range(len(loc_clients)):
            dist_vector = loc_clients[i] - loc_clients[j]
            dist_clients_clients[i][j] = np.linalg.norm(dist_vector, 2)

    # Compute probability of successful transmission to PS
    prob_success_PS = np.zeros(num_clients)
    for i in range(len(dist_clients_PS)):
        p = min(1, np.exp(-dist_clients_PS[i] / 30 + 5.2))
        prob_success_PS[i] = np.round(p * 100) / 100

    # Determine connectivity amongst clients
    connectivity_mat_clients = np.zeros([num_clients, num_clients])
    for i in range(num_clients):
        for j in range(num_clients):
            p = min(1, np.exp(-dist_clients_clients[i][j] / 30 + 5.2))
            if p > 0.99:
                connectivity_mat_clients[i][j] = 1

    with open('prob_success_PS.npy', 'wb') as f:
        np.save(f, prob_success_PS)
    with open('connectivity_mat_clients.npy', 'wb') as f:
        np.save(f, connectivity_mat_clients)

    return prob_success_PS, connectivity_mat_clients


def client_locations_mmWave_clusters_intermittent_conn2(num_clients: int = 10):
    """
    :param num_clients: Number of federated edge learning clients
    :return: probability of successful transmission to PS, and inter-client connectivity matrix
    """

    # PS is placed at origin
    PS_loc = np.array([0, 0])

    # Distance of good clients from PS
    circle_good_rad = 159                       # meters. For prob success ~ 0.9

    # Clients angles
    client_vec_deg = np.zeros(num_clients)
    client_vec_deg[3] = 2 * np.pi / 3
    client_vec_deg[6] = 4 * np.pi / 3

    x = np.zeros(num_clients)
    y = np.zeros(num_clients)

    # Determining the Cartesian coordinates of the clients with good connectivity
    x[0] = circle_good_rad * np.cos(client_vec_deg[0])
    y[0] = circle_good_rad * np.sin(client_vec_deg[0])
    x[3] = circle_good_rad * np.cos(client_vec_deg[3])
    y[3] = circle_good_rad * np.sin(client_vec_deg[3])
    x[6] = circle_good_rad * np.cos(client_vec_deg[6])
    y[6] = circle_good_rad * np.sin(client_vec_deg[6])

    d = 159                       # Distance of bad clients in each cluster to the good client of the cluster
    d1 = 156

    # Cluster 1
    ang1 = 1.582
    x[1] = x[0] + d1 * np.cos(ang1)
    y[1] = y[0] + d1 * np.sin(ang1)
    print("Distance = {}".format(np.sqrt(x[1] ** 2 + y[1] ** 2)))

    ang2 = - 1.582
    x[2] = x[0] + d1 * np.cos(ang2)
    y[2] = y[0] + d1 * np.sin(ang2)

    # Cluster 2
    ang4 = 2 * np.pi / 3 + 1.582
    x[4] = x[3] + d1 * np.cos(ang4)
    y[4] = y[3] + d1 * np.sin(ang4)

    ang5 = 2 * np.pi / 3 - 1.582
    x[5] = x[3] + d1 * np.cos(ang5)
    y[5] = y[3] + d1 * np.sin(ang5)

    # Cluster 3
    ang7 = 4 * np.pi / 3 - 1.582
    x[7] = x[6] + d1 * np.cos(ang7)
    y[7] = y[6] + d1 * np.sin(ang7)

    ang8 = 4 * np.pi / 3 - 1.54
    x[8] = x[6] + d * np.cos(ang8)
    y[8] = y[6] + d * np.sin(ang8)

    ang9 = 4 * np.pi / 3 + 1.582
    x[9] = x[6] + d1 * np.cos(ang9)
    y[9] = y[6] + d1 * np.sin(ang9)

    # Client locations
    loc_clients = []
    for idx in range(num_clients):
        loc_clients.append([x[idx], y[idx]])
    loc_clients = np.array(loc_clients)

    # Compute distances of clients from PS
    sub_clients_PS = loc_clients - PS_loc
    dist_clients_PS2 = np.zeros(len(sub_clients_PS))
    for idx in range(len(sub_clients_PS)):
        dist_clients_PS2[idx] = sub_clients_PS[idx][0] ** 2 + sub_clients_PS[idx][1] ** 2
    dist_clients_PS = np.sqrt(dist_clients_PS2)

    # Compute pairwise distances between clients
    dist_clients_clients = np.zeros([num_clients, num_clients])
    for i in range(len(loc_clients)):
        for j in range(len(loc_clients)):
            dist_vector = loc_clients[i] - loc_clients[j]
            dist_clients_clients[i][j] = np.linalg.norm(dist_vector, 2)

    # Compute probability of successful transmission to PS
    prob_success_PS = np.zeros(num_clients)
    for i in range(len(dist_clients_PS)):
        p = min(1, np.exp(-dist_clients_PS[i] / 30 + 5.2))
        prob_success_PS[i] = np.round(p * 100) / 100

    # Determine connectivity amongst clients
    connectivity_mat_clients = np.zeros([num_clients, num_clients])
    P = np.zeros([num_clients, num_clients])
    for i in range(num_clients):
        for j in range(num_clients):
            p = min(1, np.exp(-dist_clients_clients[i][j] / 30 + 5.2))
            if p > 0.5:
                P[i][j] = p
                connectivity_mat_clients[i][j] = 1

    with open('prob_success_PS.npy', 'wb') as f:
        np.save(f, prob_success_PS)
    with open('connectivity_mat_clients.npy', 'wb') as f:
        np.save(f, connectivity_mat_clients)

    return prob_success_PS, P, connectivity_mat_clients


if __name__ == "__main__":

    # probs, neighbor_matrix = client_location_mmWave_random_circle_ring(num_clients=10)
    # probs, neighbor_matrix = client_locations_mmWave_perfect_circle_ring(num_clients=10)
    # probs, neighbor_matrix = client_locations_mmWave_clusters(num_clients=10)
    #
    # print(probs)
    # print(neighbor_matrix)

    # print(mmWave_success_prob(250))
    # print(dist_from_origin(x=3, y=1))

    # prob_success_PS, P, connectivity_mat_clients = clustered_topology_intermittent_vs_perfect_compare(
    #     cluster1_ang=2 * np.pi / 3, cluster2_ang=4 * np.pi / 3)
    # print(prob_success_PS)
    # print(P)
    # print(connectivity_mat_clients)

    probs, neighbor_matrix_perfect = client_locations_mmWave_clusters_perfect_conn(num_clients=10)

    print(probs)
    print(neighbor_matrix_perfect)

    probs, P, neighbor_matrix = client_locations_mmWave_clusters_intermittent(num_clients=10)

    print(probs)
    print(P)
    print(neighbor_matrix)
    #
    # probs, neighbor_matrix_perfect = client_locations_mmWave_clusters_perfect_conn2(num_clients=10)
    #
    # print(probs)
    # print(neighbor_matrix_perfect)
    #
    # probs, ngbr_probs, neighbor_matrix_int = client_locations_mmWave_clusters_intermittent_conn2(num_clients=10)
    #
    # print(probs)
    # print(ngbr_probs)
    # print(neighbor_matrix_int)

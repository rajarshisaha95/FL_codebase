import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["figure.figsize"] = [7.50, 7.50]
plt.rcParams["figure.autolayout"] = True

from colour import Color


def test_plot():
    point1 = [1, 2]
    point2 = [3, 4]
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.plot(x_values, y_values, 'bo', linestyle="--")
    plt.text(point1[0]-0.015, point1[1]+0.25, "Point1")
    plt.text(point2[0]-0.050, point2[1]-0.25, "Point2")
    plt.grid()
    plt.show()


def mmWave_cluster_topology():

    num_clients = 10

    PS = [0, 0]
    # Distance of good clients from PS
    circle_good_rad = 159  # meters. For prob success ~ 0.9

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

    d = 70  # Distance of bad clients in each cluster to the good client of the cluster

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
    x_values = []
    y_values = []
    for i in range(num_clients):
        x_values.append(x[i])
        y_values.append(y[i])

    print(x_values)
    print(y_values)

    # Place clients and PS
    # plt.scatter(c=np.linspace(0,1,101), cmap='viridis')
    plt.scatter(x_values, y_values, s=50, c='b', marker='o')
    plt.scatter(PS[0], PS[1], s=250, c='g', marker='o')
    plt.text(PS[0] + 2.25, PS[1] + 10.25, "PS", weight="bold")

    # Show connections between good clients and PS
    plt.plot([PS[0], x[0]], [PS[1], y[0]], 'k-')
    plt.plot([PS[0], x[3]], [PS[1], y[3]], 'k-')
    plt.plot([PS[0], x[6]], [PS[1], y[6]], 'k-')

    # Show connections between bad clients and PS
    plt.plot([PS[0], x[1]], [PS[1], y[1]], 'k', linestyle='dotted')
    plt.plot([PS[0], x[2]], [PS[1], y[2]], 'k', linestyle='dotted')
    plt.plot([PS[0], x[4]], [PS[1], y[4]], 'k', linestyle='dotted')
    plt.plot([PS[0], x[5]], [PS[1], y[5]], 'k', linestyle='dotted')
    plt.plot([PS[0], x[7]], [PS[1], y[7]], 'k', linestyle='dotted')
    plt.plot([PS[0], x[8]], [PS[1], y[8]], 'k', linestyle='dotted')
    plt.plot([PS[0], x[9]], [PS[1], y[9]], 'k', linestyle='dotted')

    # Cluster 1 connections
    plt.plot([x[0], x[1]], [y[0], y[1]], 'k-')
    plt.plot([x[1], x[2]], [y[1], y[2]], 'k-')
    plt.plot([x[0], x[2]], [y[0], y[2]], 'k-')
    plt.text(x[1] - 30.25, y[1] + 10.25, "Cluster 1", weight="bold")

    # Cluster 2 connections
    plt.plot([x[3], x[4]], [y[3], y[4]], 'k-')
    plt.plot([x[4], x[5]], [y[4], y[5]], 'k-')
    plt.plot([x[3], x[5]], [y[3], y[5]], 'k-')
    plt.text(x[4] - 15.25, y[4] + 20.25, "Cluster 2", weight="bold")

    # Cluster 3 connections
    plt.plot([x[6], x[7]], [y[6], y[7]], 'k-')
    plt.plot([x[6], x[8]], [y[6], y[8]], 'k-')
    plt.plot([x[6], x[9]], [y[6], y[9]], 'k-')
    plt.plot([x[7], x[8]], [y[7], y[8]], 'k-')
    plt.plot([x[7], x[9]], [y[7], y[9]], 'k-')
    plt.plot([x[8], x[9]], [y[8], y[9]], 'k-')
    plt.text(x[9] - 15.25, y[9] - 15.25, "Cluster 3", weight="bold")

    plt.grid()
    plt.show()
    plt.colorbar()

    print(x[0])
    print(y[0])


def discrete_matshow(data):
    # get discrete colormap
    cmap = plt.get_cmap('RdBu', np.max(data) - np.min(data) + 1)
    # set limits .5 outside true range
    mat = plt.matshow(data, cmap=cmap, vmin=np.min(data) - 0.5,
                      vmax=np.max(data) + 0.5)
    # tell the colorbar to tick at integers
    cax = plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data) + 1))
    plt.show()


def test_colormap():
    """
    Test module to create a colormap
    """

    red = Color("red")
    colors = list(red.range_to(Color("green"), 100))

    print(colors)

    mat = np.random.random((10, 10))
    plt.imshow(mat, origin="lower", cmap='colors', interpolation='nearest')
    plt.colorbar()
    plt.show()

    # generate data
    a = np.random.randint(1, 9, size=(10, 10))
    discrete_matshow(a)


def mmWave_cluster_topology_2_intermittent():

    num_clients = 10

    PS = [0, 0]
    # Distance of good clients from PS
    circle_good_rad = 159  # meters. For prob success ~ 0.9

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

    d = 159  # Distance of bad clients in each cluster to the good client of the cluster
    d1 = 156

    # Cluster 1
    ang1 = 1.582
    x[1] = x[0] + d1 * np.cos(ang1)
    y[1] = y[0] + d1 * np.sin(ang1)

    ang2 = -1.582
    x[2] = x[0] + d1 * np.cos(ang2)
    y[2] = y[0] + d1 * np.sin(ang2)

    # Cluster 2
    ang4 = 2 * np.pi / 3 + 1.582
    x[4] = x[3] + d1 * np.cos(ang4)
    y[4] = y[3] + d1 * np.sin(ang4)

    ang5 = 2 * np.pi / 3 - 1.582
    x[5] = x[3] + d * np.cos(ang5)
    y[5] = y[3] + d * np.sin(ang5)

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
    sub_clients_PS = loc_clients - PS
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

    # Client locations
    x_values = []
    y_values = []
    for i in range(num_clients):
        x_values.append(x[i])
        y_values.append(y[i])

    print(prob_success_PS)
    print(connectivity_mat_clients)

    print(x_values)
    print(y_values)

    # Place clients and PS
    fig, ax = plt.subplots()
    plt.scatter(x_values, y_values, s=50, c='b', marker='o')
    plt.scatter(PS[0], PS[1], s=250, c='k', marker='o')
    plt.text(PS[0] + 2.25, PS[1] + 10.25, "PS", weight="bold")

    # Show connections between good clients and PS
    plt.plot([PS[0], x[0]], [PS[1], y[0]], 'k', linestyle="dashed")
    plt.plot([PS[0], x[3]], [PS[1], y[3]], 'k', linestyle="dashed")
    plt.plot([PS[0], x[6]], [PS[1], y[6]], 'k', linestyle="dashed")

    # Client 1's connections
    plt.plot([x[0], x[1]], [y[0], y[1]], 'k', linestyle="dotted")
    plt.plot([x[0], x[2]], [y[0], y[2]], 'k', linestyle="dotted")

    # Client 2's connections
    plt.plot([x[1], x[5]], [y[1], y[5]], 'k', linestyle="dotted")

    # Client 3's connections
    plt.plot([x[2], x[9]], [y[2], y[9]], 'k', linestyle="dotted")

    # Client 4's connections
    plt.plot([x[3], x[4]], [y[3], y[4]], 'k', linestyle="dotted")
    plt.plot([x[3], x[5]], [y[3], y[5]], 'k', linestyle="dotted")

    # Client 5's connections
    plt.plot([x[4], x[7]], [y[4], y[7]], 'k', linestyle="dotted")
    plt.plot([x[4], x[8]], [y[4], y[8]], 'k', linestyle="dotted")

    # Client 7's connections
    plt.plot([x[6], x[7]], [y[6], y[7]], 'k', linestyle="dotted")
    plt.plot([x[6], x[8]], [y[6], y[8]], 'k', linestyle="dotted")
    plt.plot([x[6], x[9]], [y[6], y[9]], 'k', linestyle="dotted")

    # Client 8's connections
    plt.plot([x[7], x[8]], [y[7], y[8]], 'k', linestyle="dotted")

    ax.set_xlabel("x", weight="bold", size=20)
    plt.xticks(weight='bold', fontsize="15")
    ax.set_ylabel("y", weight="bold", size=20)
    plt.yticks(weight='bold', fontsize="15")
    ax.set_title("mmWave: Intermittent collaboration", weight="bold", size=20)
    plt.grid(which="both", axis="both")
    plt.show()

    print(x[0])
    print(y[0])


def mmWave_cluster_topology_2_perfect():

    num_clients = 10

    PS = [0, 0]
    # Distance of good clients from PS
    circle_good_rad = 159  # meters. For prob success ~ 0.9

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

    d = 159  # Distance of bad clients in each cluster to the good client of the cluster
    d1 = 156

    # Cluster 1
    ang1 = 1.582
    x[1] = x[0] + d1 * np.cos(ang1)
    y[1] = y[0] + d1 * np.sin(ang1)

    ang2 = -1.582
    x[2] = x[0] + d1 * np.cos(ang2)
    y[2] = y[0] + d1 * np.sin(ang2)

    # Cluster 2
    ang4 = 2 * np.pi / 3 + 1.582
    x[4] = x[3] + d1 * np.cos(ang4)
    y[4] = y[3] + d1 * np.sin(ang4)

    ang5 = 2 * np.pi / 3 - 1.582
    x[5] = x[3] + d * np.cos(ang5)
    y[5] = y[3] + d * np.sin(ang5)

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
    sub_clients_PS = loc_clients - PS
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

    # Client locations
    x_values = []
    y_values = []
    for i in range(num_clients):
        x_values.append(x[i])
        y_values.append(y[i])

    print(prob_success_PS)
    print(connectivity_mat_clients)

    print(x_values)
    print(y_values)

    # Place clients and PS
    fig, ax = plt.subplots()
    plt.scatter(x_values, y_values, s=50, c='b', marker='o')
    plt.scatter(PS[0], PS[1], s=250, c='k', marker='o')
    plt.text(PS[0] + 2.25, PS[1] + 10.25, "PS", weight="bold")

    # Show connections between good clients and PS
    plt.plot([PS[0], x[0]], [PS[1], y[0]], 'k', linestyle="dashed")
    plt.plot([PS[0], x[3]], [PS[1], y[3]], 'k', linestyle="dashed")
    plt.plot([PS[0], x[6]], [PS[1], y[6]], 'k', linestyle="dashed")

    # Client 0's connections
    plt.plot([x[0], x[1]], [y[0], y[1]], 'k', linestyle="solid", linewidth=1)
    plt.plot([x[0], x[2]], [y[0], y[2]], 'k', linestyle="solid", linewidth=1)

    # Client 1's connections
    plt.plot([x[1], x[5]], [y[1], y[5]], 'k', linestyle="solid", linewidth=1)

    # Client 2's connections
    plt.plot([x[2], x[9]], [y[2], y[9]], 'k', linestyle="solid", linewidth=1)

    # Client 3's connections
    plt.plot([x[3], x[4]], [y[3], y[4]], 'k', linestyle="solid", linewidth=1)

    # Client 4's connections
    plt.plot([x[4], x[7]], [y[4], y[7]], 'k', linestyle="solid", linewidth=1)
    plt.plot([x[4], x[8]], [y[4], y[8]], 'k', linestyle="solid", linewidth=1)

    # Client 6's connections
    plt.plot([x[6], x[7]], [y[6], y[7]], 'k', linestyle="solid", linewidth=1)
    plt.plot([x[6], x[9]], [y[6], y[9]], 'k', linestyle="solid", linewidth=1)

    # Client 7's connections
    plt.plot([x[7], x[8]], [y[7], y[8]], 'k', linestyle="solid", linewidth=1)

    ax.set_xlabel("x", weight="bold", size=20)
    plt.xticks(weight='bold', fontsize="15")
    ax.set_ylabel("y", weight="bold", size=20)
    plt.yticks(weight='bold', fontsize="15")
    ax.set_title("mmWave: Perfect collaboration", weight="bold", size=20)
    plt.grid(which="both", axis="both")
    plt.show()

    print(x[0])
    print(y[0])


if __name__ == "__main__":

    # test_plot()
    # mmWave_cluster_topology()

    mmWave_cluster_topology_2_intermittent()
    mmWave_cluster_topology_2_perfect()

    # test_colormap()
import math
import numpy as np
import matplotlib
# import scipy
# import scipy.spatial as sp
from scipy.spatial.distance import pdist, squareform

def repmat(A,a,b):
    ncol = A.shape[0]
    nrow = A.shape[1]
    B = np.zeros([ncol*a, nrow*b])
    for i in range(0, ncol*a):
        for j in range(0, nrow*b):
            B[i, j] = A[i%ncol, j%nrow]
    return B

d = 7    # scale of the lattice  晶格尺度
r = 1.2 * d  # interaction range  交互作用范围
d_prime = 0.6 * d   # scale of the lattice for beta-agents
r_prime = 1.2 * d_prime    # interaction range for beta-agents
delta = 1
delta_prime = 0.5

epsilon = 0.1

a = b = 5
c = abs(a - b) / math.sqrt(4 * a * b)

c1 = np.array([1, 12, 5])
c2 = 2 * np.sqrt(c1)
Ms = np.matrix([[100, 110, 120, 130, 150, 160], [20,  60, 40, -20, 40, 0], [10, 4, 2, 5, 5, 3]])

# 画图 障碍
yk = Ms[1:2, :]
# obstacle's coordinates 障碍的坐标
Rk = Ms[3, :]
# obstacle's radius  障碍半径
h_alpha = 0.2
h_beta = 0.9

# 映射函数
r_sn = 1 / epsilon * (np.sqrt(1 + epsilon * r**2) - 1)
d_sn = 1 / epsilon * (np.sqrt(1 + epsilon * d**2) - 1)
r_primesn = 1/epsilon*(np.sqrt(1 + epsilon * r_prime**2) - 1)
d_primesn = 1/epsilon*(np.sqrt(1 + epsilon * d_prime**2) - 1)

n = 150
#  数量
q = -40 + 120 * np.random.random((2, n))
#  初始坐标(col-vector)
p = np.zeros((2, n))
# 初始速度(col-vector)
rad = np.arctan2(p[1, :], p[0, :])
#  角度
q_d = np.array([[200], [30]])
# initial position of gamma-agent 初始位置
p_d = np.array([[5], [0]])
# 初始速度
X = np.array([q[0, :]-0.5 * np.cos(rad)-0.5 * np.sin(rad), q[0, :]-0.5 * np.cos(rad)+0.5 * np.sin(rad), q[0, :] + 1 * np.cos(rad)])
# 可能出现错误点
Y = np.array([q[1, :]-0.5 * np.sin(rad)+0.5 * np.cos(rad), q[1, :]-0.5 * np.sin(rad)-0.5 * np.cos(rad), q[1, :] + 1 * np.sin(rad)])
# use multi-triangle represent the agents   使用 multi-triangle（多三角形） 表示 the agents
# p1 = patch(X,Y,'black')
# p2 = plot(q_d(1),q_d(2),'x')

q_r = q_d
p_r = p_d

q_show = q
p_show = p

delta_t = 0.03
# step-size = 0.01s
iterations = 2000
# times of interation 迭代时间
qk_rowlist = 2 * np.ones((1, n))
# 绘图点

for i in range(iterations):
    q_z = np.transpose(q)
    qji_square = squareform(pdist(q_z, 'sqeuclidean'))
    Ni_alpha = (qji_square > 0) & (qji_square < r ** 2)
    # 这里Ni_alpha并不是0或1 而是ture或false 可能的出错点
    qji_sn = 1 / epsilon * (np.sqrt(1 + epsilon * qji_square) - 1)
    # 1 / 0.1 *（√（1 + 0.1 * qji_square）-1) 范数的映射函数
    Lattice_alpha = (qji_sn > 0) & (qji_sn < d_sn + delta)

    # 凹凸函数
    z_alpha = qji_sn
    # the rho_h_alpha function
    z1_alpha = z_alpha / r_sn * Ni_alpha
    # guarantee rho_h_alpha = 0 if z1 > 1      不明
    # if z1_alpha belongs to (0,h_alpha)
    rho_h1_alpha = (z1_alpha > 0) & (z1_alpha < h_alpha)
    # if z1_alpha belongs to [h_alpha,1]
    rho_h2_alpha = 0.5 * (1 + np.cos(np.pi * (z1_alpha - h_alpha) / (1 - h_alpha))) * ((z1_alpha >= h_alpha) & (z1_alpha <= 1))
    rho_h_alpha = rho_h1_alpha + rho_h2_alpha
    A = rho_h_alpha
    # spatical adjacency matrix  邻接矩阵

    # 集群势能函数 phi
    # the phi function
    z2_alpha = (z_alpha - d_sn) * Ni_alpha
    # a global minimum at z = d_sigma
    z3_alpha = z2_alpha + c
    # the sigma1 function
    sigma1_alpha = z3_alpha / np.sqrt(1 + z3_alpha ** 2)
    phi = 0.5 * ((a+b) * sigma1_alpha + (a - b))
    # the phi_alpha function
    phi_alpha = rho_h_alpha * phi
    # reshape 改变矩阵维度
    q_gai = q_z.reshape(300, 1)
    # 不明拉普拉斯算子？
    qji = repmat(q, n, 1) - q_gai[:]
    # q(j) - q(i)repmat形成一个n * 1的矩阵以q为样本拷贝
    repelem = lambda a, x, y: np.repeat(np.repeat(a, x, axis=0), y, axis=1)
    nij = qji / repelem(1 + epsilon * qji_sn, 2, 1)
    gradient_base_alpha = repelem(phi_alpha, 2, 1) * nij
    # gradient - baseterm
    p_z = np.transpose(p)
    p_gai = p_z.reshape(300, 1)
    pji = repmat(p, n, 1) - p_gai[:]
    # consensus_alpha = repelem(A, 2, 1) * pji
    consensus_alpha = repelem(A, 2, 1) * pji
    # consensusterm

    # the control input
    u_alpha = reshape(c1(1) * sum(gradient_base_alpha, 2) + c2(1) * sum(consensus_alpha, 2), 2, [])

    qyk = q_gai[:] - repmat(yk, n, 1) \
    # q(i) - y(k)
    qyk_norm = pdist2(q',yk')
    # | | qi - yk | |
    mu = Rk / qyk_norm
    # mu = Rk / | | qi - yk | |
    ak = qyk / repelem(qyk_norm, 2, 1)
    # unit normal ak




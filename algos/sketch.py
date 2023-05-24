import torch
import time
from algos.common import Memory
from utils.utils import parameters_to_grad_vector, count_parameter

class Sketch:
    # Online Sketch object for low rank approximation of matrices. Allows for the feeding in of new data
    def __init__(self, n_params, columns=None, r=None, k=None, l=None, input_matrix=None):
        """
        init function. n_params is the size of the vectors we are adding to the sketch.
        # According to the matrix sketching paper, m is n_params, n is data_points (online updating),
        # data_points is the number of columns. Default is square. Can have a 0 input, and can increase while updating
        # r is the target rank
        # k is k the stored rank, l is l
        # We want l to be greater than k
        # input_matrix is an optional input for an initial sketched matrix
        """
        self.n_params = n_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if columns is not None:
            self.columns = columns
        else: self.columns = n_params
        # initialize sketching parameters

        if r is not None: self.r = r
        else: self.r = int(self.n_params/8)

        if k is not None: self.k = k
        else: self.k = int(self.n_params/4)

        if l is not None: self.l = l
        else: self.l = min(self.k * 2, self.n_params)

        assert self.l >= self.k, 'l needs to be greater than k'


        print("n_params in sketch: ",n_params)
        print("k,l in sketch: ", k, l)

        # initialize test matrices
        # QUESTION: .cpu() or no?
        self.omega = torch.randn(self.columns, self.k).cpu()
        self.psi = torch.randn(self.l, self.n_params).cpu()

        # initialize sketch matrices
        # QUESTION: .cpu() or no?
        self.Y = torch.zeros(self.n_params, self.k).cpu()
        self.W = torch.zeros(self.l, self.columns).cpu()  # data_points = n

        if input_matrix is not None:
            self.update_add(input_matrix)

    def update_concatenate(self, A):
        """
        Given a tensor of new column vectors we want to append/concatenate to our sketch, update
        the test and sketch matrices
        Keeps number of rows constant, just changes number of columns
        :param A: tensor of size (n_params, new_data_points)
        """
        assert A.size()[0] == self.n_params, "self.n_params does not match with input row dimension"
        num_data_points = A.size()[1]
        self.columns += num_data_points

        # update matrices, mostly by concatenation
        new_omega = torch.randn(num_data_points, self.k) # what gets concatenated onto omega
        self.omega = torch.cat((self.omega, new_omega), dim=0)
        # self.psi does not need to be updated

        self.Y = self.Y + torch.matmul(A, new_omega)

        new_W = torch.matmul(self.psi, A) # what gets concatenated onto the end of W
        self.W = torch.cat((self.W, new_W), dim=1)

    def update_add(self, A):
        """
        Adds a matrix A to the currently sketched matrix, and updates stored variables
        """
        print(A.type())
        self.Y = self.Y + torch.matmul(A,self.omega)
        self.W = self.W + torch.matmul(self.psi, A)

    def update_add_vector(self,v):
        """
        Adds the matrix vv^T to the sketched matrix and updates stored matrices
        Assumes v is a column vector of the right size
        Removes a factor of n from the time complexity compared to update_add
        """
        # we do the same thing as update_add except in a more efficient way
        self.Y = self.Y + torch.matmul(v, torch.matmul(torch.transpose(v,0,1),self.omega))  # Y + v(v^T*Omega)
        self.W = self.W + torch.matmul(torch.matmul(self.psi, v), torch.transpose(v, 0, 1))  # W + (Psi*v)v^T


    def low_rank_approx(self):
        """
        Splits the sketch into two matrices Q, X that when multiplied yield an approximation of the stored matrix
        Q has orthonormal columns
        """
        Q, R = torch.linalg.qr(self.Y)  # QUESTION: is this orthonormal or orthogonal?
        U, T = torch.linalg.qr(torch.matmul(self.psi, Q))
        X = torch.matmul(torch.linalg.pinv(T), torch.matmul(torch.transpose(U,0,1), self.W))
        #X = torch.matmul(torch.linalg.pinv(torch.matmul(self.psi, Q)), self.W)

        # Now, the full matrix A is approximately QX
        return Q, X

    def low_rank_sym_approx(self):
        """
        Yields the low rank symmetric approximation A = USU^*
        Requires the sketched matrix to be square
        U has orthonormal columns
        Returns U, S
        :return:
        """
        Q, X = self.low_rank_approx()
        #print(Q.size())
        #print(X.size())
        #print(torch.cat((Q,torch.transpose(X,0,1)),dim=1).size())
        U, T = torch.linalg.qr(torch.cat((Q,torch.transpose(X,0,1)),dim=1))
        # T1 = T[:, 1: self.k]  # old code
        # T2 = T[:, (self.k + 1): (2*self.k)]
        T1 = T[:, 0: self.k]
        T2 = T[:, self.k: 2 * self.k]
        S = (torch.matmul(T1,torch.transpose(T2,0,1)) + torch.matmul(T2,torch.transpose(T1,0,1)))/2
        return U, S

    def fixed_rank_sym_approx(self, truncate=False):
        U, S = self.low_rank_sym_approx()
        #  supposed to be eigen decomposition but it's a real symmetric matrix so should be fine
        V, D, _ = torch.linalg.svd(S)  # get eigendecomposition
        if truncate:
            V_truncated = V[:,:self.r]  # truncate eigenvectors
            diag_D_truncated = torch.diag(D[:self.r])  # truncate eigenvalues
            U = torch.matmul(U,V_truncated)
            return U, diag_D_truncated
        else:
            U = torch.matmul(U, V)
            return U,D


    def calc_sketch(self):
        """
        Yields the approximation of the sketched matrix
        :return:
        """
        Q, X = self.low_rank_approx()
        return torch.matmul(Q,X)

    def calc_ortho_basis(self):
        Q, R = torch.linalg.qr(self.Y) # QUESTION: is this orthonormal or orthogonal? Answ: it's orthonormal
        return Q


# temp_sketch = Sketch(15, 15, 11, 12, 13)  # n_params = 100, k = 50, l = 50
# # # A = torch.randn(10, 10)  # num_data_points = 50
# u = torch.FloatTensor([[2],[-1],[0],[1],[3],[9],[-1],[0],[1],[3],[9],[-1],[0],[1],[3]])
# v = torch.FloatTensor([[3],[2],[1],[-2],[0],[2.8],[2],[1.3],[-2],[0.1],[3],[2],[1],[-2],[0]])
# w = torch.FloatTensor([[2.8],[2],[1.3],[-2],[0.1],[3],[2],[1],[-2],[0],[3],[2],[1],[-2],[0]])
# #
# # #A = torch.FloatTensor([[1,1,1],[2,3,4],[5,7,9.5]])
# A = torch.matmul(u,torch.transpose(u,0,1)) + torch.matmul(v,torch.transpose(v,0,1)) + torch.matmul(w,torch.transpose(w,0,1))
# temp_sketch.update_add_vector(u)
# temp_sketch.update_add_vector(v)
# temp_sketch.update_add_vector(w)
# #
# #
# B = temp_sketch.calc_sketch()
# U, S = temp_sketch.low_rank_sym_approx()
# C = torch.matmul(torch.matmul(U,S),torch.transpose(U,0,1))
# print(A)
# print(B)
# print((B+B.T)/2)
# print(C)
# print(torch.norm(A-B))
# print(torch.norm(A-C))
# print(torch.norm(A))
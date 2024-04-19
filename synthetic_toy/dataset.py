import torch

from torch.distributions.multivariate_normal import MultivariateNormal
# make dataloader for custom data


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, n=1000, d=3):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n = n
        self.d = d

        a_mean = torch.zeros(d, dtype=torch.float32)
        a_mean[1] = 1

        b_mean = torch.zeros(d, dtype=torch.float32)
        b_mean[1] = -1

        a = MultivariateNormal(a_mean, torch.eye(d) * 0.01)
        b = MultivariateNormal(b_mean, torch.eye(d) * 0.01)

        a_points = a.sample((n,))
        b_points = b.sample((n,))  

        # normalize
        a_points = a_points / a_points.norm(dim=1).view(-1, 1)
        b_points = b_points / b_points.norm(dim=1).view(-1, 1)

        self.ab_status = 'cartesian' # 'spherical' or 'cartesian'


        # each point is 3D ([x, y, z])

        # convert to angles ([theta, phi])


        self.ab = torch.stack([a_points, b_points], dim=0)
        # shape: (2, n, d)

        # self.ab.requires_grad = True

    def convert_to_spherical(self):
        self.ab = self.get_spherical_ab()
        self.ab_status = 'spherical'

    def convert_to_cartesian(self):
        self.ab = self.get_cartesian_ab()
        self.ab_status = 'cartesian'

    def get_cartesian_ab(self):

        assert self.ab_status == 'spherical'

        a = self.ab[0]
        b = self.ab[1]

        a_cart = self.sph2cart(a)
        b_cart = self.sph2cart(b)

        ab = torch.stack([a_cart, b_cart], dim=0)

        return ab
    
    def get_spherical_ab(self):
            
        assert self.ab_status == 'cartesian'    
        
            
        a = self.ab[0]
        b = self.ab[1]

        a_sph = self.cart2sph(a)
        b_sph = self.cart2sph(b)

        ab = torch.stack([a_sph, b_sph], dim=0)

        return ab


    def clamp_spherical_coords(self):
        assert self.ab_status == 'spherical'

        self.ab[0, :, -1] = torch.clamp(self.ab[0, :, 0], min=0, max=2 * 3.14159)
        self.ab[0, :, :-1] = torch.clamp(self.ab[1], min=0, max=3.14159)

    def cart2sph(self, points):
        '''
        points is a tensor of shape (n, d)
        '''

        n_dimensions = points.shape[1]
        n_points = points.shape[0]

        # angular_point = torch.zeros(n_dimensions-1, dtype=torch.float32)
        angular_points = torch.zeros((n_points, n_dimensions-1), dtype=torch.float32, device=self.device)

        for d in range(angular_points.shape[1]):
            angular_points[:, d] = torch.atan2(torch.norm(points[:, d+1:], dim=1), points[:, d])
            # clamp to avoid NaN

        return angular_points

    def sph2cart(self, angular_points):
        '''
        angular_points is a tensor of shape (n, d-1)
        '''

        n_dimensions = angular_points.shape[1] + 1
        n_points = angular_points.shape[0]

        points = torch.zeros((n_points, n_dimensions), dtype=torch.float32, device=self.device)

        for d in range(angular_points.shape[1]):
            points[:, d] = torch.prod(torch.sin(angular_points[:, :d]), dim=1) * torch.cos(angular_points[:, d])
            # points[:, d+1] = torch.cos(angular_points[:, d])


        return points


    def normalize_points(self):
        self.ab.requires_grad = False
        self.ab = self.ab / self.ab.norm(dim=2).view(2, self.n, 1)
        self.ab.requires_grad = True

    def move_to_device(self):

        # self.ab.requires_grad = False
        self.ab = self.ab.to(self.device)
        self.ab.requires_grad = True


    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # return self.ab[idx], self.labels[idx]
        return self.ab[0, idx], self.ab[1, idx]

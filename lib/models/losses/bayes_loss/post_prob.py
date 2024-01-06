import torch
from torch.nn import Module

class Post_Prob(Module):
    def __init__(self, sigma, c_size, stride, background_ratio, use_background, device):
        super(Post_Prob, self).__init__()
        assert c_size % stride == 0

        self.sigma = sigma
        self.bg_ratio = background_ratio
        self.device = device
        # coordinate is same to image space, set to constant since crop size is same
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2
        self.cood.unsqueeze_(0)
        self.softmax = torch.nn.Softmax(dim=0)
        self.use_bg = use_background

    #这段代码是在计算两个点集之间的欧氏距离。下面是每行代码的解释：
    # 1. `x = all_points[:, 0].unsqueeze_(1)`: 这行代码从`all_points`中取出所有点的x坐标，并将其形状从`(n,)`变为`(n,1)`，其中`n`是点的数量。
    # 2. `y = all_points[:, 1].unsqueeze_(1)`: 这行代码从`all_points`中取出所有点的y坐标，并将其形状从`(n,)`变为`(n,1)`。
    # 3. `x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood`: 这行代码计算了所有点的x坐标与`self.cood`之间的平方欧氏距离。
    # 4. `y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood`: 这行代码计算了所有点的y坐标与`self.cood`之间的平方欧氏距离。
    # 5. `y_dis.unsqueeze_(2)`: 这行代码将`y_dis`的形状从`(n, m)`变为`(n, m, 1)`，其中`m`是`self.cood`的长度。
    # 6. `x_dis.unsqueeze_(1)`: 这行代码将`x_dis`的形状从`(n, m)`变为`(n, 1, m)`。
    # 7. `dis = y_dis + x_dis`: 这行代码计算了所有点与`self.cood`之间的平方欧氏距离。由于`y_dis`和`x_dis`的形状分别为`(n, m, 1)`和`(n, 1, m)`，
    # 所以它们可以通过广播机制相加，得到的结果`dis`的形状为`(n, m, m)`。这意味着`dis[i, j, k]`表示`all_points[i]`与`(self.cood[j], self.cood[k])`之间的平方欧氏距离。
    
    def forward(self, points, st_sizes):
        num_points_per_image = [len(points_per_image) for points_per_image in points]
        all_points = torch.cat(points, dim=0)

        if len(all_points) > 0:
            x = all_points[:, 0].unsqueeze_(1)
            y = all_points[:, 1].unsqueeze_(1)
            x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood
            y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
            y_dis.unsqueeze_(2)
            x_dis.unsqueeze_(1)
            dis = y_dis + x_dis
            dis = dis.view((dis.size(0), -1))

            dis_list = torch.split(dis, num_points_per_image)
            prob_list = []
            for dis, st_size in zip(dis_list, st_sizes):
                if len(dis) > 0:
                    if self.use_bg:
                        min_dis = torch.clamp(torch.min(dis, dim=0, keepdim=True)[0], min=0.0)
                        bg_dis = (st_size * self.bg_ratio) ** 2 / (min_dis + 1e-5)
                        dis = torch.cat([dis, bg_dis], 0)  # concatenate background distance to the last
                    dis = -dis / (2.0 * self.sigma ** 2)
                    prob = self.softmax(dis)
                else:
                    prob = None
                prob_list.append(prob)
        else:
            prob_list = []
            for _ in range(len(points)):
                prob_list.append(None)
        return prob_list



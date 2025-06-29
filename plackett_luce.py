import torch
import torch.nn as nn


class PlackettLuceLoss(nn.Module):
    """
    Plackett-Luce Loss for Learning-to-Rank.

    接受一批预测分数和一批真实标签，并计算 Plackett-Luce 负对数似然损失。
    """

    def __init__(self, epsilon=1e-10):
        """
        Args:
            epsilon (float): 用于增强数值稳定性的一个小常数。
        """
        super(PlackettLuceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        计算损失。

        Args:
            y_pred (torch.Tensor): 模型的预测分数，形状为 (batch_size, )。
            y_true (torch.Tensor): 真实的标签，数值越小表示排名越靠前，形状与 y_pred 相同。

        Returns:
            torch.Tensor: 一个标量，表示该批次的平均损失。
        """

        # 1. 获取真实排序的索引
        # 对 y_true 进行升序排序，得到每个位置上应该放置的物品的原始索引
        # descending=False 因为 y_true 值越小，排名越靠前
        true_ranking_indices = torch.argsort(y_true, descending=False)

        # 2. 根据真实排序，重新排列预测分数
        # torch.gather 用于根据索引从源张量中收集值
        # y_pred_sorted_by_true 的每一行都是按照真实排名顺序排列的预测分数
        y_pred_sorted_by_true = torch.gather(y_pred, index=true_ranking_indices)

        # 3. 计算对数似然
        # exp(theta) -> 得到价值 alpha
        alpha_sorted_by_true = torch.exp(y_pred_sorted_by_true)

        # 4. 计算分母项：log(sum(alpha_k)) for k from j to n
        # 这是一个从后往前的累积和。我们可以通过翻转、累加、再翻转的方式高效计算
        flipped_alpha = torch.flip(alpha_sorted_by_true)
        cumulative_sum_alpha = torch.cumsum(flipped_alpha)
        flipped_log_denominator = torch.log(cumulative_sum_alpha + self.epsilon)
        log_denominator = torch.flip(flipped_log_denominator)

        # 5. 计算每个排序的对数似然
        log_likelihood = torch.sum(y_pred_sorted_by_true - log_denominator, dim=1)

        # 6. 计算负对数似然并取平均值作为最终损失
        # 我们希望最大化似然，所以最小化负对数似然
        loss = -torch.mean(log_likelihood)

        return loss


if __name__ == "__main__":
    # 测试 PlackettLuceLoss
    y_pred = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    y_true = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1])
    plackett_luce_loss = PlackettLuceLoss()
    loss = plackett_luce_loss(y_pred, y_true)
    print(loss)

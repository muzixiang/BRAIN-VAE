import argparse
import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import os
import pandas as pd
import math
from numpy import array
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from scipy import signal
import random
from dtaidistance import dtw  # 动态时间规整库
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import pywt
from packages.fsnet import TSEncoder
from einops import rearrange
from packages.timefeatures import time_features
from fastdtw import fastdtw
from sklearn.metrics.pairwise import cosine_similarity

def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    # Ensure deterministic behavior in cuDNN (can impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

#===================================数据加载与处理相关的函数代码============================


#基于Pandas读取xlsx数据加载数据
def load_data(dataSource, dates_col):
    data = pd.read_excel(dataSource, parse_dates=[dates_col])
    return data

def train_test_split(ts, test_ratio):
    train_ratio = 1-test_ratio
    ts_len = len(ts)
    ts_train_len = int(ts_len*train_ratio)
    ts_train = ts[0:ts_train_len]
    ts_test = ts[ts_train_len:-1]
    return ts_train, ts_test

def split_sequence(sequence, hw, pw):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + hw #训练序列窗口截止点
        out_end_ix = end_ix + pw #预测序列窗口截止点
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        
        X.append(seq_x)
        y.append(seq_y)
    print(len(X), len(y))
    return array(X), array(y)


class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input):
        return self.encoder(input, mask=self.mask)



#================================================ 改进的支持Online机制的，结合各层决策融合的VAE模型 ==============================
class Model(nn.Module):
    def __init__(self, device, X_dim, Y_dim, h_dim, Z_dim):
        super(Model, self).__init__()
        self.feature_dim = feature_dim
        self.device = device
        depth = 1
        encoder1 = TSEncoder(input_dims=X_dim,
                            output_dims=h_dim,  # standard ts2vec backbone value
                            hidden_dims=64,  # standard ts2vec backbone value
                            depth=depth,
                            device=device)
        self.encoder1= TS2VecEncoderWrapper(encoder1, mask='all_true').to(self.device)
        self.regressor1 = nn.Linear(h_dim, Y_dim).to(self.device)
        encoder2 = TSEncoder(input_dims= 1,
                            output_dims=h_dim,  # standard ts2vec backbone value
                            hidden_dims=64,  # standard ts2vec backbone value
                            depth=depth,
                            device=device)
        self.encoder2 = TS2VecEncoderWrapper(encoder2, mask='all_true').to(self.device)
        self.regressor2 = nn.Linear(h_dim, Y_dim).to(self.device)
        encoder3 = TSEncoder(input_dims = 1+7,
                             output_dims=h_dim,  # standard ts2vec backbone value
                             hidden_dims=64,  # standard ts2vec backbone value
                             depth=depth,
                             device=device)
        self.encoder3 = TS2VecEncoderWrapper(encoder3, mask='all_true').to(self.device)
        self.regressor3 = nn.Linear(h_dim, Y_dim).to(self.device)

        self.linear_mu = torch.nn.Linear(h_dim, Z_dim).to(self.device)
        self.linear_var = torch.nn.Linear(h_dim, Z_dim).to(self.device)

        self.P_generate = torch.nn.Sequential(
            nn.Linear(Z_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, X_dim)
        ).to(self.device)

        # 注意力网络
        self.AttentionNet = nn.Sequential(
            nn.Linear(h_dim, 50),
            nn.ELU(),
            nn.Linear(50, 1)  # 输出注意力权重
        ).to(self.device)


    def forward(self, X, X_mark): # x: B x T
        X = X.unsqueeze(-1) # x: B x T x 1(input_dim)
        rep1 = self.encoder1.encoder.forward_time(X)
        rep1 = rep1.squeeze(1)
        y_pred1 = self.regressor1(rep1)
        attn_weight1 = self.AttentionNet(rep1)  # 计算注意力权重

        rep2 = self.encoder2(X)
        rep2 = rep2[:, -1]
        y_pred2 = self.regressor2(rep2)
        attn_weight2 = self.AttentionNet(rep2)  # 计算注意力权重

        X = torch.cat([X, X_mark], dim=-1)
        rep3 = self.encoder3(X)
        rep3 = rep3[:, -1]
        y_pred3 = self.regressor3(rep3)
        attn_weight3 = self.AttentionNet(rep3)  # 计算注意力权重

        Att_weight = torch.cat([attn_weight1, attn_weight2, attn_weight3], 1)  # 形状: (mb_size, 3)
        Att_weight = F.softmax(Att_weight, dim=1)  # 形状: (mb_size, 3) #5个TCN block
        Att_weight = Att_weight.unsqueeze(2)  # 形状: (mb_size, 3, 1)

        y_preds = torch.cat([y_pred1, y_pred2, y_pred3], 1)
        y_preds = y_preds.view(X.shape[0], 3, -1)  # 形状: (mb_size, 3, Y_dim)

        reps = torch.cat([rep1, rep2, rep3], 1)
        reps = reps.view(X.shape[0], 3, -1)  # 形状: (mb_size, 3, h_dim)

        # 加权求和
        y_pred = (Att_weight * y_preds).sum(dim=1)  # 形状: (mb_size, Y_dim)
        rep = (Att_weight * reps).sum(dim=1)  # 形状: (mb_size, h_dim)


        z_mu = self.linear_mu(rep)
        z_var = self.linear_var(rep)
        # ============ sample z ===========
        eps = torch.randn(X.shape[0], Z_dim).to(self.device)
        z_sample = z_mu + torch.exp(z_var / 2) * eps
        # ==========  P(X_sample | z) ============
        X_decoded = self.P_generate(z_sample)

        #del X, rep1, rep2, rep3, rep, reps, attn_weight1, attn_weight2, attn_weight3, Att_weight
        return X_decoded, z_mu, z_var, y_pred1, y_pred2, y_pred3, y_pred
        # return y_pred1

    def store_grad(self):
        for name, layer in self.encoder1.named_modules():
            if 'PadConv' in type(layer).__name__:
                #print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()
        for name, layer in self.encoder2.named_modules():
            if 'PadConv' in type(layer).__name__:
                #print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()


def My_loss(X, X_decoded, z_mu, z_var, Y, Y_pred1, Y_pred2, Y_pred3, Y_pred):
    # 重建损失（均方误差）
    recon_loss = F.mse_loss(X_decoded, X, reduction='mean')  # 直接使用 mean 减少手动归一化

    # KL 散度损失
    kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1 - z_var, dim=1)  # 对每个样本求和
    kl_loss = torch.mean(kl_loss)  # 对 batch 求平均

    # 预测损失（均方误差）
    y_pred_loss = F.mse_loss(Y_pred, Y, reduction='mean')  # 直接使用 mean 减少手动归一化
    y_pred1_loss = F.mse_loss(Y_pred1, Y, reduction='mean')  # 直接使用 mean 减少手动归一化
    y_pred2_loss = F.mse_loss(Y_pred2, Y, reduction='mean')  # 直接使用 mean 减少手动归一化
    y_pred3_loss = F.mse_loss(Y_pred3, Y, reduction='mean')  # 直接使用 mean 减少手动归一化
    # 总损失
    Loss = recon_loss + kl_loss + y_pred1_loss + y_pred2_loss + y_pred3_loss + y_pred_loss

    return Loss

######################### 优先经验重放网络 ##################################################

class PriorityNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.net(x)


# 改进后的奖励计算（增加趋势一致性奖励）
def calculate_reward(original, pred_before_replay, pred_after_replay):
    # 基础奖励：MAE改进量
    mae_improve = F.l1_loss(pred_before_replay, original) - F.l1_loss(pred_after_replay, original)

    return mae_improve


def gaussian_kl_divergence(z_mu1, z_var1, z_mu2, z_var2):
    """计算两个高斯分布之间的KL散度：KL(N1||N2)"""
    # 确保方差为正
    z_var1 = torch.clamp(z_var1, min=1e-10)
    z_var2 = torch.clamp(z_var2, min=1e-10)

    # 计算KL散度（逐维度）
    kl = 0.5 * (torch.log(z_var2 / z_var1) +
                (z_var1 + (z_mu1 - z_mu2) ** 2) / z_var2 - 1)

    # 对所有维度求和（使用dim=0表示对第0维求和）
    return torch.sum(kl, dim=0)

if __name__ == '__main__':
    # ================== 参数解析 ==================
    parser = argparse.ArgumentParser(description="使用指定参数运行带有优先级重放的在线VAE预测模型。")

    # --- 数据参数 ---
    parser.add_argument("--data_dir", type=str, default="./data", help="包含输入数据文件的目录")
    parser.add_argument("--filename", type=str, required=True, help="数据文件名 (例如 'TUR4_3')，不含 .csv 后缀")
    parser.add_argument("--train_ratio", type=float, default=0.3, help="用于归一化器拟合的数据比例 (剩余用于在线处理)")

    # --- 模型参数 ---
    parser.add_argument("--hw", type=int, default=60, help="历史窗口大小 (输入序列长度)")
    parser.add_argument("--pw", type=int, default=48, help="预测窗口大小 (输出序列长度)")
    parser.add_argument("--h_dim", type=int, default=128, help="编码器输出和内部网络的隐藏层维度")
    parser.add_argument("--Z_dim", type=int, default=8, help="VAE潜空间维度")
    parser.add_argument('--priority_input_dim', type=int, default=4*8+3, help='优先级网络的输入维度')
    parser.add_argument("--encoder_depth", type=int, default=1, help="TSEncoder模块的深度")

    # --- 在线学习与重放参数 ---
    parser.add_argument("--mb_size", type=int, default=1, help="在线学习的小批量大小 (通常为1)")
    parser.add_argument("--lr", type=float, default=1e-3, help="lr1 主VAE模型优化器的学习率")
    parser.add_argument("--priority_lr", type=float, default=1e-3, help="lr2 优先级网络优化器的学习率")
    parser.add_argument("--replay_frequency", type=int, default=1, help="每处理多少个新批次后执行一次重放")
    parser.add_argument("--replay_times", type=int, default=1, help="每次所选经验的重放次数")
    parser.add_argument("--priority_updates", type=int, default=2, help="每次经验重放时，优先级网络（PriorityNetwork）更新的次数")
    parser.add_argument("--max_storage_size", type=int, default=100000, help="用于重放的最大存储样本数量")
    parser.add_argument('--top_k', type=int, default=10, help='Top-K experiences to replay.')
    # parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for epsilon-greedy exploration in replay selection (if used)") # 移除epsilon，因为代码使用argmax

    # --- 环境与复现性 ---
    parser.add_argument("--seed", type=int, default=42, help="设置随机种子以保证可复现性")
    parser.add_argument("--device", type=str, default="cuda:0", help="用于计算的设备 (例如 'cuda:0' 或 'cpu')")

    # --- 输出参数 ---
    parser.add_argument("--results_dir", type=str, default="./results", # 使用新目录区分
                       help="保存结果CSV文件的目录")
    parser.add_argument("--plots_dir", type=str, default="./plots", # 使用新目录区分
                       help="保存绘图结果的目录")
    parser.add_argument("--output_prefix", type=str, default="BRAIN-VAE-Final",
                       help="输出文件名的前缀")
    # parser.add_argument("--log_interval", type=int, default=100, help="打印日志的间隔（批次数）")

    args = parser.parse_args()

    # ============================================== 数据加载与参数设置 ========================================
    from sklearn.preprocessing import MinMaxScaler
    from pandas import read_csv
    set_seed(args.seed)

    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {args.device}")

    # 确保输出目录存在
    for directory in [args.results_dir, args.plots_dir]:
        os.makedirs(directory, exist_ok=True)

    # 从文件名中提取元素类型
    eletype = args.filename.split('_')[0]
    
    from pandas import read_csv
    # 加载数据
    filepath = os.path.join(args.data_dir, eletype, f"{args.filename}.csv")
    # filepath = os.path.join(args.data_dir, f"{args.filename}.csv")
    df_raw = read_csv(filepath)
    
    # 数据预处理
    first_column_name = df_raw.columns[0]
    df_raw = df_raw.rename(columns={first_column_name: 'date'})
    df_stamp = df_raw[['date']]
    df_raw['date'] = pd.to_datetime(df_stamp['date'])
    ts_stamp = time_features(df_raw, timeenc=2, freq='t')

    ts = df_raw['value'].to_numpy()
    start_index = int(len(ts) * args.train_ratio)

    # 分割数据
    ts_warm = ts[:start_index]
    ts_stamp_warm = ts_stamp[:start_index]
    ts_infer = ts[start_index:]
    ts_stamp_infer = ts_stamp[start_index:]

    # 归一化
    scaler = StandardScaler()
    scaler.fit(ts_warm.reshape(-1, 1))
    ts_warm = scaler.transform(ts_warm.reshape(-1, 1)).flatten()
    ts_infer = scaler.transform(ts_infer.reshape(-1, 1)).flatten()
    
    scaler2 = StandardScaler()
    scaler2.fit(ts_stamp_warm)
    ts_stamp_warm = scaler2.transform(ts_stamp_warm)
    ts_stamp_infer = scaler2.transform(ts_stamp_infer)

    # 分割序列
    X_infer, Y_infer = split_sequence(ts_infer, args.hw, args.pw)
    X_infer_mark, Y_infer_mark = split_sequence(ts_stamp_infer, args.hw, args.pw)
    X_warm, Y_warm = split_sequence(ts_warm, args.hw, args.pw)
    X_warm_mark, Y_warm_mark = split_sequence(ts_stamp_warm, args.hw, args.pw)


    # 将数据移动到GPU上
    X_infer_tensor = torch.FloatTensor(X_infer).to(args.device)
    X_infer_mark_tensor = torch.FloatTensor(X_infer_mark).to(args.device)
    Y_infer_tensor = torch.FloatTensor(Y_infer).to(args.device)


    hw = args.hw  #历史窗口样本序列的长度,决定了输入VAE的维度
    pw = args.pw #预测窗口

    # 创建数据集和数据加载器
    from torch.utils.data import DataLoader, TensorDataset
    mb_size = args.mb_size
    dataset = TensorDataset(X_infer_tensor, X_infer_mark_tensor, Y_infer_tensor)
    dataloader = DataLoader(dataset, batch_size=mb_size)

    Z_dim = args.Z_dim


    h_dim = args.h_dim # FFN的隐层
    feature_dim = 320 # wavelet分解后小波系数特征维度
    c = 0
    lr1 = args.lr
    lr2 = args.priority_lr
    OLVAE = Model(args.device, X_dim=hw, Y_dim=pw, h_dim=h_dim, Z_dim=Z_dim)
    opt = optim.RMSprop(OLVAE.parameters(), lr=lr1)

    priority_net_input_dim = args.priority_input_dim  # 4*8 + 3

    priority_net = PriorityNetwork(input_dim=priority_net_input_dim).to(args.device)
    priority_opt = optim.RMSprop(priority_net.parameters(), lr=lr2)
    epsilon = 0.1  # ε - greedy 策略中的 ε 值


    criterion = nn.MSELoss()


    replay_frequency = args.replay_frequency  # 每训练几个批次进行一次重放
    from collections import deque

    # 初始化全局步数
    global_step = 0
    # 存储潜在表示的列表
    max_storage_size = args.max_storage_size  #  可根据实际情况调整
    experience_cache = deque(maxlen=max_storage_size)


    use_amp = False
    Originals = []
    Preds = []
    OLVAE.eval()
    replay_times = args.replay_times  # 1

    inference_start_time = time.time()
    #============== 在线学习与推理部分 ================
    for i, (X, X_mark, Y) in enumerate(dataloader):
        if i % pw != 0:
            continue
        X_decoded, z_mu, z_var, Y_pred1, Y_pred2, Y_pred3, Y_pred_before = OLVAE.forward(X, X_mark)
        # 从2个子网角度
        loss_current = F.l1_loss(Y_pred_before, Y).cpu()
        opt.zero_grad()
        loss_current.backward()
        opt.step()

        step_current = i
        with torch.no_grad():
            experience_cache.append({
                'z_mu_avg': z_mu.detach().cpu().mean().item(),
                'z_var_avg': z_var.detach().cpu().mean().item(),
                'z_mu': z_mu.detach(),
                'z_var': z_var.detach(),
                'step': step_current,
                'loss': loss_current,
                'X': X.detach(),
                'X_mark': X_mark.detach(),
                'Y': Y.detach()
            })

        # 对所有存储的经验一一构建特征
        experience_features = []
        for experience in experience_cache:
            z_mu_avg_past = experience['z_mu_avg']
            z_var_avg_past = experience['z_var_avg']
            step_past = experience['step']
            loss_past = experience['loss']
            z_mu_past = experience['z_mu'].cpu().flatten()
            z_var_past = experience['z_var'].cpu().flatten()
            time_diff = step_current - step_past
            # 特征向量
            X_past = experience['X'].cpu().flatten()
            X_mark_past = experience['X_mark'].cpu().flatten()

            z_mu_current = z_mu.detach().cpu().flatten()
            z_var_current = z_var.detach().cpu().flatten()
            z_mu_avg_current = z_mu.detach().cpu().mean().item()
            z_var_avg_current = z_var.detach().cpu().mean().item()
            X_current = X.cpu().flatten()
            X_mark_current = X_mark.cpu().flatten()

            kl_divergence = gaussian_kl_divergence(z_mu_past, z_var_past, z_mu_current, z_var_current)

            experience_feature = np.concatenate((z_mu_past, z_var_past,
                                                z_mu_current, z_var_current,
                                                [time_diff, loss_past.detach().cpu(), loss_current.detach().cpu()]
                                                ))  # 注意stored_loss是历史loss, current_loss是当前loss
            experience_features.append(experience_feature)
        experience_features = np.array(experience_features)
        #feature_scaler = StandardScaler()
        feature_scaler = MinMaxScaler()
        feature_scaler.fit(experience_features)
        experience_features = feature_scaler.transform(experience_features)
        experience_features = torch.FloatTensor(experience_features).to(args.device)
        K = args.top_k  # 10
        # 选择进行重放的历史记忆
        if len(experience_cache) < K:
            continue
        else:
            experience_predicted_rewards = priority_net(experience_features).squeeze()

            top_k_values, top_k_indices = torch.topk(experience_predicted_rewards, K)
            top_k_indices = top_k_indices.cpu().numpy().tolist()
            top_k_indices.append(len(experience_features)-1)
            print('选中的重放experience索引: ', top_k_indices)
            for _ in range(replay_times):
                for selected_idx in top_k_indices:
                    selected_experience = experience_cache[selected_idx]
                    # 基于选中的经验，执行重放
                    z_mu_select = selected_experience['z_mu']
                    z_var_select = selected_experience['z_var']
                    X_select = selected_experience['X']
                    X_mark_select = selected_experience['X_mark']
                    Y_select = selected_experience['Y']

                    with torch.no_grad():
                        eps = torch.randn(z_mu_select.shape[0], Z_dim).to(args.device)
                        z_sample_select = z_mu_select + torch.exp(z_var_select/2) * eps
                        X_select_decoded = OLVAE.P_generate(z_sample_select)

                    # 在当前的模型中，基于选择的经验样本，进行再训练
                    X_select_decoded_decoded, z_mu_replay, z_var_replay, Y_pred1_replay, Y_pred2_replay, Y_pred3_replay, Y_pred_replay = OLVAE.forward(X_select, X_mark_select)
                    loss_replay = My_loss(X_select_decoded, X_select_decoded_decoded, z_mu_replay, z_var_replay, Y_select, Y_pred1_replay, Y_pred2_replay, Y_pred3_replay, Y_pred_replay)
                    opt.zero_grad()
                    loss_replay.backward()
                    opt.step()

                    # 在更新的BRAIN-VAE上计算奖励，例如损失的减少量
                    X_decoded, z_mu, z_var, Y_pred1, Y_pred2, Y_pred3, Y_pred_after = OLVAE.forward(X, X_mark)
                    reward = calculate_reward(Y, Y_pred_before, Y_pred_after)

                    for _ in range(args.priority_updates):
                        # 更新 PriorityNetwork
                        real_reward = torch.FloatTensor([reward]).to(args.device)
                        selected_feature = experience_features[selected_idx]
                        predicted_reward = priority_net(selected_feature)
                        loss_priority = F.mse_loss(predicted_reward, real_reward)
                        priority_opt.zero_grad()
                        loss_priority.backward()
                        priority_opt.step()

        #在优化后的模型上重新预测
        with torch.no_grad():
            X_decoded, z_mu, z_var, Y_pred1, Y_pred2, Y_pred3, Y_pred = OLVAE.forward(X, X_mark)
            Originals = np.append(Originals, Y.cpu())
            Preds = np.append(Preds, Y_pred.detach().cpu().numpy())
        print("batch: {} | bach_size: {}".format(i + 1, mb_size))
        torch.cuda.empty_cache()
        gc.collect()

    inference_end_time = time.time()
    inference_duration = inference_end_time - inference_start_time
    print(f"模型推理完成，总共耗时: {inference_duration:.2f} 秒 ({inference_duration / 60:.2f} 分钟)")

    Originals = scaler.inverse_transform(Originals.reshape(-1, 1))
    Preds = scaler.inverse_transform(Preds.reshape(-1, 1))
    print(len(Originals),len(Preds))


    # 将 Preds 转换为 NumPy 数组
    Preds = np.array(Preds)

    # =========== 计算准确性指标 =====================
    mape = metrics.mean_absolute_percentage_error(Originals, Preds)
    print("performance mape {}: {}".format(args.filename, mape))
    rmse = metrics.root_mean_squared_error(Originals, Preds)
    print("performance rmse {}: {}".format(args.filename, rmse))
    mae = metrics.mean_absolute_error(Originals, Preds)
    print("performance mae {}: {}".format(args.filename, mae))
    r2 = metrics.r2_score(Originals, Preds)
    print("performance r2 {}: {}".format(args.filename, r2))

    # ============ 保存csv预测结果 ==============
    results_df = pd.DataFrame({
        'Originals': Originals.flatten(),
        'Predictions': Preds.flatten()
    })
    
    from datetime import datetime
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(args.results_dir, f"{args.output_prefix}_{args.filename}_hw{args.hw}_pw{args.pw}_{current_time}.csv")
    results_df.to_csv(csv_filename, index=False)

    #============ 画图呈现结果 ==================
    plt.figure(figsize=(30, 5))

    # Assuming Originals and Preds are defined and contain data
    plt.grid(True, which='both')
    plt.xticks(size=18)  # 设置x轴坐标大小
    plt.yticks(size=18)  # 设置y轴坐标大小

    plt.plot(Originals, color="blue", ls='-', lw=1, label='original data')
    plt.plot(Preds, color="red", ls='-', lw=1, label='prediction')
    # 在图上添加 MAPE 和 RMSE
    plt.text(0.5, 0.9, f'MAPE: {mape:.4f}', transform=plt.gca().transAxes, fontsize=12, color='red')
    plt.text(0.5, 0.85, f'RMSE: {rmse:.4f}', transform=plt.gca().transAxes, fontsize=12, color='blue')
    plt.text(0.5, 0.80, f'MAE: {mae:.4f}', transform=plt.gca().transAxes, fontsize=12, color='green')
    plt.text(0.5, 0.75, f'R2: {r2:.4f}', transform=plt.gca().transAxes, fontsize=12, color='green')


    # 添加图例
    plt.legend(loc='upper right',fontsize=18)

    # 保存并显示图片
    # from datetime import datetime
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # plt.savefig('./plot/REHZ(reinforcement)_Encoder预测_{}_{}_{}步.png'.format(filename, current_time, pw))
    # plt.show()
    plot_filename = os.path.join(args.plots_dir, f"{args.output_prefix}_{args.filename}_hw{args.hw}_pw{args.pw}_{current_time}.png")
    plt.savefig(plot_filename)
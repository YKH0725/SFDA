"""
PyTorch to Jittor Migration Examples

This file demonstrates how to convert common PyTorch patterns to Jittor.
"""

# ============================================================================
# EXAMPLE 1: 基础张量操作
# ============================================================================

# PyTorch版本
def pytorch_example():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    # 张量创建
    x = torch.randn(2, 3, 224, 224)
    y = torch.zeros((2, 3))
    z = torch.ones((3, 3))
    
    # 张量操作
    stacked = torch.stack([x, x], dim=0)
    catted = torch.cat([y, y], dim=0)
    
    # 形状变换
    reshaped = x.reshape(2, -1)
    permuted = x.permute(0, 2, 3, 1)
    
    # 切片
    sliced = x[:, :, 10:20, 10:20]
    
    return reshaped


# Jittor版本
def jittor_example():
    import jittor as jt
    import jittor.nn as nn
    from jittor.nn import functional as F
    
    # 张量创建 - 基本相同，只需改 torch 为 jt
    x = jt.randn(2, 3, 224, 224)
    y = jt.zeros((2, 3))
    z = jt.ones((3, 3))
    
    # 张量操作 - 相同的API
    stacked = jt.stack([x, x], dim=0)
    catted = jt.concat([y, y], dim=0)
    
    # 形状变换 - 相同
    reshaped = x.reshape(2, -1)
    permuted = x.permute(0, 2, 3, 1)
    
    # 切片 - 相同
    sliced = x[:, :, 10:20, 10:20]
    
    return reshaped


# ============================================================================
# EXAMPLE 2: 神经网络模块
# ============================================================================

# PyTorch版本
class PyTorchModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


# Jittor版本
class JittorModel(jt.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = jt.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = jt.nn.BatchNorm2d(64)
        self.relu = jt.nn.ReLU(inplace=True)
        self.conv2 = jt.nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


# ============================================================================
# EXAMPLE 3: 反向传播和优化器
# ============================================================================

# PyTorch版本
def pytorch_training_step():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    model = PyTorchModel(3, 10)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 前向传播
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    # 计算损失
    target = torch.randint(0, 10, (2,))
    loss = criterion(output, target)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


# Jittor版本
def jittor_training_step():
    import jittor as jt
    import jittor.nn as nn
    import jittor.optim as optim
    
    model = JittorModel(3, 10)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 前向传播
    x = jt.randn(2, 3, 224, 224)
    output = model(x)
    
    # 计算损失
    target = jt.randint(0, 10, (2,))
    loss = criterion(output, target)
    
    # Jittor的反向传播方式不同
    optimizer.step(loss)
    optimizer.zero_grad()
    
    return loss.item()


# ============================================================================
# EXAMPLE 4: 带有梯度的条件执行
# ============================================================================

# PyTorch版本
def pytorch_conditional():
    import torch
    
    x = torch.randn(3, 4, requires_grad=True)
    y = torch.randn(3, 4)
    
    # 计算条件
    with torch.no_grad():
        z = x + y  # 不计算梯度
    
    # 计算会计算梯度
    w = x + y
    loss = w.sum()
    loss.backward()
    
    return loss


# Jittor版本
def jittor_conditional():
    import jittor as jt
    
    x = jt.randn(3, 4)
    y = jt.randn(3, 4)
    
    # Jittor中使用 no_grad() 的不同方式
    # 方式1：使用with语句（如果支持）
    # with jt.no_grad():
    #     z = x + y
    
    # 方式2：停用梯度计算
    y_no_grad = y.detach() if hasattr(y, 'detach') else y.stop_gradient()
    z = x + y_no_grad
    
    # 计算会计算梯度
    w = x + y
    loss = w.sum()
    
    return loss


# ============================================================================
# EXAMPLE 5: 自定义损失函数
# ============================================================================

# PyTorch版本
class PyTorchLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        # 计算L1损失
        loss = torch.abs(predictions - targets).mean()
        return loss


# Jittor版本
class JittorLoss(jt.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        # 计算L1损失 - API相同
        loss = jt.abs(predictions - targets).mean()
        return loss


# ============================================================================
# EXAMPLE 6: 批量操作和循环
# ============================================================================

# PyTorch版本
def pytorch_batch_processing():
    import torch
    import torch.nn.functional as F
    
    batch_size = 32
    num_images = 100
    model = PyTorchModel(3, 10)
    
    losses = []
    for i in range(0, num_images, batch_size):
        # 构建批次
        x = torch.randn(batch_size, 3, 224, 224)
        targets = torch.randint(0, 10, (batch_size,))
        
        # 前向传播
        logits = model(x)
        loss = F.cross_entropy(logits, targets)
        losses.append(loss.item())
    
    avg_loss = sum(losses) / len(losses)
    return avg_loss


# Jittor版本
def jittor_batch_processing():
    import jittor as jt
    from jittor.nn import functional as F
    
    batch_size = 32
    num_images = 100
    model = JittorModel(3, 10)
    
    losses = []
    for i in range(0, num_images, batch_size):
        # 构建批次 - 相同
        x = jt.randn(batch_size, 3, 224, 224)
        targets = jt.randint(0, 10, (batch_size,))
        
        # 前向传播 - 相同API
        logits = model(x)
        loss = F.cross_entropy(logits, targets)
        losses.append(loss.item() if hasattr(loss, 'item') else float(loss))
    
    avg_loss = sum(losses) / len(losses)
    return avg_loss


# ============================================================================
# EXAMPLE 7: 权重初始化
# ============================================================================

# PyTorch版本
def pytorch_weight_init():
    import torch
    import torch.nn as nn
    
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3)
            self.fc = nn.Linear(64, 10)
            self._init_weights()
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
    
    return CustomModel()


# Jittor版本
def jittor_weight_init():
    import jittor as jt
    import jittor.nn as nn
    
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3)
            self.fc = nn.Linear(64, 10)
            self._init_weights()
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
    
    return CustomModel()


# ============================================================================
# EXAMPLE 8: 检查点保存和加载
# ============================================================================

# PyTorch版本
def pytorch_checkpoint_example():
    import torch
    
    model = PyTorchModel(3, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 保存
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 10,
    }
    torch.save(checkpoint, 'checkpoint.pth')
    
    # 加载
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Jittor版本
def jittor_checkpoint_example():
    import jittor as jt
    
    model = JittorModel(3, 10)
    optimizer = jt.optim.SGD(model.parameters(), lr=0.01)
    
    # 保存
    checkpoint = {
        'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else model,
        'optimizer_state_dict': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else optimizer,
        'epoch': 10,
    }
    jt.save(checkpoint, 'checkpoint.pkl')
    
    # 加载
    checkpoint = jt.load('checkpoint.pkl')
    if hasattr(model, 'load_state_dict'):
        model.load_state_dict(checkpoint['model_state_dict'])
    if hasattr(optimizer, 'load_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# ============================================================================
# EXAMPLE 9: 多GPU / 分布式
# ============================================================================

# PyTorch版本
def pytorch_multi_gpu():
    import torch
    import torch.nn as nn
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    
    model = PyTorchModel(3, 10)
    
    # 数据并行
    if torch.cuda.is_available():
        model = DataParallel(model)
        model = model.cuda()
    
    # 分布式数据并行
    # model = DistributedDataParallel(model, device_ids=[local_rank])


# Jittor版本
def jittor_multi_gpu():
    import jittor as jt
    
    model = JittorModel(3, 10)
    
    # Jittor自动处理GPU分布
    # 默认会在可用的GPU上运行
    # 分布式训练需要通过 jt.nn.Module 和环境变量配置


# ============================================================================
# 总结：关键区别
# ============================================================================

"""
PyTorch vs Jittor 主要区别总结：

1. 导入
   PyTorch: import torch, torch.nn, torch.optim
   Jittor:  import jittor as jt, jittor.nn as nn, jittor.optim as optim

2. 张量创建 - 基本相同
   torch.randn() → jt.randn()
   torch.zeros() → jt.zeros()
   
3. 模块定义 - 基本相同
   nn.Module → nn.Module (都继承自相同的基类)
   
4. 反向传播 - 不同！
   PyTorch:  loss.backward() + optimizer.step() + optimizer.zero_grad()
   Jittor:   optimizer.step(loss) + optimizer.zero_grad()
   
5. 设备管理 - Jittor自动处理
   PyTorch: model.to(device), x.to(device)
   Jittor:  不需要显式设备管理
   
6. 数据类型转换
   PyTorch: x.float(), x.long()
   Jittor:  x.float() 等方法可能需要验证
   
7. 无梯度计算
   PyTorch: with torch.no_grad():
   Jittor:  使用 .detach() 或其他机制
   
8. 检查点保存
   PyTorch: torch.save(), torch.load()
   Jittor:  jt.save(), jt.load()

最重要的是：大多数操作和API都是相同的！
主要区别在于反向传播的方式。
"""

if __name__ == "__main__":
    print("这个文件包含PyTorch到Jittor的迁移示例")
    print("不要直接运行这个文件，而是参考其中的代码模式")

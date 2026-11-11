# import torch
# from torch import autograd

# # =========================
# # 假设你已经有：
# #   net_x, net_y : 两个网络（分别对应 x, y）
# #   L_train(net_x, net_y, batch_train) -> 标量 loss   (g)
# #   L_val  (net_x, net_y, batch_val)   -> 标量 loss   (f)
# # =========================

# def _sqnorm(tensors):
#     s = 0.0
#     for t in tensors:
#         s = s + t.detach().pow(2).sum()
#     return s

# @torch.no_grad()
# def _perturb_params(params, vecs, scale: float):
#     for p, v in zip(params, vecs):
#         p.add_(scale * v)

# def one_step_scale_penalty_fd(
#     net_x, net_y,
#     opt_x, opt_y,
#     batch_train, batch_val,
#     L_train, L_val,
#     beta=1.0, delta=1e-12, tau0=1e-3,
#     clip_grad_norm=None,
# ):
#     """
#     一次“尺度罚 + 一阶近似(Hessian-free 差分HVP)”的关键更新：
#       h(x,y)=||∇_y g(x,y)||^2
#       theta_bal = beta * ||∇f||^2 / (||∇h||^2 + delta)
#       用中心差分近似 ∇h（需要在 y±τv 处计算 g 的梯度）
#       最终把 (∇f + theta∇h) 写入 .grad，然后 optimizer.step()
#     """

#     x_params = [p for p in net_x.parameters() if p.requires_grad]
#     y_params = [p for p in net_y.parameters() if p.requires_grad]
#     print(f"[DEBUG] #y_params = {len(y_params)}")


#     # ---- 0) 清梯度
#     opt_x.zero_grad(set_to_none=True)
#     opt_y.zero_grad(set_to_none=True)

#     # ---- 1) g = L_train, v = ∇_y g, h = ||v||^2
#     g = L_train(net_x, net_y, batch_train)
#     v = autograd.grad(g, y_params, create_graph=False, retain_graph=False, allow_unused=False)
#     h = sum((vi ** 2).sum() for vi in v)

#     # ---- 2) f = L_val, 计算 ∇f（对 x,y）
#     f = L_val(net_x, net_y, batch_val)
#     grad_f_x = autograd.grad(f, x_params, create_graph=False, retain_graph=False, allow_unused=False)
#     grad_f_y = autograd.grad(f, y_params, create_graph=False, retain_graph=False, allow_unused=False)

#     # ---- 3) 中心差分近似 ∇h（HVP-free）
#     #      τ 自适应：避免 v 很大时扰动爆炸
#     v_norm = torch.sqrt(_sqnorm(v) + 1e-30)
#     tau = float(tau0 / (v_norm + 1.0))

#     # 3.1 y <- y + τv
#     with torch.no_grad():
#         _perturb_params(y_params, v, +tau)
#     g_plus = L_train(net_x, net_y, batch_train)
#     gy_plus = autograd.grad(g_plus, y_params, create_graph=False, retain_graph=False, allow_unused=False)
#     gx_plus = autograd.grad(g_plus, x_params, create_graph=False, retain_graph=False, allow_unused=False)

#     # 3.2 y <- y - 2τv (从 plus 到 minus)
#     with torch.no_grad():
#         _perturb_params(y_params, v, -2.0 * tau)
#     g_minus = L_train(net_x, net_y, batch_train)
#     gy_minus = autograd.grad(g_minus, y_params, create_graph=False, retain_graph=False, allow_unused=False)
#     gx_minus = autograd.grad(g_minus, x_params, create_graph=False, retain_graph=False, allow_unused=False)

#     # 3.3 restore y <- y + τv (回原位)
#     with torch.no_grad():
#         _perturb_params(y_params, v, +tau)

#     # HVP 近似：∇^2_yy g · v  和  ∇^2_xy g · v
#     hvp_yy = [(gp - gm) / (2.0 * tau) for gp, gm in zip(gy_plus, gy_minus)]
#     hvp_xy = [(gp - gm) / (2.0 * tau) for gp, gm in zip(gx_plus, gx_minus)]

#     # ∇h = 2 * [ ∇^2_xy g · v ; ∇^2_yy g · v ]
#     grad_h_x = [2.0 * t for t in hvp_xy]
#     grad_h_y = [2.0 * t for t in hvp_yy]

#     # ---- 4) 尺度平衡罚权 theta_bal
#     norm_f2 = _sqnorm(grad_f_x) + _sqnorm(grad_f_y)
#     norm_h2 = _sqnorm(grad_h_x) + _sqnorm(grad_h_y)
#     theta = float(beta * norm_f2 / (norm_h2 + delta))

#     # ---- 5) 写入 .grad，然后 step()
#     # x-grad: ∇_x f + theta ∇_x h
#     for p, gf, gh in zip(x_params, grad_f_x, grad_h_x):
#         p.grad = (gf.detach() + theta * gh.detach())

#     # y-grad: ∇_y f + theta ∇_y h
#     for p, gf, gh in zip(y_params, grad_f_y, grad_h_y):
#         p.grad = (gf.detach() + theta * gh.detach())

#     # 可选：clip（对两个网络分别裁剪或合并裁剪都可）
#     if clip_grad_norm is not None:
#         torch.nn.utils.clip_grad_norm_(x_params, clip_grad_norm)
#         torch.nn.utils.clip_grad_norm_(y_params, clip_grad_norm)

#     # ---- 6) 真正更新（optimizer.step）
#     opt_x.step()
#     opt_y.step()

#     return {
#         "loss_val_f": float(f.detach().cpu()),
#         "loss_train_g": float(g.detach().cpu()),
#         "kkt_res_h": float(h.detach().cpu()),
#         "theta_bal": theta,
#         "tau_fd": tau,
#         "norm_f2": float(norm_f2.detach().cpu()),
#         "norm_h2": float(norm_h2.detach().cpu()),
#     }


# # =========================
# # 用法示例（把你的网络/数据/损失接进来）
# # =========================
# # opt_x = torch.optim.Adam(net_x.parameters(), lr=1e-4)
# # opt_y = torch.optim.Adam(net_y.parameters(), lr=1e-4)
# #
# # for it, (batch_train, batch_val) in enumerate(loader):
# #     info = one_step_scale_penalty_fd(
# #         net_x, net_y, opt_x, opt_y,
# #         batch_train, batch_val,
# #         L_train=your_train_loss_fn,
# #         L_val=your_val_loss_fn,
# #         beta=1.0, delta=1e-12, tau0=1e-3,
# #         clip_grad_norm=1.0,
# #     )
# #     if it % 50 == 0:
# #         print(it, info)
import torch
from torch import autograd


# def _sqnorm(tensors):
#     s = 0.0
#     for t in tensors:
#         s = s + t.detach().pow(2).sum()
#     return s


# def _replace_none_with_zero(grads, params):
#     out = []
#     for g, p in zip(grads, params):
#         if g is None:
#             out.append(torch.zeros_like(p))
#         else:
#             out.append(g)
#     return out


# @torch.no_grad()
# def _perturb_params(params, vecs, scale: float):
#     for p, v in zip(params, vecs):
#         p.add_(scale * v)


# import torch
# from torch import autograd

# def _sqnorm(tensors):
#     return sum(t.detach().pow(2).sum() for t in tensors)

# @torch.no_grad()
# def _perturb_params(params, vecs, scale: float):
#     for p, v in zip(params, vecs):
#         p.add_(scale * v)

# def one_step_bilevel_fd(
#     x_params, y_params,
#     opt_x, opt_y,
#     get_loss_train,
#     get_loss_val,
#     beta=1.0, delta=1e-12, tau0=1e-3,
# ):
#     # 转换为列表并过滤 requires_grad=False 的参数
#     x_params = [p for p in x_params if p.requires_grad]
#     y_params = [p for p in y_params if p.requires_grad]
#     print(f"DEBUG: x_params count = {len(x_params)}, y_params count = {len(y_params)}")

#     if len(x_params) == 0 or len(y_params) == 0:
#         raise ValueError("Check your parameter filtering! One of the parameter lists is empty.")
#     # ---- 1. 计算 g 的梯度 v = ∇_y g ----
#     g = get_loss_train()
#     # 使用 allow_unused=True 处理那些没被用到的 Adapter 层
#     v_raw = autograd.grad(g, y_params, retain_graph=False, allow_unused=True)
    
#     # 关键：识别哪些参数真正产生了梯度
#     active_y_indices = []
#     v_active = []
#     y_active = []
#     for i, grad in enumerate(v_raw):
#         if grad is not None:
#             active_y_indices.append(i)
#             v_active.append(grad.detach())
#             y_active.append(y_params[i])

#     if not v_active:
#         return {"loss_f": 0, "loss_g": g.item(), "theta": 0}

#     # ---- 2. 计算 f 的梯度 ----
#     f = get_loss_val()
#     grad_f_x_raw = autograd.grad(f, x_params, retain_graph=True, allow_unused=True)
#     grad_f_y_raw = autograd.grad(f, y_params, retain_graph=True, allow_unused=True)

#     # 安全处理：None 梯度转为 0
#     grad_f_x = [gx.detach() if gx is not None else torch.zeros_like(p) for p, gx in zip(x_params, grad_f_x_raw)]
#     grad_f_y_active = [grad_f_y_raw[i].detach() if grad_f_y_raw[i] is not None else torch.zeros_like(y_params[i]) 
#                        for i in active_y_indices]

#     # ---- 3. 有限差分计算 HVP (只针对活跃参数) ----
#     v_norm = torch.sqrt(_sqnorm(v_active) + 1e-30)
#     tau = float(tau0 / (v_norm + 1.0))

#     # y + tau * v
#     _perturb_params(y_active, v_active, tau)
#     g_plus = get_loss_train()
#     gy_plus = autograd.grad(g_plus, y_active, retain_graph=True, allow_unused=True)
#     gx_plus = autograd.grad(g_plus, x_params, retain_graph=True, allow_unused=True)

#     # y - 2 * tau * v
#     _perturb_params(y_active, v_active, -2.0 * tau)
#     g_minus = get_loss_train()
#     gy_minus = autograd.grad(g_minus, y_active, retain_graph=True, allow_unused=True)
#     gx_minus = autograd.grad(g_minus, x_params, retain_graph=True, allow_unused=True)

#     # 恢复 y
#     _perturb_params(y_active, v_active, tau)

#     # 差分近似 HVP
#     def safe_diff_div(pos, neg, p_ref):
#         if pos is not None and neg is not None:
#             return (pos.detach() - neg.detach()) / (2.0 * tau)
#         return torch.zeros_like(p_ref)

#     hvp_x = [safe_diff_div(p, n, ref) for p, n, ref in zip(gx_plus, gx_minus, x_params)]
#     hvp_y_active = [safe_diff_div(p, n, ref) for p, n, ref in zip(gy_plus, gy_minus, y_active)]

#     # ---- 4. 尺度平衡 (Theta) ----
#     grad_h_x = [2.0 * h for h in hvp_x]
#     grad_h_y_active = [2.0 * h for h in hvp_y_active]

#     norm_f2 = _sqnorm(grad_f_x) + _sqnorm(grad_f_y_active)
#     norm_h2 = _sqnorm(grad_h_x) + _sqnorm(grad_h_y_active)
    
#     # 这里的 delta 防止除以 0
#     theta = float(beta * norm_f2 / (norm_h2 + delta))

#     # ---- 5. 更新参数 ----
#     # opt_x.zero_grad(set_to_none=True)
#     # opt_y.zero_grad(set_to_none=True)

#     # for p, gf, gh in zip(x_params, grad_f_x, grad_h_x):
#     #     p.grad = gf + theta * gh
    
#     # for p, gf, gh in zip(y_active, grad_f_y_active, grad_h_y_active):
#     #     p.grad = gf + theta * gh
#     # ---- 5. 更新参数 ----
# # ---- 5. 写入梯度并确保没有 None ----
#     opt_x.zero_grad(set_to_none=True)
#     opt_y.zero_grad(set_to_none=True)

#     # 处理 x_params (上层参数)
#     for p, gf, gh in zip(x_params, grad_f_x, grad_h_x):
#         # 确保 gf 和 gh 都是 Tensor (之前的代码中应该已经转为 0 了，这里做二次保险)
#         final_grad = gf + theta * gh
#         if final_grad is None:
#             p.grad = torch.zeros_like(p)
#         else:
#             p.grad = final_grad.detach()

#     # 处理 y_params (下层参数 - 注意要处理全部 y_params，而不仅仅是 active 的)
#     # 先建立一个全 0 梯度的映射
#     y_grad_map = {id(p): torch.zeros_like(p) for p in y_params}
    
#     # 把计算出来的 active 梯度填进去
#     for p, gf, gh in zip(y_active, grad_f_y_active, grad_h_y_active):
#         y_grad_map[id(p)] = (gf + theta * gh).detach()
    
#     # 正式赋值给 y_params
#     for p in y_params:
#         p.grad = y_grad_map[id(p)]

#     # ---- 6. 健壮性检查后再 Step ----
#     # 检查是否所有管理的参数都有梯度
#     def has_grads(optimizer):
#         for group in optimizer.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     return False
#         return True

#     if has_grads(opt_x):
#         opt_x.step()
#     else:
#         print("[Warning] opt_x contains parameters without grads, skipping step.")

#     if has_grads(opt_y):
#         opt_y.step()
#     else:
#         print("[Warning] opt_y contains parameters without grads, skipping step.")

#     return {"loss_f": f.item(), "theta": theta}
import torch
from torch import autograd

def _filter_none(params, grads):
    """过滤掉 grad is None 的参数"""
    new_p, new_g = [], []
    for p, g in zip(params, grads):
        if g is not None:
            new_p.append(p)
            new_g.append(g)
    return new_p, new_g


def _sqnorm(tensors):
    s = 0.0
    for t in tensors:
        s = s + t.pow(2).sum()
    return s


@torch.no_grad()
def _perturb_params(params, vecs, scale: float):
    for p, v in zip(params, vecs):
        p.add_(scale * v)
def one_step_bilevel_fd(
    *,
    x_params,
    y_params,
    opt_x,
    opt_y,
    loss_train,
    loss_val,
    beta=1.0,
    delta=1e-12,
    tau0=1e-3,
    clip_grad_norm=None,
):
    print("\n[DEBUG] ===== requires_grad sanity =====")

    for i, p in enumerate(x_params):
        if not p.requires_grad:
            print(f"[X] idx={i} | requires_grad=False | shape={tuple(p.shape)}")
        elif not p.is_leaf:
            print(f"[X] idx={i} | NOT leaf | shape={tuple(p.shape)}")
    print("\n[DEBUG] ===== requires_grad sanity =====")
    for i, p in enumerate(y_params):
        if not p.requires_grad:
            print(f"[Y] idx={i} | requires_grad=False | shape={tuple(p.shape)}")
        elif not p.is_leaf:
            print(f"[Y] idx={i} | NOT leaf | shape={tuple(p.shape)}")
    print("\n[DEBUG] ===== requires_grad sanity =====")
    # -----------------------------------------
    # 0) 清梯度
    # -----------------------------------------
    opt_x.zero_grad(set_to_none=True)
    opt_y.zero_grad(set_to_none=True)

    # -----------------------------------------
    # 1) g = L_train, v = ∇_y g
    # -----------------------------------------
    v_all = autograd.grad(
        loss_train,
        y_params,
        allow_unused=True,
        retain_graph=True,   # 后面还要用 g 的图
        create_graph=False,
    )

    y_params_v, v = _filter_none(y_params, v_all)
    assert len(v) > 0, "∇_y g 全是 None：y 没进训练 loss"

    h = _sqnorm(v)

    # -----------------------------------------
    # 2) f = L_val, ∇f
    # -----------------------------------------
    grad_f_x_all = autograd.grad(
        loss_val,
        x_params,
        allow_unused=True,
        retain_graph=True,
        create_graph=False,
    )
    grad_f_y_all = autograd.grad(
        loss_val,
        y_params,
        allow_unused=True,
        retain_graph=True,
        create_graph=False,
    )

    x_params_f, grad_f_x = _filter_none(x_params, grad_f_x_all)
    y_params_f, grad_f_y = _filter_none(y_params, grad_f_y_all)

    # -----------------------------------------
    # 3) FD 近似 ∇h
    # -----------------------------------------
    v_norm = torch.sqrt(_sqnorm(v) + 1e-30)
    tau = float(tau0 / (v_norm.item() + 1.0))

    # y + τv
    _perturb_params(y_params_v, v, +tau)
    g_plus = loss_train.detach().clone()
    gy_plus_all = autograd.grad(g_plus, y_params, allow_unused=True)
    gx_plus_all = autograd.grad(g_plus, x_params, allow_unused=True)

    # y - τv
    _perturb_params(y_params_v, v, -2.0 * tau)
    g_minus = loss_train.detach().clone()
    gy_minus_all = autograd.grad(g_minus, y_params, allow_unused=True)
    gx_minus_all = autograd.grad(g_minus, x_params, allow_unused=True)

    # restore
    _perturb_params(y_params_v, v, +tau)

    _, gy_plus = _filter_none(y_params, gy_plus_all)
    _, gy_minus = _filter_none(y_params, gy_minus_all)
    _, gx_plus = _filter_none(x_params, gx_plus_all)
    _, gx_minus = _filter_none(x_params, gx_minus_all)

    hvp_yy = [(gp - gm) / (2.0 * tau) for gp, gm in zip(gy_plus, gy_minus)]
    hvp_xy = [(gp - gm) / (2.0 * tau) for gp, gm in zip(gx_plus, gx_minus)]

    grad_h_y = [2.0 * t for t in hvp_yy]
    grad_h_x = [2.0 * t for t in hvp_xy]

    # -----------------------------------------
    # 4) theta
    # -----------------------------------------
    norm_f2 = _sqnorm(grad_f_x) + _sqnorm(grad_f_y)
    norm_h2 = _sqnorm(grad_h_x) + _sqnorm(grad_h_y)

    theta = float(beta * norm_f2 / (norm_h2 + delta))

    # -----------------------------------------
    # 5) 写回梯度
    # -----------------------------------------
    for p, gf, gh in zip(x_params_f, grad_f_x, grad_h_x):
        p.grad = (gf + theta * gh).detach()

    for p, gf, gh in zip(y_params_f, grad_f_y, grad_h_y):
        p.grad = (gf + theta * gh).detach()

    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(x_params, clip_grad_norm)
        torch.nn.utils.clip_grad_norm_(y_params, clip_grad_norm)

    opt_x.step()
    opt_y.step()

    return {
        "loss_train_g": float(loss_train.detach()),
        "loss_val_f": float(loss_val.detach()),
        "kkt_h": float(h.detach()),
        "theta": theta,
        "tau": tau,
        "norm_f2": float(norm_f2.detach()),
        "norm_h2": float(norm_h2.detach()),
    }

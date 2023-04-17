# FIRE descent optimizer
class FireDescent(torch.optim.Optimizer):
    def __init__(self, params, dt=0.1, N_min=5, f_inc=1.1, f_dec=0.5, f_alpha=0.99, alpha_start=0.1):
        if dt < 0.0:
            raise ValueError("Invalid learning rate: {}".format(dt))
        if N_min < 0:
            raise ValueError("Invalid N_min value: {}".format(N_min))
        if f_inc < 0.0:
            raise ValueError("Invalid f_inc value: {}".format(f_inc))
        if f_dec < 0.0:
            raise ValueError("Invalid f_dec value: {}".format(f_dec))
        if f_alpha < 0.0:
            raise ValueError("Invalid f_alpha value: {}".format(f_alpha))
        if alpha_start < 0.0:
            raise ValueError("Invalid alpha_start value: {}".format(alpha_start))

        defaults = dict(dt=dt, N_min=N_min, f_inc=f_inc, f_dec=f_dec, f_alpha=f_alpha, alpha_start=alpha_start)
        super(FireDescent, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            dt = group['dt']
            N_min = group['N_min']
            f_inc = group['f_inc']
            f_dec = group['f_dec']
            f_alpha = group['f_alpha']
            alpha_start = group['alpha_start']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['v'] = torch.zeros_like(p)
                    state['alpha'] = alpha_start
                    state['dt'] = dt

                v, alpha = state['v'], state['alpha']
                state['step']
                state['step'] += 1

                # Compute force and update velocities
                F = -d_p
                P = torch.sum(F * v)
                v = (1 - alpha) * v + alpha * F / torch.norm(F) * torch.norm(v)
                if P > 0:
                    if state['step'] > N_min:
                        state['dt'] = min(state['dt'] * f_inc, dt)
                        state['alpha'] *= f_alpha
                else:
                    state['dt'] = state['dt'] * f_dec
                    state['alpha'] = alpha_start
                    state['step'] = 0
                    v = torch.zeros_like(p)

                # Update the position
                p.add_(v, alpha=state['dt'])

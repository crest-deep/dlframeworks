from chainer.backends import cuda
from chainer.functions import im2col


class FisherBlock():

    def __init__(self, link, linkname, inv_alg=None):
        self.link = link
        self.linkname = linkname
        self.inv_alg = inv_alg
        self.in_acts = None
        self.out_grads = None
        self.cov_emas = None
        self.invs = None

    def update_cov_emas(self, alpha):
        covs = self.compute_covs()
        cov_emas = self.cov_emas
        if cov_emas is not None:
            for i, cov_ema in enumerate(cov_emas):
                self.cov_emas[i] = alpha * covs[i] + (1 - alpha) * cov_ema
        else:
            self.cov_emas = covs

    def compute_covs(self):
        data = self.in_acts, self.out_grads
        xp = cuda.get_array_module(*data)
        with cuda.get_device_from_array(*data):
            return self.compute_covs_core(xp, *data)

    def compute_covs_core(self, xp, acts, grads):
        pass

    def update_invs(self, damping):
        cov_emas = self.cov_emas
        if cov_emas is None:
            return
        self.invs = [self.compute_inv_2factors(cov_ema, damping)
                     for cov_ema in cov_emas]

    # TODO add plus value (pi) for damping
    def compute_inv_2factors(self, cov_ema, damping):
        xp = cuda.get_array_module(cov_ema)
        with cuda.get_device_from_array(cov_ema):
            dmp = xp.identity(cov_ema.shape[0]) * \
              xp.sqrt(damping)
        return self.compute_inv_core(xp, cov_ema + dmp)

    def compute_inv_core(self, xp, X):
        alg = self.inv_alg
        if alg == 'cholesky':
            c = xp.linalg.inv(xp.linalg.cholesky(X))
            return xp.dot(c.T, c)
        else:
            return xp.linalg.inv(X)

    def update_kfgrads(self):
        param_W = self.link.W
        param_b = self.link.b
        invs = self.invs
        if invs is None:
            return
        kfgrads = self.compute_kfgrads(param_W, param_b, invs)
        if param_b is not None:
            param_W.kfgrad = kfgrads[:, :-1].reshape(param_W.grad.shape)
            param_b.kfgrad = kfgrads[:, -1].reshape(param_b.grad.shape)
        else:
            param_W.kfgrad = kfgrads.reshape(param_W.grad.shape)

    def compute_kfgrads(self, param_W, param_b, invs):
        data = (param_W.grad, param_b.grad, invs) \
            if param_b is not None else (param_W.grad, invs)
        xp = cuda.get_array_module(*data)
        with cuda.get_device_from_array(*data):
            return self.compute_kfgrads_core(xp, param_W, param_b, invs)

    def compute_kfgrads_core(self):
        pass

    def nobias(self):
        return self.link.b is None

    def load_data(self, in_acts, out_grads):
        assert in_acts is not None
        assert out_grads is not None
        self.in_acts = in_acts
        self.out_grads = out_grads

    def load_conv2d_args(self, conv2d, param):
        # Only for FisherBlockConv2D
        pass


class FisherBlockLinear(FisherBlock):

    def __init__(self, link, linkname):
        super(FisherBlockLinear, self).__init__(link, linkname)

    def compute_covs_core(self, xp, acts, grads):
        # Note that this method is called inside a with-statement of xp module
        n, _ = acts.shape
        if not self.nobias():
            ones = xp.ones(n)
            acts = xp.column_stack((acts, ones))
        A = acts.T.dot(acts) / n
        G = grads.T.dot(grads) / n
        return [A, G]

    def compute_kfgrads_core(self, xp, param_W, param_b, invs):
        A_inv, G_inv = invs
        grad = param_W.grad
        if param_b is not None:
            grad = xp.column_stack([grad, param_b.grad])
        return xp.dot(xp.dot(G_inv, grad), A_inv).astype(grad.dtype)


class FisherBlockConv2D(FisherBlock):

    def __init__(self, link, linkname):
        super(FisherBlockConv2D, self).__init__(link, linkname)
        self.conv2d_args = None

    def compute_covs_core(self, xp, acts, grads):
        # Note that this method is called inside a with-statement of xp module
        n, _, _, _ = acts.shape
        acts_expand = self.acts_expand_convolution_2d()
        if not self.nobias():
            ones = xp.ones(acts_expand.shape[0])
            acts_expand = xp.column_stack((acts_expand, ones))
        A = acts_expand.T.dot(acts_expand) / n
        n, _, ho, wo = grads.shape
        grads = grads.transpose(0, 2, 3, 1)
        grads = grads.reshape(n*ho*wo, -1)
        G = grads.T.dot(grads) / (n*ho*wo)
        return [A, G]

    def acts_expand_convolution_2d(self):
        acts = self.in_acts
        ksize, stride, pad = self.conv2d_args
        acts_expand = im2col(acts, ksize, stride, pad).data
        # n x c*ksize*ksize x ho x wo
        n, _, ho, wo = acts_expand.shape
        # n x ho x wo x c*ksize*ksize
        acts_expand = acts_expand.transpose(0, 2, 3, 1)
        # n*ho*wo x c*ksize*ksize
        acts_expand = acts_expand.reshape(n*ho*wo, -1)
        return acts_expand

    def compute_kfgrads_core(self, xp, param_W, param_b, invs):
        A_inv, G_inv = invs
        grad = param_W.grad
        c_o, c_i, h, w = grad.shape
        grad = grad.reshape(c_o, -1)
        if param_b is not None:
            grad = xp.column_stack([grad, param_b.grad])
        return xp.dot(xp.dot(G_inv, grad), A_inv).astype(grad.dtype)

    def load_conv2d_args(self, conv2d, param):
        stride, pad = conv2d.sy, conv2d.ph
        _, _, ksize, _ = param.data.shape
        self.conv2d_args = ksize, stride, pad


class FisherBlockBatchNorm(FisherBlock):

    def __init__(self, link, linkname):
        super(FisherBlockBatchNorm, self).__init__(link, linkname)

    def update_kfgrads(self):
        pass

import numpy as np


class PhotometricJitter:
    def __init__(self, p=0.5, brightness=0.08, contrast=0.08, seed=0):
        self.p = float(p)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        if hasattr(np.random, "default_rng"):
            self.rng = np.random.default_rng(seed)
        else:
            # Compatibility for older NumPy in the project Docker image.
            self.rng = np.random.RandomState(seed)

    def __call__(self, obs):
        # obs can be (H, W, C) or (N, H, W, C)
        if self.p <= 0.0:
            return obs, False

        if hasattr(self.rng, "random"):
            rand_val = self.rng.random()
        else:
            rand_val = self.rng.rand()

        if rand_val >= self.p:
            return obs, False

        x = obs.astype(np.float32)

        # Apply per-sample if batched, otherwise once.
        if x.ndim == 4:
            n_batch = x.shape[0]
            c = 1.0 + self.rng.uniform(
                -self.contrast, self.contrast, size=(n_batch, 1, 1, 1)
            )
            b = 255.0 * self.rng.uniform(
                -self.brightness, self.brightness, size=(n_batch, 1, 1, 1)
            )
            x = (x - 127.5) * c + 127.5 + b
        else:
            c = 1.0 + self.rng.uniform(-self.contrast, self.contrast)
            b = 255.0 * self.rng.uniform(-self.brightness, self.brightness)
            x = (x - 127.5) * c + 127.5 + b

        x = np.clip(x, 0, 255)
        return x.astype(obs.dtype), True


class ObsTransformWrapper:
    def __init__(self, env, transform_fn):
        self.env = env
        self.transform_fn = transform_fn
        self.action_space = getattr(env, "action_space", None)
        self.observation_space = getattr(env, "observation_space", None)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs2, _ = self.transform_fn(obs)
        return obs2

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs2, did = self.transform_fn(obs)

        if isinstance(info, dict):
            info["did_jitter"] = did
        elif isinstance(info, (list, tuple)):
            for one_info in info:
                if isinstance(one_info, dict):
                    one_info["did_jitter"] = did

        return obs2, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)

from dataclasses import asdict
import numpy as np
from pathlib import Path
import json, os, time
from planar_eval import methods
from planar_physics.fixtures import ROBOT_EQUAL_030 as R
from planar_learn.step_env import StepEnv


class ComparisonEnv(StepEnv):
    def __init__(self, tasks, config, seed, preview=None):
        methods.pin_threads(1)
        self.transitions, self.episodes, self.preview = 0, 0, preview
        super().__init__(tasks, R, config, seed=seed, wall_time_s=3.)

    def charge_inference(self, seconds):
        self.compute_s += float(seconds)

    def step(self, action):
        self.transitions += 1
        result = super().step(action)
        if result[2] and self.preview:
            self.episodes += 1
            if self.episodes % self.preview[1] == 0:
                self._write_preview(result[4])
        return result

    def _write_preview(self, info):
        folder, index = Path(self.preview[0]), self.preview[2]
        try:
            folder.mkdir(parents=True, exist_ok=True)
            body = dict(env=index, episode=self.episodes, transitions=self.transitions, written=time.time(),
                        seed=self.task.get('seed'), scene=asdict(self.task['scene']),
                        start=self.task['start'].tolist(), goal_xy=self.task['goal_xy'].tolist(),
                        path=[q.tolist() for q in self.path], code=info.get('code'),
                        reached=bool(info.get('reached')), steps=info.get('steps'), blocked=info.get('blocked'),
                        duration_s=info.get('duration_s'), reference_s=self.task['reference_s'],
                        margin=info.get('margin'))
            done = self.kept.get('prepared')
            if done is not None and done.path is not None:
                body['pruned'] = np.asarray(done.path).tolist()
            if done is not None and done.reference is not None:
                end = done.reference.end_time()
                t = np.append(np.arange(0., end, .002), end)
                body['curve'] = dict(t=np.round(t, 6).tolist(),
                                     q=np.round(done.reference.vector_values(t).T, 7).tolist(),
                                     dq=np.round(done.reference.MakeDerivative().vector_values(t).T, 7).tolist())
            names = [f'env{index}.json'] + ([f'env{index}-reached.json'] if body['reached'] else [])
            text = json.dumps(body)
            for name in names:
                tmp = folder/(name+'.tmp')
                tmp.write_text(text)
                os.replace(tmp, folder/name)
        except (OSError, TypeError, ValueError):
            pass

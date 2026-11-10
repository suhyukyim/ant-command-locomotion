"""수정 전(04dad9e) propulsion_analysis.Recorder 복사본. 새 Recorder가 같은 기록을 내는지 비교하는 데만 쓴다."""
import mujoco
import numpy as np

FEET = ["left_ankle_geom", "right_ankle_geom", "third_ankle_geom", "fourth_ankle_geom"]


class Recorder:
    """물리 스텝마다 발별 접촉·반력·접촉점 속도와 로봇 전체 반력을 기록."""

    def __init__(self, env, u):
        uw = env.unwrapped
        self.m, self.d, self.u = uw.model, uw.data, np.asarray(u)
        gid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.floor, self.feet = gid("floor"), [gid(n) for n in FEET]
        self.on = False
        self.rows = []   # (con[4], fz[4], fa[4], slip[4], fh[4], total_f[3])
        self._f6, self._v6 = np.zeros(6), np.zeros(6)

        def step_sim(ctrl, n_frames):          # MujocoEnv._step_mujoco_simulation과 같은 동작, 1스텝씩
            uw.data.ctrl[:] = ctrl
            for _ in range(n_frames):
                if self.on:   # Ant는 RK4: mj_step 뒤의 접촉력은 마지막 RK 단계 값이라, 스텝 시작 상태에서 다시 계산해 기록
                    mujoco.mj_forward(uw.model, uw.data)   # 궤적은 바뀌지 않음 (확인: qpos 차이 0)
                    self.sample()
                mujoco.mj_step(uw.model, uw.data)
            mujoco.mj_rnePostConstraint(uw.model, uw.data)
        uw._step_mujoco_simulation = step_sim

    def sample(self):
        m, d = self.m, self.d
        con = np.zeros(4, bool); fz = np.zeros(4); fa = np.zeros(4); fh = np.zeros(4)
        slip = np.full(4, np.nan); total = np.zeros(3)
        for i in range(d.ncon):
            c = d.contact[i]
            if self.floor not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.floor else c.geom1
            mujoco.mj_contactForce(m, d, i, self._f6)
            fw = c.frame.reshape(3, 3).T @ self._f6[:3]     # 접촉 좌표계 → 월드 (geom1이 geom2에 주는 힘)
            if c.geom2 == self.floor:                         # 로봇이 받는 힘으로
                fw = -fw
            total += fw
            if other in self.feet:
                k = self.feet.index(other)
                con[k] = True
                fz[k] += fw[2]; fa[k] += fw[:2] @ self.u; fh[k] += np.linalg.norm(fw[:2])
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, other, self._v6, 0)
                vp = self._v6[3:6] + np.cross(self._v6[:3], c.pos - d.geom_xpos[other])
                slip[k] = np.linalg.norm(vp[:2])
        self.rows.append((con, fz, fa, slip, fh, total))


